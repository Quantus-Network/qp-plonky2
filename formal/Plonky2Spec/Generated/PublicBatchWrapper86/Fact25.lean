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

theorem publicBatchWrapper86_f19003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1941 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14585)) := by
  have c118371 := (publicBatchWrapper86_copies3699 a h).2.2.2.1
  have c118372 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.1
  have c118373 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.1
  have c118374 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.1
  have c118375 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.1
  have c118376 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1941_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_2
  simp only [← c118371, ← c118372, ← c118373] at e_1941_2
  have e_1941_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_3
  simp only [← c118374, ← c118375, k2, ← c118376] at e_1941_3
  have hr := e_1941_3
  simp only [e_1941_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1941 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14586)) := by
  have c118377 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.1
  have c118378 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118379 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118380 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118381 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118382 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1941_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_4
  simp only [← c118377, ← c118378, ← c118379] at e_1941_4
  have e_1941_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_5
  simp only [← c118380, ← c118381, k2, ← c118382] at e_1941_5
  have hr := e_1941_5
  simp only [e_1941_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1941 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14587)) := by
  have c118383 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118384 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118385 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118386 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118387 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118388 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1941_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_6
  simp only [← c118383, ← c118384, ← c118385] at e_1941_6
  have e_1941_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_7
  simp only [← c118386, ← c118387, k2, ← c118388] at e_1941_7
  have hr := e_1941_7
  simp only [e_1941_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1941 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14588)) := by
  have c118389 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118390 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118391 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118392 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118393 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118394 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1941_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_8
  simp only [← c118389, ← c118390, ← c118391] at e_1941_8
  have e_1941_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_9
  simp only [← c118392, ← c118393, k2, ← c118394] at e_1941_9
  have hr := e_1941_9
  simp only [e_1941_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1941 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14589)) := by
  have c118395 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118396 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118397 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118398 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118399 := (publicBatchWrapper86_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118400 := (publicBatchWrapper86_copies3700 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1941_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_10
  simp only [← c118395, ← c118396, ← c118397] at e_1941_10
  have e_1941_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_11
  simp only [← c118398, ← c118399, k2, ← c118400] at e_1941_11
  have hr := e_1941_11
  simp only [e_1941_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1941 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14590)) := by
  have c118401 := (publicBatchWrapper86_copies3700 a h).2.1
  have c118402 := (publicBatchWrapper86_copies3700 a h).2.2.1
  have c118403 := (publicBatchWrapper86_copies3700 a h).2.2.2.1
  have c118404 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.1
  have c118405 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.1
  have c118406 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1941_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_12
  simp only [← c118401, ← c118402, ← c118403] at e_1941_12
  have e_1941_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_13
  simp only [← c118404, ← c118405, k2, ← c118406] at e_1941_13
  have hr := e_1941_13
  simp only [e_1941_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1941 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14591)) := by
  have c118407 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.1
  have c118408 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.1
  have c118409 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.1
  have c118410 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118411 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118412 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1941_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_14
  simp only [← c118407, ← c118408, ← c118409] at e_1941_14
  have e_1941_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_15
  simp only [← c118410, ← c118411, k2, ← c118412] at e_1941_15
  have hr := e_1941_15
  simp only [e_1941_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1941 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14592)) := by
  have c118413 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118414 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118415 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118416 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118417 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118418 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1941_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_16
  simp only [← c118413, ← c118414, ← c118415] at e_1941_16
  have e_1941_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_17
  simp only [← c118416, ← c118417, k2, ← c118418] at e_1941_17
  have hr := e_1941_17
  simp only [e_1941_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1941 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14593)) := by
  have c118419 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118420 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118421 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118422 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118423 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118424 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1941_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_18
  simp only [← c118419, ← c118420, ← c118421] at e_1941_18
  have e_1941_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1941 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_19
  simp only [← c118422, ← c118423, k2, ← c118424] at e_1941_19
  have hr := e_1941_19
  simp only [e_1941_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1942 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14594)) := by
  have c118425 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118426 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118427 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118428 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118429 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118430 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1942_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_0
  simp only [← c118425, ← c118426, ← c118427] at e_1942_0
  have e_1942_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_1
  simp only [← c118428, ← c118429, k2, ← c118430] at e_1942_1
  have hr := e_1942_1
  simp only [e_1942_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1942 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14595)) := by
  have c118431 := (publicBatchWrapper86_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118432 := (publicBatchWrapper86_copies3701 a h).1
  have c118433 := (publicBatchWrapper86_copies3701 a h).2.1
  have c118434 := (publicBatchWrapper86_copies3701 a h).2.2.1
  have c118435 := (publicBatchWrapper86_copies3701 a h).2.2.2.1
  have c118436 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1942_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_2
  simp only [← c118431, ← c118432, ← c118433] at e_1942_2
  have e_1942_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_3
  simp only [← c118434, ← c118435, k2, ← c118436] at e_1942_3
  have hr := e_1942_3
  simp only [e_1942_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1942 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14596)) := by
  have c118437 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.1
  have c118438 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.1
  have c118439 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.1
  have c118440 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.1
  have c118441 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.1
  have c118442 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1942_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_4
  simp only [← c118437, ← c118438, ← c118439] at e_1942_4
  have e_1942_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_5
  simp only [← c118440, ← c118441, k2, ← c118442] at e_1942_5
  have hr := e_1942_5
  simp only [e_1942_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1942 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14597)) := by
  have c118443 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118444 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118445 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118446 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118447 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118448 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1942_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_6
  simp only [← c118443, ← c118444, ← c118445] at e_1942_6
  have e_1942_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_7
  simp only [← c118446, ← c118447, k2, ← c118448] at e_1942_7
  have hr := e_1942_7
  simp only [e_1942_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1942 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14598)) := by
  have c118449 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118450 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118451 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118452 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118453 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118454 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1942_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_8
  simp only [← c118449, ← c118450, ← c118451] at e_1942_8
  have e_1942_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_9
  simp only [← c118452, ← c118453, k2, ← c118454] at e_1942_9
  have hr := e_1942_9
  simp only [e_1942_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1942 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14599)) := by
  have c118455 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118456 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118457 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118458 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118459 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118460 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1942_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_10
  simp only [← c118455, ← c118456, ← c118457] at e_1942_10
  have e_1942_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_11
  simp only [← c118458, ← c118459, k2, ← c118460] at e_1942_11
  have hr := e_1942_11
  simp only [e_1942_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1942 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14600)) := by
  have c118461 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118462 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118463 := (publicBatchWrapper86_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118464 := (publicBatchWrapper86_copies3702 a h).1
  have c118465 := (publicBatchWrapper86_copies3702 a h).2.1
  have c118466 := (publicBatchWrapper86_copies3702 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1942_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_12
  simp only [← c118461, ← c118462, ← c118463] at e_1942_12
  have e_1942_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_13
  simp only [← c118464, ← c118465, k2, ← c118466] at e_1942_13
  have hr := e_1942_13
  simp only [e_1942_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1942 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14601)) := by
  have c118467 := (publicBatchWrapper86_copies3702 a h).2.2.2.1
  have c118468 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.1
  have c118469 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.1
  have c118470 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.1
  have c118471 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.1
  have c118472 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1942_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_14
  simp only [← c118467, ← c118468, ← c118469] at e_1942_14
  have e_1942_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_15
  simp only [← c118470, ← c118471, k2, ← c118472] at e_1942_15
  have hr := e_1942_15
  simp only [e_1942_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1942 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14770)) := by
  have c118473 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.1
  have c118474 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118475 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118476 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118477 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118478 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1942_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_16
  simp only [← c118473, ← c118474, ← c118475] at e_1942_16
  have e_1942_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_17
  simp only [← c118476, ← c118477, k2, ← c118478] at e_1942_17
  have hr := e_1942_17
  simp only [e_1942_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1942 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14771)) := by
  have c118479 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118480 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118481 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118482 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118483 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118484 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1942_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_18
  simp only [← c118479, ← c118480, ← c118481] at e_1942_18
  have e_1942_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1942 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_19
  simp only [← c118482, ← c118483, k2, ← c118484] at e_1942_19
  have hr := e_1942_19
  simp only [e_1942_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1943 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14772)) := by
  have c118485 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118486 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118487 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118488 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118489 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118490 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1943_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_0
  simp only [← c118485, ← c118486, ← c118487] at e_1943_0
  have e_1943_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_1
  simp only [← c118488, ← c118489, k2, ← c118490] at e_1943_1
  have hr := e_1943_1
  simp only [e_1943_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1943 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14773)) := by
  have c118491 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118492 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118493 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118494 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118495 := (publicBatchWrapper86_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118496 := (publicBatchWrapper86_copies3703 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1943_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_2
  simp only [← c118491, ← c118492, ← c118493] at e_1943_2
  have e_1943_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_3
  simp only [← c118494, ← c118495, k2, ← c118496] at e_1943_3
  have hr := e_1943_3
  simp only [e_1943_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1943 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14774)) := by
  have c118497 := (publicBatchWrapper86_copies3703 a h).2.1
  have c118498 := (publicBatchWrapper86_copies3703 a h).2.2.1
  have c118499 := (publicBatchWrapper86_copies3703 a h).2.2.2.1
  have c118500 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.1
  have c118501 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.1
  have c118502 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1943_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_4
  simp only [← c118497, ← c118498, ← c118499] at e_1943_4
  have e_1943_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_5
  simp only [← c118500, ← c118501, k2, ← c118502] at e_1943_5
  have hr := e_1943_5
  simp only [e_1943_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1943 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14775)) := by
  have c118503 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.1
  have c118504 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.1
  have c118505 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.1
  have c118506 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118507 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118508 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1943_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_6
  simp only [← c118503, ← c118504, ← c118505] at e_1943_6
  have e_1943_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_7
  simp only [← c118506, ← c118507, k2, ← c118508] at e_1943_7
  have hr := e_1943_7
  simp only [e_1943_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1943 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14776)) := by
  have c118509 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118510 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118511 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118512 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118513 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118514 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1943_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_8
  simp only [← c118509, ← c118510, ← c118511] at e_1943_8
  have e_1943_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_9
  simp only [← c118512, ← c118513, k2, ← c118514] at e_1943_9
  have hr := e_1943_9
  simp only [e_1943_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1943 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14777)) := by
  have c118515 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118516 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118517 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118518 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118519 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118520 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1943_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_10
  simp only [← c118515, ← c118516, ← c118517] at e_1943_10
  have e_1943_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_11
  simp only [← c118518, ← c118519, k2, ← c118520] at e_1943_11
  have hr := e_1943_11
  simp only [e_1943_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1943 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14778)) := by
  have c118521 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118522 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118523 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118524 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118525 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118526 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1943_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_12
  simp only [← c118521, ← c118522, ← c118523] at e_1943_12
  have e_1943_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_13
  simp only [← c118524, ← c118525, k2, ← c118526] at e_1943_13
  have hr := e_1943_13
  simp only [e_1943_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1943 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14779)) := by
  have c118527 := (publicBatchWrapper86_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118528 := (publicBatchWrapper86_copies3704 a h).1
  have c118529 := (publicBatchWrapper86_copies3704 a h).2.1
  have c118530 := (publicBatchWrapper86_copies3704 a h).2.2.1
  have c118531 := (publicBatchWrapper86_copies3704 a h).2.2.2.1
  have c118532 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1943_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_14
  simp only [← c118527, ← c118528, ← c118529] at e_1943_14
  have e_1943_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_15
  simp only [← c118530, ← c118531, k2, ← c118532] at e_1943_15
  have hr := e_1943_15
  simp only [e_1943_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1943 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14780)) := by
  have c118533 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.1
  have c118534 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.1
  have c118535 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.1
  have c118536 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.1
  have c118537 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.1
  have c118538 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1943_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_16
  simp only [← c118533, ← c118534, ← c118535] at e_1943_16
  have e_1943_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_17
  simp only [← c118536, ← c118537, k2, ← c118538] at e_1943_17
  have hr := e_1943_17
  simp only [e_1943_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1943 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14781)) := by
  have c118539 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118540 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118541 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118542 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118543 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118544 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1943_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_18
  simp only [← c118539, ← c118540, ← c118541] at e_1943_18
  have e_1943_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1943 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_19
  simp only [← c118542, ← c118543, k2, ← c118544] at e_1943_19
  have hr := e_1943_19
  simp only [e_1943_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1944 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14782)) := by
  have c118545 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118546 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118547 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118548 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118549 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118550 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1944_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_0
  simp only [← c118545, ← c118546, ← c118547] at e_1944_0
  have e_1944_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_1
  simp only [← c118548, ← c118549, k2, ← c118550] at e_1944_1
  have hr := e_1944_1
  simp only [e_1944_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1944 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14783)) := by
  have c118551 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118552 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118553 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118554 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118555 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118556 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1944_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_2
  simp only [← c118551, ← c118552, ← c118553] at e_1944_2
  have e_1944_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_3
  simp only [← c118554, ← c118555, k2, ← c118556] at e_1944_3
  have hr := e_1944_3
  simp only [e_1944_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1944 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14784)) := by
  have c118557 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118558 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118559 := (publicBatchWrapper86_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118560 := (publicBatchWrapper86_copies3705 a h).1
  have c118561 := (publicBatchWrapper86_copies3705 a h).2.1
  have c118562 := (publicBatchWrapper86_copies3705 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1944_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_4
  simp only [← c118557, ← c118558, ← c118559] at e_1944_4
  have e_1944_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_5
  simp only [← c118560, ← c118561, k2, ← c118562] at e_1944_5
  have hr := e_1944_5
  simp only [e_1944_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1944 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14785)) := by
  have c118563 := (publicBatchWrapper86_copies3705 a h).2.2.2.1
  have c118564 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.1
  have c118565 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.1
  have c118566 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.1
  have c118567 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.1
  have c118568 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1944_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_6
  simp only [← c118563, ← c118564, ← c118565] at e_1944_6
  have e_1944_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_7
  simp only [← c118566, ← c118567, k2, ← c118568] at e_1944_7
  have hr := e_1944_7
  simp only [e_1944_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1944 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14786)) := by
  have c118569 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.1
  have c118570 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118571 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118572 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118573 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118574 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1944_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_8
  simp only [← c118569, ← c118570, ← c118571] at e_1944_8
  have e_1944_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_9
  simp only [← c118572, ← c118573, k2, ← c118574] at e_1944_9
  have hr := e_1944_9
  simp only [e_1944_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1944 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14787)) := by
  have c118575 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118576 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118577 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118578 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118579 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118580 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1944_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_10
  simp only [← c118575, ← c118576, ← c118577] at e_1944_10
  have e_1944_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_11
  simp only [← c118578, ← c118579, k2, ← c118580] at e_1944_11
  have hr := e_1944_11
  simp only [e_1944_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1944 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14788)) := by
  have c118581 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118582 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118583 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118584 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118585 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118586 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1944_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_12
  simp only [← c118581, ← c118582, ← c118583] at e_1944_12
  have e_1944_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_13
  simp only [← c118584, ← c118585, k2, ← c118586] at e_1944_13
  have hr := e_1944_13
  simp only [e_1944_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1944 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14789)) := by
  have c118587 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118588 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118589 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118590 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118591 := (publicBatchWrapper86_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118592 := (publicBatchWrapper86_copies3706 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1944_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_14
  simp only [← c118587, ← c118588, ← c118589] at e_1944_14
  have e_1944_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_15
  simp only [← c118590, ← c118591, k2, ← c118592] at e_1944_15
  have hr := e_1944_15
  simp only [e_1944_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1944 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14790)) := by
  have c118593 := (publicBatchWrapper86_copies3706 a h).2.1
  have c118594 := (publicBatchWrapper86_copies3706 a h).2.2.1
  have c118595 := (publicBatchWrapper86_copies3706 a h).2.2.2.1
  have c118596 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.1
  have c118597 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.1
  have c118598 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1944_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_16
  simp only [← c118593, ← c118594, ← c118595] at e_1944_16
  have e_1944_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_17
  simp only [← c118596, ← c118597, k2, ← c118598] at e_1944_17
  have hr := e_1944_17
  simp only [e_1944_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1944 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14791)) := by
  have c118599 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.1
  have c118600 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.1
  have c118601 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.1
  have c118602 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118603 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118604 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1944_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_18
  simp only [← c118599, ← c118600, ← c118601] at e_1944_18
  have e_1944_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1944 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_19
  simp only [← c118602, ← c118603, k2, ← c118604] at e_1944_19
  have hr := e_1944_19
  simp only [e_1944_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1945 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14792)) := by
  have c118605 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118606 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118607 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118608 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118609 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118610 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1945_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_0
  simp only [← c118605, ← c118606, ← c118607] at e_1945_0
  have e_1945_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_1
  simp only [← c118608, ← c118609, k2, ← c118610] at e_1945_1
  have hr := e_1945_1
  simp only [e_1945_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1945 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14793)) := by
  have c118611 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118612 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118613 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118614 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118615 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118616 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1945_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_2
  simp only [← c118611, ← c118612, ← c118613] at e_1945_2
  have e_1945_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_3
  simp only [← c118614, ← c118615, k2, ← c118616] at e_1945_3
  have hr := e_1945_3
  simp only [e_1945_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1945 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14794)) := by
  have c118617 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118618 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118619 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118620 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118621 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118622 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1945_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_4
  simp only [← c118617, ← c118618, ← c118619] at e_1945_4
  have e_1945_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_5
  simp only [← c118620, ← c118621, k2, ← c118622] at e_1945_5
  have hr := e_1945_5
  simp only [e_1945_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1945 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14795)) := by
  have c118623 := (publicBatchWrapper86_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118624 := (publicBatchWrapper86_copies3707 a h).1
  have c118625 := (publicBatchWrapper86_copies3707 a h).2.1
  have c118626 := (publicBatchWrapper86_copies3707 a h).2.2.1
  have c118627 := (publicBatchWrapper86_copies3707 a h).2.2.2.1
  have c118628 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1945_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_6
  simp only [← c118623, ← c118624, ← c118625] at e_1945_6
  have e_1945_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_7
  simp only [← c118626, ← c118627, k2, ← c118628] at e_1945_7
  have hr := e_1945_7
  simp only [e_1945_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1945 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14796)) := by
  have c118629 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.1
  have c118630 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.1
  have c118631 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.1
  have c118632 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.1
  have c118633 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.1
  have c118634 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1945_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_8
  simp only [← c118629, ← c118630, ← c118631] at e_1945_8
  have e_1945_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_9
  simp only [← c118632, ← c118633, k2, ← c118634] at e_1945_9
  have hr := e_1945_9
  simp only [e_1945_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1945 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14797)) := by
  have c118635 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118636 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118637 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118638 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118639 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118640 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1945_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_10
  simp only [← c118635, ← c118636, ← c118637] at e_1945_10
  have e_1945_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_11
  simp only [← c118638, ← c118639, k2, ← c118640] at e_1945_11
  have hr := e_1945_11
  simp only [e_1945_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1945 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14798)) := by
  have c118641 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118642 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118643 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118644 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118645 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118646 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1945_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_12
  simp only [← c118641, ← c118642, ← c118643] at e_1945_12
  have e_1945_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_13
  simp only [← c118644, ← c118645, k2, ← c118646] at e_1945_13
  have hr := e_1945_13
  simp only [e_1945_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1945 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14799)) := by
  have c118647 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118648 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118649 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118650 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118651 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118652 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1945_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_14
  simp only [← c118647, ← c118648, ← c118649] at e_1945_14
  have e_1945_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_15
  simp only [← c118650, ← c118651, k2, ← c118652] at e_1945_15
  have hr := e_1945_15
  simp only [e_1945_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1945 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14800)) := by
  have c118653 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118654 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118655 := (publicBatchWrapper86_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118656 := (publicBatchWrapper86_copies3708 a h).1
  have c118657 := (publicBatchWrapper86_copies3708 a h).2.1
  have c118658 := (publicBatchWrapper86_copies3708 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1945_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_16
  simp only [← c118653, ← c118654, ← c118655] at e_1945_16
  have e_1945_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_17
  simp only [← c118656, ← c118657, k2, ← c118658] at e_1945_17
  have hr := e_1945_17
  simp only [e_1945_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1945 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14801)) := by
  have c118659 := (publicBatchWrapper86_copies3708 a h).2.2.2.1
  have c118660 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.1
  have c118661 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.1
  have c118662 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.1
  have c118663 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.1
  have c118664 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1945_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_18
  simp only [← c118659, ← c118660, ← c118661] at e_1945_18
  have e_1945_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1945 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_19
  simp only [← c118662, ← c118663, k2, ← c118664] at e_1945_19
  have hr := e_1945_19
  simp only [e_1945_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1946 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14970)) := by
  have c118665 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.1
  have c118666 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118667 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118668 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118669 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118670 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1946_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_0
  simp only [← c118665, ← c118666, ← c118667] at e_1946_0
  have e_1946_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_1
  simp only [← c118668, ← c118669, k2, ← c118670] at e_1946_1
  have hr := e_1946_1
  simp only [e_1946_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1946 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14971)) := by
  have c118671 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118672 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118673 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118674 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118675 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118676 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1946_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_2
  simp only [← c118671, ← c118672, ← c118673] at e_1946_2
  have e_1946_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_3
  simp only [← c118674, ← c118675, k2, ← c118676] at e_1946_3
  have hr := e_1946_3
  simp only [e_1946_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1946 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14972)) := by
  have c118677 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118678 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118679 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118680 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118681 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118682 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1946_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_4
  simp only [← c118677, ← c118678, ← c118679] at e_1946_4
  have e_1946_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_5
  simp only [← c118680, ← c118681, k2, ← c118682] at e_1946_5
  have hr := e_1946_5
  simp only [e_1946_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1946 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14973)) := by
  have c118683 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118684 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118685 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118686 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118687 := (publicBatchWrapper86_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118688 := (publicBatchWrapper86_copies3709 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1946_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_6
  simp only [← c118683, ← c118684, ← c118685] at e_1946_6
  have e_1946_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_7
  simp only [← c118686, ← c118687, k2, ← c118688] at e_1946_7
  have hr := e_1946_7
  simp only [e_1946_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1946 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14974)) := by
  have c118689 := (publicBatchWrapper86_copies3709 a h).2.1
  have c118690 := (publicBatchWrapper86_copies3709 a h).2.2.1
  have c118691 := (publicBatchWrapper86_copies3709 a h).2.2.2.1
  have c118692 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.1
  have c118693 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.1
  have c118694 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1946_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_8
  simp only [← c118689, ← c118690, ← c118691] at e_1946_8
  have e_1946_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_9
  simp only [← c118692, ← c118693, k2, ← c118694] at e_1946_9
  have hr := e_1946_9
  simp only [e_1946_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1946 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14975)) := by
  have c118695 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.1
  have c118696 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.1
  have c118697 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.1
  have c118698 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118699 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118700 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1946_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_10
  simp only [← c118695, ← c118696, ← c118697] at e_1946_10
  have e_1946_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_11
  simp only [← c118698, ← c118699, k2, ← c118700] at e_1946_11
  have hr := e_1946_11
  simp only [e_1946_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1946 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14976)) := by
  have c118701 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118702 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118703 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118704 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118705 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118706 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1946_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_12
  simp only [← c118701, ← c118702, ← c118703] at e_1946_12
  have e_1946_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_13
  simp only [← c118704, ← c118705, k2, ← c118706] at e_1946_13
  have hr := e_1946_13
  simp only [e_1946_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1946 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14977)) := by
  have c118707 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118708 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118709 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118710 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118711 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118712 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1946_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_14
  simp only [← c118707, ← c118708, ← c118709] at e_1946_14
  have e_1946_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_15
  simp only [← c118710, ← c118711, k2, ← c118712] at e_1946_15
  have hr := e_1946_15
  simp only [e_1946_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1946 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14978)) := by
  have c118713 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118714 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118715 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118716 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118717 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118718 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1946_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_16
  simp only [← c118713, ← c118714, ← c118715] at e_1946_16
  have e_1946_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_17
  simp only [← c118716, ← c118717, k2, ← c118718] at e_1946_17
  have hr := e_1946_17
  simp only [e_1946_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1946 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14979)) := by
  have c118719 := (publicBatchWrapper86_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118720 := (publicBatchWrapper86_copies3710 a h).1
  have c118721 := (publicBatchWrapper86_copies3710 a h).2.1
  have c118722 := (publicBatchWrapper86_copies3710 a h).2.2.1
  have c118723 := (publicBatchWrapper86_copies3710 a h).2.2.2.1
  have c118724 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1946_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_18
  simp only [← c118719, ← c118720, ← c118721] at e_1946_18
  have e_1946_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1946 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_19
  simp only [← c118722, ← c118723, k2, ← c118724] at e_1946_19
  have hr := e_1946_19
  simp only [e_1946_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1947 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14980)) := by
  have c118725 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.1
  have c118726 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.1
  have c118727 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.1
  have c118728 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.1
  have c118729 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.1
  have c118730 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1947_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_0
  simp only [← c118725, ← c118726, ← c118727] at e_1947_0
  have e_1947_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_1
  simp only [← c118728, ← c118729, k2, ← c118730] at e_1947_1
  have hr := e_1947_1
  simp only [e_1947_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1947 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14981)) := by
  have c118731 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118732 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118733 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118734 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118735 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118736 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1947_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_2
  simp only [← c118731, ← c118732, ← c118733] at e_1947_2
  have e_1947_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_3
  simp only [← c118734, ← c118735, k2, ← c118736] at e_1947_3
  have hr := e_1947_3
  simp only [e_1947_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1947 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14982)) := by
  have c118737 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118738 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118739 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118740 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118741 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118742 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1947_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_4
  simp only [← c118737, ← c118738, ← c118739] at e_1947_4
  have e_1947_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_5
  simp only [← c118740, ← c118741, k2, ← c118742] at e_1947_5
  have hr := e_1947_5
  simp only [e_1947_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1947 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14983)) := by
  have c118743 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118744 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118745 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118746 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118747 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118748 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1947_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_6
  simp only [← c118743, ← c118744, ← c118745] at e_1947_6
  have e_1947_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_7
  simp only [← c118746, ← c118747, k2, ← c118748] at e_1947_7
  have hr := e_1947_7
  simp only [e_1947_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1947 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14984)) := by
  have c118749 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118750 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118751 := (publicBatchWrapper86_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118752 := (publicBatchWrapper86_copies3711 a h).1
  have c118753 := (publicBatchWrapper86_copies3711 a h).2.1
  have c118754 := (publicBatchWrapper86_copies3711 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1947_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_8
  simp only [← c118749, ← c118750, ← c118751] at e_1947_8
  have e_1947_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_9
  simp only [← c118752, ← c118753, k2, ← c118754] at e_1947_9
  have hr := e_1947_9
  simp only [e_1947_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1947 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14985)) := by
  have c118755 := (publicBatchWrapper86_copies3711 a h).2.2.2.1
  have c118756 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.1
  have c118757 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.1
  have c118758 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.1
  have c118759 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.1
  have c118760 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1947_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_10
  simp only [← c118755, ← c118756, ← c118757] at e_1947_10
  have e_1947_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_11
  simp only [← c118758, ← c118759, k2, ← c118760] at e_1947_11
  have hr := e_1947_11
  simp only [e_1947_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1947 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14986)) := by
  have c118761 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.1
  have c118762 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118763 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118764 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118765 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118766 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1947_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_12
  simp only [← c118761, ← c118762, ← c118763] at e_1947_12
  have e_1947_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_13
  simp only [← c118764, ← c118765, k2, ← c118766] at e_1947_13
  have hr := e_1947_13
  simp only [e_1947_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1947 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14987)) := by
  have c118767 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118768 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118769 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118770 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118771 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118772 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1947_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_14
  simp only [← c118767, ← c118768, ← c118769] at e_1947_14
  have e_1947_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_15
  simp only [← c118770, ← c118771, k2, ← c118772] at e_1947_15
  have hr := e_1947_15
  simp only [e_1947_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1947 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14988)) := by
  have c118773 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118774 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118775 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118776 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118777 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118778 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1947_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_16
  simp only [← c118773, ← c118774, ← c118775] at e_1947_16
  have e_1947_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_17
  simp only [← c118776, ← c118777, k2, ← c118778] at e_1947_17
  have hr := e_1947_17
  simp only [e_1947_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1947 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14989)) := by
  have c118779 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118780 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118781 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118782 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118783 := (publicBatchWrapper86_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118784 := (publicBatchWrapper86_copies3712 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1947_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_18
  simp only [← c118779, ← c118780, ← c118781] at e_1947_18
  have e_1947_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1947 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_19
  simp only [← c118782, ← c118783, k2, ← c118784] at e_1947_19
  have hr := e_1947_19
  simp only [e_1947_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1948 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14990)) := by
  have c118785 := (publicBatchWrapper86_copies3712 a h).2.1
  have c118786 := (publicBatchWrapper86_copies3712 a h).2.2.1
  have c118787 := (publicBatchWrapper86_copies3712 a h).2.2.2.1
  have c118788 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.1
  have c118789 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.1
  have c118790 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1948_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_0
  simp only [← c118785, ← c118786, ← c118787] at e_1948_0
  have e_1948_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_1
  simp only [← c118788, ← c118789, k2, ← c118790] at e_1948_1
  have hr := e_1948_1
  simp only [e_1948_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1948 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14991)) := by
  have c118791 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.1
  have c118792 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.1
  have c118793 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.1
  have c118794 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118795 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118796 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1948_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_2
  simp only [← c118791, ← c118792, ← c118793] at e_1948_2
  have e_1948_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_3
  simp only [← c118794, ← c118795, k2, ← c118796] at e_1948_3
  have hr := e_1948_3
  simp only [e_1948_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1948 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14992)) := by
  have c118797 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118798 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118799 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118800 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118801 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118802 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1948_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_4
  simp only [← c118797, ← c118798, ← c118799] at e_1948_4
  have e_1948_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_5
  simp only [← c118800, ← c118801, k2, ← c118802] at e_1948_5
  have hr := e_1948_5
  simp only [e_1948_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1948 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14993)) := by
  have c118803 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118804 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118805 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118806 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118807 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118808 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1948_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_6
  simp only [← c118803, ← c118804, ← c118805] at e_1948_6
  have e_1948_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_7
  simp only [← c118806, ← c118807, k2, ← c118808] at e_1948_7
  have hr := e_1948_7
  simp only [e_1948_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1948 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14994)) := by
  have c118809 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118810 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118811 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118812 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118813 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118814 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1948_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_8
  simp only [← c118809, ← c118810, ← c118811] at e_1948_8
  have e_1948_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_9
  simp only [← c118812, ← c118813, k2, ← c118814] at e_1948_9
  have hr := e_1948_9
  simp only [e_1948_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1948 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14995)) := by
  have c118815 := (publicBatchWrapper86_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118816 := (publicBatchWrapper86_copies3713 a h).1
  have c118817 := (publicBatchWrapper86_copies3713 a h).2.1
  have c118818 := (publicBatchWrapper86_copies3713 a h).2.2.1
  have c118819 := (publicBatchWrapper86_copies3713 a h).2.2.2.1
  have c118820 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1948_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_10
  simp only [← c118815, ← c118816, ← c118817] at e_1948_10
  have e_1948_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_11
  simp only [← c118818, ← c118819, k2, ← c118820] at e_1948_11
  have hr := e_1948_11
  simp only [e_1948_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1948 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14996)) := by
  have c118821 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.1
  have c118822 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.1
  have c118823 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.1
  have c118824 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.1
  have c118825 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.1
  have c118826 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1948_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_12
  simp only [← c118821, ← c118822, ← c118823] at e_1948_12
  have e_1948_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_13
  simp only [← c118824, ← c118825, k2, ← c118826] at e_1948_13
  have hr := e_1948_13
  simp only [e_1948_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1948 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14997)) := by
  have c118827 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118828 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118829 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118830 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118831 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118832 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1948_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_14
  simp only [← c118827, ← c118828, ← c118829] at e_1948_14
  have e_1948_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_15
  simp only [← c118830, ← c118831, k2, ← c118832] at e_1948_15
  have hr := e_1948_15
  simp only [e_1948_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1948 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14998)) := by
  have c118833 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118834 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118835 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118836 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118837 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118838 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1948_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_16
  simp only [← c118833, ← c118834, ← c118835] at e_1948_16
  have e_1948_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_17
  simp only [← c118836, ← c118837, k2, ← c118838] at e_1948_17
  have hr := e_1948_17
  simp only [e_1948_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1948 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14999)) := by
  have c118839 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118840 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118841 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118842 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118843 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118844 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1948_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_18
  simp only [← c118839, ← c118840, ← c118841] at e_1948_18
  have e_1948_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1948 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_19
  simp only [← c118842, ← c118843, k2, ← c118844] at e_1948_19
  have hr := e_1948_19
  simp only [e_1948_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1949 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 15000)) := by
  have c118845 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118846 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118847 := (publicBatchWrapper86_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118848 := (publicBatchWrapper86_copies3714 a h).1
  have c118849 := (publicBatchWrapper86_copies3714 a h).2.1
  have c118850 := (publicBatchWrapper86_copies3714 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1949_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_0
  simp only [← c118845, ← c118846, ← c118847] at e_1949_0
  have e_1949_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_1
  simp only [← c118848, ← c118849, k2, ← c118850] at e_1949_1
  have hr := e_1949_1
  simp only [e_1949_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1949 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 15001)) := by
  have c118851 := (publicBatchWrapper86_copies3714 a h).2.2.2.1
  have c118852 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.1
  have c118853 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.1
  have c118854 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.1
  have c118855 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.1
  have c118856 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1949_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_2
  simp only [← c118851, ← c118852, ← c118853] at e_1949_2
  have e_1949_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_3
  simp only [← c118854, ← c118855, k2, ← c118856] at e_1949_3
  have hr := e_1949_3
  simp only [e_1949_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1949 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15170)) := by
  have c118857 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.1
  have c118858 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118859 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118860 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118861 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118862 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1949_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_4
  simp only [← c118857, ← c118858, ← c118859] at e_1949_4
  have e_1949_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_5
  simp only [← c118860, ← c118861, k2, ← c118862] at e_1949_5
  have hr := e_1949_5
  simp only [e_1949_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1949 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15171)) := by
  have c118863 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118864 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118865 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118866 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118867 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118868 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1949_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_6
  simp only [← c118863, ← c118864, ← c118865] at e_1949_6
  have e_1949_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_7
  simp only [← c118866, ← c118867, k2, ← c118868] at e_1949_7
  have hr := e_1949_7
  simp only [e_1949_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1949 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15172)) := by
  have c118869 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118870 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118871 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118872 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118873 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118874 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1949_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_8
  simp only [← c118869, ← c118870, ← c118871] at e_1949_8
  have e_1949_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_9
  simp only [← c118872, ← c118873, k2, ← c118874] at e_1949_9
  have hr := e_1949_9
  simp only [e_1949_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1949 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15173)) := by
  have c118875 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118876 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118877 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118878 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118879 := (publicBatchWrapper86_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118880 := (publicBatchWrapper86_copies3715 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1949_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_10
  simp only [← c118875, ← c118876, ← c118877] at e_1949_10
  have e_1949_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_11
  simp only [← c118878, ← c118879, k2, ← c118880] at e_1949_11
  have hr := e_1949_11
  simp only [e_1949_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1949 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15174)) := by
  have c118881 := (publicBatchWrapper86_copies3715 a h).2.1
  have c118882 := (publicBatchWrapper86_copies3715 a h).2.2.1
  have c118883 := (publicBatchWrapper86_copies3715 a h).2.2.2.1
  have c118884 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.1
  have c118885 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.1
  have c118886 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1949_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_12
  simp only [← c118881, ← c118882, ← c118883] at e_1949_12
  have e_1949_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_13
  simp only [← c118884, ← c118885, k2, ← c118886] at e_1949_13
  have hr := e_1949_13
  simp only [e_1949_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1949 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15175)) := by
  have c118887 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.1
  have c118888 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.1
  have c118889 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.1
  have c118890 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118891 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118892 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1949_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_14
  simp only [← c118887, ← c118888, ← c118889] at e_1949_14
  have e_1949_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_15
  simp only [← c118890, ← c118891, k2, ← c118892] at e_1949_15
  have hr := e_1949_15
  simp only [e_1949_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1949 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15176)) := by
  have c118893 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118894 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118895 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118896 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118897 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118898 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1949_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_16
  simp only [← c118893, ← c118894, ← c118895] at e_1949_16
  have e_1949_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_17
  simp only [← c118896, ← c118897, k2, ← c118898] at e_1949_17
  have hr := e_1949_17
  simp only [e_1949_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1949 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15177)) := by
  have c118899 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118900 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118901 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118902 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118903 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118904 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1949_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_18
  simp only [← c118899, ← c118900, ← c118901] at e_1949_18
  have e_1949_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1949 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_19
  simp only [← c118902, ← c118903, k2, ← c118904] at e_1949_19
  have hr := e_1949_19
  simp only [e_1949_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1950 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15178)) := by
  have c118905 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118906 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118907 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118908 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118909 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118910 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1950_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_0
  simp only [← c118905, ← c118906, ← c118907] at e_1950_0
  have e_1950_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_1
  simp only [← c118908, ← c118909, k2, ← c118910] at e_1950_1
  have hr := e_1950_1
  simp only [e_1950_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1950 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15179)) := by
  have c118911 := (publicBatchWrapper86_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118912 := (publicBatchWrapper86_copies3716 a h).1
  have c118913 := (publicBatchWrapper86_copies3716 a h).2.1
  have c118914 := (publicBatchWrapper86_copies3716 a h).2.2.1
  have c118915 := (publicBatchWrapper86_copies3716 a h).2.2.2.1
  have c118916 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1950_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_2
  simp only [← c118911, ← c118912, ← c118913] at e_1950_2
  have e_1950_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_3
  simp only [← c118914, ← c118915, k2, ← c118916] at e_1950_3
  have hr := e_1950_3
  simp only [e_1950_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1950 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15180)) := by
  have c118917 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.1
  have c118918 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.1
  have c118919 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.1
  have c118920 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.1
  have c118921 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.1
  have c118922 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1950_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_4
  simp only [← c118917, ← c118918, ← c118919] at e_1950_4
  have e_1950_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_5
  simp only [← c118920, ← c118921, k2, ← c118922] at e_1950_5
  have hr := e_1950_5
  simp only [e_1950_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1950 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15181)) := by
  have c118923 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118924 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118925 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118926 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118927 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118928 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1950_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_6
  simp only [← c118923, ← c118924, ← c118925] at e_1950_6
  have e_1950_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_7
  simp only [← c118926, ← c118927, k2, ← c118928] at e_1950_7
  have hr := e_1950_7
  simp only [e_1950_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1950 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15182)) := by
  have c118929 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118930 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118931 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118932 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118933 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118934 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1950_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_8
  simp only [← c118929, ← c118930, ← c118931] at e_1950_8
  have e_1950_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_9
  simp only [← c118932, ← c118933, k2, ← c118934] at e_1950_9
  have hr := e_1950_9
  simp only [e_1950_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1950 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15183)) := by
  have c118935 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118936 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118937 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118938 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118939 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118940 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1950_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_10
  simp only [← c118935, ← c118936, ← c118937] at e_1950_10
  have e_1950_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_11
  simp only [← c118938, ← c118939, k2, ← c118940] at e_1950_11
  have hr := e_1950_11
  simp only [e_1950_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1950 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15184)) := by
  have c118941 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118942 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118943 := (publicBatchWrapper86_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118944 := (publicBatchWrapper86_copies3717 a h).1
  have c118945 := (publicBatchWrapper86_copies3717 a h).2.1
  have c118946 := (publicBatchWrapper86_copies3717 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1950_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_12
  simp only [← c118941, ← c118942, ← c118943] at e_1950_12
  have e_1950_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_13
  simp only [← c118944, ← c118945, k2, ← c118946] at e_1950_13
  have hr := e_1950_13
  simp only [e_1950_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1950 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15185)) := by
  have c118947 := (publicBatchWrapper86_copies3717 a h).2.2.2.1
  have c118948 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.1
  have c118949 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.1
  have c118950 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.1
  have c118951 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.1
  have c118952 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1950_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_14
  simp only [← c118947, ← c118948, ← c118949] at e_1950_14
  have e_1950_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_15
  simp only [← c118950, ← c118951, k2, ← c118952] at e_1950_15
  have hr := e_1950_15
  simp only [e_1950_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1950 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15186)) := by
  have c118953 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.1
  have c118954 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118955 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118956 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118957 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118958 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1950_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_16
  simp only [← c118953, ← c118954, ← c118955] at e_1950_16
  have e_1950_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_17
  simp only [← c118956, ← c118957, k2, ← c118958] at e_1950_17
  have hr := e_1950_17
  simp only [e_1950_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1950 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15187)) := by
  have c118959 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118960 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118961 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118962 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118963 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118964 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1950_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_18
  simp only [← c118959, ← c118960, ← c118961] at e_1950_18
  have e_1950_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1950 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_19
  simp only [← c118962, ← c118963, k2, ← c118964] at e_1950_19
  have hr := e_1950_19
  simp only [e_1950_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1951 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15188)) := by
  have c118965 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118966 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118967 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118968 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118969 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118970 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1951_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_0
  simp only [← c118965, ← c118966, ← c118967] at e_1951_0
  have e_1951_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_1
  simp only [← c118968, ← c118969, k2, ← c118970] at e_1951_1
  have hr := e_1951_1
  simp only [e_1951_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1951 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15189)) := by
  have c118971 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118972 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118973 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118974 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118975 := (publicBatchWrapper86_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118976 := (publicBatchWrapper86_copies3718 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1951_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_2
  simp only [← c118971, ← c118972, ← c118973] at e_1951_2
  have e_1951_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_3
  simp only [← c118974, ← c118975, k2, ← c118976] at e_1951_3
  have hr := e_1951_3
  simp only [e_1951_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1951 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15190)) := by
  have c118977 := (publicBatchWrapper86_copies3718 a h).2.1
  have c118978 := (publicBatchWrapper86_copies3718 a h).2.2.1
  have c118979 := (publicBatchWrapper86_copies3718 a h).2.2.2.1
  have c118980 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.1
  have c118981 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.1
  have c118982 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1951_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_4
  simp only [← c118977, ← c118978, ← c118979] at e_1951_4
  have e_1951_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_5
  simp only [← c118980, ← c118981, k2, ← c118982] at e_1951_5
  have hr := e_1951_5
  simp only [e_1951_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1951 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15191)) := by
  have c118983 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.1
  have c118984 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.1
  have c118985 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.1
  have c118986 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118987 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118988 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1951_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_6
  simp only [← c118983, ← c118984, ← c118985] at e_1951_6
  have e_1951_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_7
  simp only [← c118986, ← c118987, k2, ← c118988] at e_1951_7
  have hr := e_1951_7
  simp only [e_1951_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1951 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15192)) := by
  have c118989 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118990 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118991 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118992 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118993 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118994 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1951_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_8
  simp only [← c118989, ← c118990, ← c118991] at e_1951_8
  have e_1951_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_9
  simp only [← c118992, ← c118993, k2, ← c118994] at e_1951_9
  have hr := e_1951_9
  simp only [e_1951_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1951 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15193)) := by
  have c118995 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118996 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118997 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118998 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118999 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119000 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1951_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_10
  simp only [← c118995, ← c118996, ← c118997] at e_1951_10
  have e_1951_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_11
  simp only [← c118998, ← c118999, k2, ← c119000] at e_1951_11
  have hr := e_1951_11
  simp only [e_1951_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1951 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15194)) := by
  have c119001 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119002 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119003 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119004 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119005 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119006 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1951_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_12
  simp only [← c119001, ← c119002, ← c119003] at e_1951_12
  have e_1951_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_13
  simp only [← c119004, ← c119005, k2, ← c119006] at e_1951_13
  have hr := e_1951_13
  simp only [e_1951_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1951 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15195)) := by
  have c119007 := (publicBatchWrapper86_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119008 := (publicBatchWrapper86_copies3719 a h).1
  have c119009 := (publicBatchWrapper86_copies3719 a h).2.1
  have c119010 := (publicBatchWrapper86_copies3719 a h).2.2.1
  have c119011 := (publicBatchWrapper86_copies3719 a h).2.2.2.1
  have c119012 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1951_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_14
  simp only [← c119007, ← c119008, ← c119009] at e_1951_14
  have e_1951_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_15
  simp only [← c119010, ← c119011, k2, ← c119012] at e_1951_15
  have hr := e_1951_15
  simp only [e_1951_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1951 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15196)) := by
  have c119013 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.1
  have c119014 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.1
  have c119015 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.1
  have c119016 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.1
  have c119017 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.1
  have c119018 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1951_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_16
  simp only [← c119013, ← c119014, ← c119015] at e_1951_16
  have e_1951_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_17
  simp only [← c119016, ← c119017, k2, ← c119018] at e_1951_17
  have hr := e_1951_17
  simp only [e_1951_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1951 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15197)) := by
  have c119019 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119020 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119021 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119022 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119023 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119024 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1951_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_18
  simp only [← c119019, ← c119020, ← c119021] at e_1951_18
  have e_1951_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1951 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_19
  simp only [← c119022, ← c119023, k2, ← c119024] at e_1951_19
  have hr := e_1951_19
  simp only [e_1951_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1952 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15198)) := by
  have c119025 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119026 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119027 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119028 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119029 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119030 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1952_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_0
  simp only [← c119025, ← c119026, ← c119027] at e_1952_0
  have e_1952_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_1
  simp only [← c119028, ← c119029, k2, ← c119030] at e_1952_1
  have hr := e_1952_1
  simp only [e_1952_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1952 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15199)) := by
  have c119031 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119032 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119033 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119034 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119035 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119036 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1952_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_2
  simp only [← c119031, ← c119032, ← c119033] at e_1952_2
  have e_1952_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_3
  simp only [← c119034, ← c119035, k2, ← c119036] at e_1952_3
  have hr := e_1952_3
  simp only [e_1952_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1952 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15200)) := by
  have c119037 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119038 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119039 := (publicBatchWrapper86_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119040 := (publicBatchWrapper86_copies3720 a h).1
  have c119041 := (publicBatchWrapper86_copies3720 a h).2.1
  have c119042 := (publicBatchWrapper86_copies3720 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1952_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_4
  simp only [← c119037, ← c119038, ← c119039] at e_1952_4
  have e_1952_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_5
  simp only [← c119040, ← c119041, k2, ← c119042] at e_1952_5
  have hr := e_1952_5
  simp only [e_1952_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1952 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15201)) := by
  have c119043 := (publicBatchWrapper86_copies3720 a h).2.2.2.1
  have c119044 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.1
  have c119045 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.1
  have c119046 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.1
  have c119047 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.1
  have c119048 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1952_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_6
  simp only [← c119043, ← c119044, ← c119045] at e_1952_6
  have e_1952_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_7
  simp only [← c119046, ← c119047, k2, ← c119048] at e_1952_7
  have hr := e_1952_7
  simp only [e_1952_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1952 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15370)) := by
  have c119049 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.1
  have c119050 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119051 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119052 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119053 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119054 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1952_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_8
  simp only [← c119049, ← c119050, ← c119051] at e_1952_8
  have e_1952_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_9
  simp only [← c119052, ← c119053, k2, ← c119054] at e_1952_9
  have hr := e_1952_9
  simp only [e_1952_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1952 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15371)) := by
  have c119055 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119056 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119057 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119058 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119059 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119060 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1952_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_10
  simp only [← c119055, ← c119056, ← c119057] at e_1952_10
  have e_1952_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_11
  simp only [← c119058, ← c119059, k2, ← c119060] at e_1952_11
  have hr := e_1952_11
  simp only [e_1952_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1952 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15372)) := by
  have c119061 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119062 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119063 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119064 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119065 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119066 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1952_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_12
  simp only [← c119061, ← c119062, ← c119063] at e_1952_12
  have e_1952_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_13
  simp only [← c119064, ← c119065, k2, ← c119066] at e_1952_13
  have hr := e_1952_13
  simp only [e_1952_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1952 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15373)) := by
  have c119067 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119068 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119069 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119070 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119071 := (publicBatchWrapper86_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119072 := (publicBatchWrapper86_copies3721 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1952_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_14
  simp only [← c119067, ← c119068, ← c119069] at e_1952_14
  have e_1952_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_15
  simp only [← c119070, ← c119071, k2, ← c119072] at e_1952_15
  have hr := e_1952_15
  simp only [e_1952_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1952 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15374)) := by
  have c119073 := (publicBatchWrapper86_copies3721 a h).2.1
  have c119074 := (publicBatchWrapper86_copies3721 a h).2.2.1
  have c119075 := (publicBatchWrapper86_copies3721 a h).2.2.2.1
  have c119076 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.1
  have c119077 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.1
  have c119078 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1952_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_16
  simp only [← c119073, ← c119074, ← c119075] at e_1952_16
  have e_1952_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_17
  simp only [← c119076, ← c119077, k2, ← c119078] at e_1952_17
  have hr := e_1952_17
  simp only [e_1952_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1952 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15375)) := by
  have c119079 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.1
  have c119080 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.1
  have c119081 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.1
  have c119082 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119083 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119084 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1952_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_18
  simp only [← c119079, ← c119080, ← c119081] at e_1952_18
  have e_1952_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1952 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_19
  simp only [← c119082, ← c119083, k2, ← c119084] at e_1952_19
  have hr := e_1952_19
  simp only [e_1952_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1953 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15376)) := by
  have c119085 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119086 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119087 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119088 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119089 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119090 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1953_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_0
  simp only [← c119085, ← c119086, ← c119087] at e_1953_0
  have e_1953_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_1
  simp only [← c119088, ← c119089, k2, ← c119090] at e_1953_1
  have hr := e_1953_1
  simp only [e_1953_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1953 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15377)) := by
  have c119091 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119092 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119093 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119094 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119095 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119096 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1953_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_2
  simp only [← c119091, ← c119092, ← c119093] at e_1953_2
  have e_1953_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_3
  simp only [← c119094, ← c119095, k2, ← c119096] at e_1953_3
  have hr := e_1953_3
  simp only [e_1953_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1953 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15378)) := by
  have c119097 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119098 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119099 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119100 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119101 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119102 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1953_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_4
  simp only [← c119097, ← c119098, ← c119099] at e_1953_4
  have e_1953_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_5
  simp only [← c119100, ← c119101, k2, ← c119102] at e_1953_5
  have hr := e_1953_5
  simp only [e_1953_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1953 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15379)) := by
  have c119103 := (publicBatchWrapper86_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119104 := (publicBatchWrapper86_copies3722 a h).1
  have c119105 := (publicBatchWrapper86_copies3722 a h).2.1
  have c119106 := (publicBatchWrapper86_copies3722 a h).2.2.1
  have c119107 := (publicBatchWrapper86_copies3722 a h).2.2.2.1
  have c119108 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1953_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_6
  simp only [← c119103, ← c119104, ← c119105] at e_1953_6
  have e_1953_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_7
  simp only [← c119106, ← c119107, k2, ← c119108] at e_1953_7
  have hr := e_1953_7
  simp only [e_1953_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1953 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15380)) := by
  have c119109 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.1
  have c119110 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.1
  have c119111 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.1
  have c119112 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.1
  have c119113 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.1
  have c119114 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1953_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_8
  simp only [← c119109, ← c119110, ← c119111] at e_1953_8
  have e_1953_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_9
  simp only [← c119112, ← c119113, k2, ← c119114] at e_1953_9
  have hr := e_1953_9
  simp only [e_1953_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1953 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15381)) := by
  have c119115 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119116 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119117 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119118 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119119 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119120 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1953_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_10
  simp only [← c119115, ← c119116, ← c119117] at e_1953_10
  have e_1953_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_11
  simp only [← c119118, ← c119119, k2, ← c119120] at e_1953_11
  have hr := e_1953_11
  simp only [e_1953_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1953 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15382)) := by
  have c119121 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119122 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119123 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119124 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119125 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119126 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1953_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_12
  simp only [← c119121, ← c119122, ← c119123] at e_1953_12
  have e_1953_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_13
  simp only [← c119124, ← c119125, k2, ← c119126] at e_1953_13
  have hr := e_1953_13
  simp only [e_1953_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1953 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15383)) := by
  have c119127 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119128 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119129 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119130 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119131 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119132 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1953_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_14
  simp only [← c119127, ← c119128, ← c119129] at e_1953_14
  have e_1953_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_15
  simp only [← c119130, ← c119131, k2, ← c119132] at e_1953_15
  have hr := e_1953_15
  simp only [e_1953_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1953 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15384)) := by
  have c119133 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119134 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119135 := (publicBatchWrapper86_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119136 := (publicBatchWrapper86_copies3723 a h).1
  have c119137 := (publicBatchWrapper86_copies3723 a h).2.1
  have c119138 := (publicBatchWrapper86_copies3723 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1953_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_16
  simp only [← c119133, ← c119134, ← c119135] at e_1953_16
  have e_1953_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_17
  simp only [← c119136, ← c119137, k2, ← c119138] at e_1953_17
  have hr := e_1953_17
  simp only [e_1953_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1953 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15385)) := by
  have c119139 := (publicBatchWrapper86_copies3723 a h).2.2.2.1
  have c119140 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.1
  have c119141 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.1
  have c119142 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.1
  have c119143 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.1
  have c119144 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1953_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_18
  simp only [← c119139, ← c119140, ← c119141] at e_1953_18
  have e_1953_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1953 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_19
  simp only [← c119142, ← c119143, k2, ← c119144] at e_1953_19
  have hr := e_1953_19
  simp only [e_1953_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1954 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15386)) := by
  have c119145 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.1
  have c119146 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119147 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119148 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119149 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119150 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1954_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_0
  simp only [← c119145, ← c119146, ← c119147] at e_1954_0
  have e_1954_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_1
  simp only [← c119148, ← c119149, k2, ← c119150] at e_1954_1
  have hr := e_1954_1
  simp only [e_1954_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1954 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15387)) := by
  have c119151 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119152 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119153 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119154 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119155 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119156 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1954_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_2
  simp only [← c119151, ← c119152, ← c119153] at e_1954_2
  have e_1954_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_3
  simp only [← c119154, ← c119155, k2, ← c119156] at e_1954_3
  have hr := e_1954_3
  simp only [e_1954_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1954 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15388)) := by
  have c119157 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119158 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119159 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119160 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119161 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119162 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1954_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_4
  simp only [← c119157, ← c119158, ← c119159] at e_1954_4
  have e_1954_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_5
  simp only [← c119160, ← c119161, k2, ← c119162] at e_1954_5
  have hr := e_1954_5
  simp only [e_1954_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1954 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15389)) := by
  have c119163 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119164 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119165 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119166 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119167 := (publicBatchWrapper86_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119168 := (publicBatchWrapper86_copies3724 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1954_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_6
  simp only [← c119163, ← c119164, ← c119165] at e_1954_6
  have e_1954_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_7
  simp only [← c119166, ← c119167, k2, ← c119168] at e_1954_7
  have hr := e_1954_7
  simp only [e_1954_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1954 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15390)) := by
  have c119169 := (publicBatchWrapper86_copies3724 a h).2.1
  have c119170 := (publicBatchWrapper86_copies3724 a h).2.2.1
  have c119171 := (publicBatchWrapper86_copies3724 a h).2.2.2.1
  have c119172 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.1
  have c119173 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.1
  have c119174 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1954_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_8
  simp only [← c119169, ← c119170, ← c119171] at e_1954_8
  have e_1954_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_9
  simp only [← c119172, ← c119173, k2, ← c119174] at e_1954_9
  have hr := e_1954_9
  simp only [e_1954_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1954 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15391)) := by
  have c119175 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.1
  have c119176 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.1
  have c119177 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.1
  have c119178 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119179 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119180 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1954_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_10
  simp only [← c119175, ← c119176, ← c119177] at e_1954_10
  have e_1954_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_11
  simp only [← c119178, ← c119179, k2, ← c119180] at e_1954_11
  have hr := e_1954_11
  simp only [e_1954_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1954 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15392)) := by
  have c119181 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119182 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119183 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119184 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119185 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119186 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1954_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_12
  simp only [← c119181, ← c119182, ← c119183] at e_1954_12
  have e_1954_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_13
  simp only [← c119184, ← c119185, k2, ← c119186] at e_1954_13
  have hr := e_1954_13
  simp only [e_1954_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1954 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15393)) := by
  have c119187 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119188 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119189 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119190 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119191 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119192 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1954_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_14
  simp only [← c119187, ← c119188, ← c119189] at e_1954_14
  have e_1954_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_15
  simp only [← c119190, ← c119191, k2, ← c119192] at e_1954_15
  have hr := e_1954_15
  simp only [e_1954_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1954 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15394)) := by
  have c119193 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119194 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119195 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119196 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119197 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119198 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1954_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_16
  simp only [← c119193, ← c119194, ← c119195] at e_1954_16
  have e_1954_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_17
  simp only [← c119196, ← c119197, k2, ← c119198] at e_1954_17
  have hr := e_1954_17
  simp only [e_1954_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1954 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15395)) := by
  have c119199 := (publicBatchWrapper86_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119200 := (publicBatchWrapper86_copies3725 a h).1
  have c119201 := (publicBatchWrapper86_copies3725 a h).2.1
  have c119202 := (publicBatchWrapper86_copies3725 a h).2.2.1
  have c119203 := (publicBatchWrapper86_copies3725 a h).2.2.2.1
  have c119204 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1954_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_18
  simp only [← c119199, ← c119200, ← c119201] at e_1954_18
  have e_1954_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1954 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_19
  simp only [← c119202, ← c119203, k2, ← c119204] at e_1954_19
  have hr := e_1954_19
  simp only [e_1954_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1955 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15396)) := by
  have c119205 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.1
  have c119206 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.1
  have c119207 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.1
  have c119208 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.1
  have c119209 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.1
  have c119210 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1955_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_0
  simp only [← c119205, ← c119206, ← c119207] at e_1955_0
  have e_1955_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_1
  simp only [← c119208, ← c119209, k2, ← c119210] at e_1955_1
  have hr := e_1955_1
  simp only [e_1955_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1955 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15397)) := by
  have c119211 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119212 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119213 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119214 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119215 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119216 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1955_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_2
  simp only [← c119211, ← c119212, ← c119213] at e_1955_2
  have e_1955_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_3
  simp only [← c119214, ← c119215, k2, ← c119216] at e_1955_3
  have hr := e_1955_3
  simp only [e_1955_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1955 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15398)) := by
  have c119217 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119218 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119219 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119220 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119221 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119222 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1955_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_4
  simp only [← c119217, ← c119218, ← c119219] at e_1955_4
  have e_1955_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_5
  simp only [← c119220, ← c119221, k2, ← c119222] at e_1955_5
  have hr := e_1955_5
  simp only [e_1955_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1955 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15399)) := by
  have c119223 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119224 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119225 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119226 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119227 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119228 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1955_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_6
  simp only [← c119223, ← c119224, ← c119225] at e_1955_6
  have e_1955_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_7
  simp only [← c119226, ← c119227, k2, ← c119228] at e_1955_7
  have hr := e_1955_7
  simp only [e_1955_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1955 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15400)) := by
  have c119229 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119230 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119231 := (publicBatchWrapper86_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119232 := (publicBatchWrapper86_copies3726 a h).1
  have c119233 := (publicBatchWrapper86_copies3726 a h).2.1
  have c119234 := (publicBatchWrapper86_copies3726 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1955_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_8
  simp only [← c119229, ← c119230, ← c119231] at e_1955_8
  have e_1955_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_9
  simp only [← c119232, ← c119233, k2, ← c119234] at e_1955_9
  have hr := e_1955_9
  simp only [e_1955_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1955 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15401)) := by
  have c119235 := (publicBatchWrapper86_copies3726 a h).2.2.2.1
  have c119236 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.1
  have c119237 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.1
  have c119238 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.1
  have c119239 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.1
  have c119240 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1955_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_10
  simp only [← c119235, ← c119236, ← c119237] at e_1955_10
  have e_1955_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_11
  simp only [← c119238, ← c119239, k2, ← c119240] at e_1955_11
  have hr := e_1955_11
  simp only [e_1955_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1955 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15570)) := by
  have c119241 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.1
  have c119242 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119243 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119244 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119245 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119246 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1955_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_12
  simp only [← c119241, ← c119242, ← c119243] at e_1955_12
  have e_1955_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_13
  simp only [← c119244, ← c119245, k2, ← c119246] at e_1955_13
  have hr := e_1955_13
  simp only [e_1955_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1955 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15571)) := by
  have c119247 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119248 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119249 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119250 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119251 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119252 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1955_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_14
  simp only [← c119247, ← c119248, ← c119249] at e_1955_14
  have e_1955_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_15
  simp only [← c119250, ← c119251, k2, ← c119252] at e_1955_15
  have hr := e_1955_15
  simp only [e_1955_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1955 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15572)) := by
  have c119253 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119254 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119255 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119256 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119257 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119258 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1955_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_16
  simp only [← c119253, ← c119254, ← c119255] at e_1955_16
  have e_1955_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_17
  simp only [← c119256, ← c119257, k2, ← c119258] at e_1955_17
  have hr := e_1955_17
  simp only [e_1955_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1955 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15573)) := by
  have c119259 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119260 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119261 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119262 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119263 := (publicBatchWrapper86_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119264 := (publicBatchWrapper86_copies3727 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1955_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_18
  simp only [← c119259, ← c119260, ← c119261] at e_1955_18
  have e_1955_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1955 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_19
  simp only [← c119262, ← c119263, k2, ← c119264] at e_1955_19
  have hr := e_1955_19
  simp only [e_1955_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1956 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15574)) := by
  have c119265 := (publicBatchWrapper86_copies3727 a h).2.1
  have c119266 := (publicBatchWrapper86_copies3727 a h).2.2.1
  have c119267 := (publicBatchWrapper86_copies3727 a h).2.2.2.1
  have c119268 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.1
  have c119269 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.1
  have c119270 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1956_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_0
  simp only [← c119265, ← c119266, ← c119267] at e_1956_0
  have e_1956_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_1
  simp only [← c119268, ← c119269, k2, ← c119270] at e_1956_1
  have hr := e_1956_1
  simp only [e_1956_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1956 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15575)) := by
  have c119271 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.1
  have c119272 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.1
  have c119273 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.1
  have c119274 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119275 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119276 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1956_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_2
  simp only [← c119271, ← c119272, ← c119273] at e_1956_2
  have e_1956_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_3
  simp only [← c119274, ← c119275, k2, ← c119276] at e_1956_3
  have hr := e_1956_3
  simp only [e_1956_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1956 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15576)) := by
  have c119277 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119278 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119279 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119280 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119281 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119282 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1956_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_4
  simp only [← c119277, ← c119278, ← c119279] at e_1956_4
  have e_1956_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_5
  simp only [← c119280, ← c119281, k2, ← c119282] at e_1956_5
  have hr := e_1956_5
  simp only [e_1956_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1956 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15577)) := by
  have c119283 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119284 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119285 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119286 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119287 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119288 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1956_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_6
  simp only [← c119283, ← c119284, ← c119285] at e_1956_6
  have e_1956_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_7
  simp only [← c119286, ← c119287, k2, ← c119288] at e_1956_7
  have hr := e_1956_7
  simp only [e_1956_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1956 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15578)) := by
  have c119289 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119290 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119291 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119292 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119293 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119294 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1956_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_8
  simp only [← c119289, ← c119290, ← c119291] at e_1956_8
  have e_1956_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_9
  simp only [← c119292, ← c119293, k2, ← c119294] at e_1956_9
  have hr := e_1956_9
  simp only [e_1956_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1956 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15579)) := by
  have c119295 := (publicBatchWrapper86_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119296 := (publicBatchWrapper86_copies3728 a h).1
  have c119297 := (publicBatchWrapper86_copies3728 a h).2.1
  have c119298 := (publicBatchWrapper86_copies3728 a h).2.2.1
  have c119299 := (publicBatchWrapper86_copies3728 a h).2.2.2.1
  have c119300 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1956_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_10
  simp only [← c119295, ← c119296, ← c119297] at e_1956_10
  have e_1956_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_11
  simp only [← c119298, ← c119299, k2, ← c119300] at e_1956_11
  have hr := e_1956_11
  simp only [e_1956_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1956 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15580)) := by
  have c119301 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.1
  have c119302 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.1
  have c119303 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.1
  have c119304 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.1
  have c119305 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.1
  have c119306 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1956_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_12
  simp only [← c119301, ← c119302, ← c119303] at e_1956_12
  have e_1956_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_13
  simp only [← c119304, ← c119305, k2, ← c119306] at e_1956_13
  have hr := e_1956_13
  simp only [e_1956_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1956 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15581)) := by
  have c119307 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119308 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119309 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119310 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119311 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119312 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1956_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_14
  simp only [← c119307, ← c119308, ← c119309] at e_1956_14
  have e_1956_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_15
  simp only [← c119310, ← c119311, k2, ← c119312] at e_1956_15
  have hr := e_1956_15
  simp only [e_1956_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1956 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15582)) := by
  have c119313 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119314 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119315 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119316 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119317 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119318 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1956_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_16
  simp only [← c119313, ← c119314, ← c119315] at e_1956_16
  have e_1956_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_17
  simp only [← c119316, ← c119317, k2, ← c119318] at e_1956_17
  have hr := e_1956_17
  simp only [e_1956_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1956 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15583)) := by
  have c119319 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119320 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119321 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119322 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119323 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119324 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1956_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_18
  simp only [← c119319, ← c119320, ← c119321] at e_1956_18
  have e_1956_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1956 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_19
  simp only [← c119322, ← c119323, k2, ← c119324] at e_1956_19
  have hr := e_1956_19
  simp only [e_1956_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1957 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15584)) := by
  have c119325 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119326 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119327 := (publicBatchWrapper86_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119328 := (publicBatchWrapper86_copies3729 a h).1
  have c119329 := (publicBatchWrapper86_copies3729 a h).2.1
  have c119330 := (publicBatchWrapper86_copies3729 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1957_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_0
  simp only [← c119325, ← c119326, ← c119327] at e_1957_0
  have e_1957_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_1
  simp only [← c119328, ← c119329, k2, ← c119330] at e_1957_1
  have hr := e_1957_1
  simp only [e_1957_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1957 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15585)) := by
  have c119331 := (publicBatchWrapper86_copies3729 a h).2.2.2.1
  have c119332 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.1
  have c119333 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.1
  have c119334 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.1
  have c119335 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.1
  have c119336 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1957_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_2
  simp only [← c119331, ← c119332, ← c119333] at e_1957_2
  have e_1957_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_3
  simp only [← c119334, ← c119335, k2, ← c119336] at e_1957_3
  have hr := e_1957_3
  simp only [e_1957_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1957 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15586)) := by
  have c119337 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.1
  have c119338 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119339 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119340 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119341 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119342 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1957_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_4
  simp only [← c119337, ← c119338, ← c119339] at e_1957_4
  have e_1957_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_5
  simp only [← c119340, ← c119341, k2, ← c119342] at e_1957_5
  have hr := e_1957_5
  simp only [e_1957_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1957 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15587)) := by
  have c119343 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119344 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119345 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119346 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119347 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119348 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1957_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_6
  simp only [← c119343, ← c119344, ← c119345] at e_1957_6
  have e_1957_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_7
  simp only [← c119346, ← c119347, k2, ← c119348] at e_1957_7
  have hr := e_1957_7
  simp only [e_1957_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1957 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15588)) := by
  have c119349 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119350 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119351 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119352 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119353 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119354 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1957_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_8
  simp only [← c119349, ← c119350, ← c119351] at e_1957_8
  have e_1957_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_9
  simp only [← c119352, ← c119353, k2, ← c119354] at e_1957_9
  have hr := e_1957_9
  simp only [e_1957_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1957 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15589)) := by
  have c119355 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119356 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119357 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119358 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119359 := (publicBatchWrapper86_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119360 := (publicBatchWrapper86_copies3730 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1957_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_10
  simp only [← c119355, ← c119356, ← c119357] at e_1957_10
  have e_1957_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_11
  simp only [← c119358, ← c119359, k2, ← c119360] at e_1957_11
  have hr := e_1957_11
  simp only [e_1957_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1957 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15590)) := by
  have c119361 := (publicBatchWrapper86_copies3730 a h).2.1
  have c119362 := (publicBatchWrapper86_copies3730 a h).2.2.1
  have c119363 := (publicBatchWrapper86_copies3730 a h).2.2.2.1
  have c119364 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.1
  have c119365 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.1
  have c119366 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1957_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_12
  simp only [← c119361, ← c119362, ← c119363] at e_1957_12
  have e_1957_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_13
  simp only [← c119364, ← c119365, k2, ← c119366] at e_1957_13
  have hr := e_1957_13
  simp only [e_1957_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1957 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15591)) := by
  have c119367 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.1
  have c119368 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.1
  have c119369 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.1
  have c119370 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119371 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119372 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1957_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_14
  simp only [← c119367, ← c119368, ← c119369] at e_1957_14
  have e_1957_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_15
  simp only [← c119370, ← c119371, k2, ← c119372] at e_1957_15
  have hr := e_1957_15
  simp only [e_1957_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1957 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15592)) := by
  have c119373 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119374 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119375 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119376 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119377 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119378 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1957_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_16
  simp only [← c119373, ← c119374, ← c119375] at e_1957_16
  have e_1957_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_17
  simp only [← c119376, ← c119377, k2, ← c119378] at e_1957_17
  have hr := e_1957_17
  simp only [e_1957_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1957 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15593)) := by
  have c119379 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119380 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119381 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119382 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119383 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119384 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1957_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_18
  simp only [← c119379, ← c119380, ← c119381] at e_1957_18
  have e_1957_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1957 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_19
  simp only [← c119382, ← c119383, k2, ← c119384] at e_1957_19
  have hr := e_1957_19
  simp only [e_1957_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1958 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15594)) := by
  have c119385 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119386 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119387 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119388 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119389 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119390 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1958_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_0
  simp only [← c119385, ← c119386, ← c119387] at e_1958_0
  have e_1958_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_1
  simp only [← c119388, ← c119389, k2, ← c119390] at e_1958_1
  have hr := e_1958_1
  simp only [e_1958_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1958 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15595)) := by
  have c119391 := (publicBatchWrapper86_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119392 := (publicBatchWrapper86_copies3731 a h).1
  have c119393 := (publicBatchWrapper86_copies3731 a h).2.1
  have c119394 := (publicBatchWrapper86_copies3731 a h).2.2.1
  have c119395 := (publicBatchWrapper86_copies3731 a h).2.2.2.1
  have c119396 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1958_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_2
  simp only [← c119391, ← c119392, ← c119393] at e_1958_2
  have e_1958_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_3
  simp only [← c119394, ← c119395, k2, ← c119396] at e_1958_3
  have hr := e_1958_3
  simp only [e_1958_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1958 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15596)) := by
  have c119397 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.1
  have c119398 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.1
  have c119399 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.1
  have c119400 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.1
  have c119401 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.1
  have c119402 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1958_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_4
  simp only [← c119397, ← c119398, ← c119399] at e_1958_4
  have e_1958_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_5
  simp only [← c119400, ← c119401, k2, ← c119402] at e_1958_5
  have hr := e_1958_5
  simp only [e_1958_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1958 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15597)) := by
  have c119403 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119404 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119405 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119406 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119407 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119408 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1958_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_6
  simp only [← c119403, ← c119404, ← c119405] at e_1958_6
  have e_1958_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_7
  simp only [← c119406, ← c119407, k2, ← c119408] at e_1958_7
  have hr := e_1958_7
  simp only [e_1958_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1958 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15598)) := by
  have c119409 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119410 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119411 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119412 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119413 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119414 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1958_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_8
  simp only [← c119409, ← c119410, ← c119411] at e_1958_8
  have e_1958_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_9
  simp only [← c119412, ← c119413, k2, ← c119414] at e_1958_9
  have hr := e_1958_9
  simp only [e_1958_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1958 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15599)) := by
  have c119415 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119416 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119417 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119418 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119419 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119420 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1958_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_10
  simp only [← c119415, ← c119416, ← c119417] at e_1958_10
  have e_1958_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_11
  simp only [← c119418, ← c119419, k2, ← c119420] at e_1958_11
  have hr := e_1958_11
  simp only [e_1958_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1958 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15600)) := by
  have c119421 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119422 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119423 := (publicBatchWrapper86_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119424 := (publicBatchWrapper86_copies3732 a h).1
  have c119425 := (publicBatchWrapper86_copies3732 a h).2.1
  have c119426 := (publicBatchWrapper86_copies3732 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1958_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_12
  simp only [← c119421, ← c119422, ← c119423] at e_1958_12
  have e_1958_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_13
  simp only [← c119424, ← c119425, k2, ← c119426] at e_1958_13
  have hr := e_1958_13
  simp only [e_1958_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1958 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15601)) := by
  have c119427 := (publicBatchWrapper86_copies3732 a h).2.2.2.1
  have c119428 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.1
  have c119429 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.1
  have c119430 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.1
  have c119431 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.1
  have c119432 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1958_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_14
  simp only [← c119427, ← c119428, ← c119429] at e_1958_14
  have e_1958_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_15
  simp only [← c119430, ← c119431, k2, ← c119432] at e_1958_15
  have hr := e_1958_15
  simp only [e_1958_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1958 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15770)) := by
  have c119433 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.1
  have c119434 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119435 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119436 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119437 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119438 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1958_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_16
  simp only [← c119433, ← c119434, ← c119435] at e_1958_16
  have e_1958_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_17
  simp only [← c119436, ← c119437, k2, ← c119438] at e_1958_17
  have hr := e_1958_17
  simp only [e_1958_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1958 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15771)) := by
  have c119439 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119440 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119441 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119442 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119443 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119444 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1958_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_18
  simp only [← c119439, ← c119440, ← c119441] at e_1958_18
  have e_1958_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1958 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_19
  simp only [← c119442, ← c119443, k2, ← c119444] at e_1958_19
  have hr := e_1958_19
  simp only [e_1958_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1959 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15772)) := by
  have c119445 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119446 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119447 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119448 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119449 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119450 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1959_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_0
  simp only [← c119445, ← c119446, ← c119447] at e_1959_0
  have e_1959_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_1
  simp only [← c119448, ← c119449, k2, ← c119450] at e_1959_1
  have hr := e_1959_1
  simp only [e_1959_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1959 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15773)) := by
  have c119451 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119452 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119453 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119454 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119455 := (publicBatchWrapper86_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119456 := (publicBatchWrapper86_copies3733 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1959_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_2
  simp only [← c119451, ← c119452, ← c119453] at e_1959_2
  have e_1959_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_3
  simp only [← c119454, ← c119455, k2, ← c119456] at e_1959_3
  have hr := e_1959_3
  simp only [e_1959_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1959 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15774)) := by
  have c119457 := (publicBatchWrapper86_copies3733 a h).2.1
  have c119458 := (publicBatchWrapper86_copies3733 a h).2.2.1
  have c119459 := (publicBatchWrapper86_copies3733 a h).2.2.2.1
  have c119460 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.1
  have c119461 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.1
  have c119462 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1959_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_4
  simp only [← c119457, ← c119458, ← c119459] at e_1959_4
  have e_1959_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_5
  simp only [← c119460, ← c119461, k2, ← c119462] at e_1959_5
  have hr := e_1959_5
  simp only [e_1959_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1959 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15775)) := by
  have c119463 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.1
  have c119464 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.1
  have c119465 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.1
  have c119466 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119467 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119468 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1959_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_6
  simp only [← c119463, ← c119464, ← c119465] at e_1959_6
  have e_1959_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_7
  simp only [← c119466, ← c119467, k2, ← c119468] at e_1959_7
  have hr := e_1959_7
  simp only [e_1959_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1959 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15776)) := by
  have c119469 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119470 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119471 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119472 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119473 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119474 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1959_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_8
  simp only [← c119469, ← c119470, ← c119471] at e_1959_8
  have e_1959_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_9
  simp only [← c119472, ← c119473, k2, ← c119474] at e_1959_9
  have hr := e_1959_9
  simp only [e_1959_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1959 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15777)) := by
  have c119475 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119476 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119477 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119478 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119479 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119480 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1959_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_10
  simp only [← c119475, ← c119476, ← c119477] at e_1959_10
  have e_1959_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_11
  simp only [← c119478, ← c119479, k2, ← c119480] at e_1959_11
  have hr := e_1959_11
  simp only [e_1959_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1959 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15778)) := by
  have c119481 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119482 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119483 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119484 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119485 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119486 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1959_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_12
  simp only [← c119481, ← c119482, ← c119483] at e_1959_12
  have e_1959_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_13
  simp only [← c119484, ← c119485, k2, ← c119486] at e_1959_13
  have hr := e_1959_13
  simp only [e_1959_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1959 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15779)) := by
  have c119487 := (publicBatchWrapper86_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119488 := (publicBatchWrapper86_copies3734 a h).1
  have c119489 := (publicBatchWrapper86_copies3734 a h).2.1
  have c119490 := (publicBatchWrapper86_copies3734 a h).2.2.1
  have c119491 := (publicBatchWrapper86_copies3734 a h).2.2.2.1
  have c119492 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1959_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_14
  simp only [← c119487, ← c119488, ← c119489] at e_1959_14
  have e_1959_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_15
  simp only [← c119490, ← c119491, k2, ← c119492] at e_1959_15
  have hr := e_1959_15
  simp only [e_1959_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1959 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15780)) := by
  have c119493 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.1
  have c119494 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.1
  have c119495 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.1
  have c119496 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.1
  have c119497 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.1
  have c119498 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1959_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_16
  simp only [← c119493, ← c119494, ← c119495] at e_1959_16
  have e_1959_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_17
  simp only [← c119496, ← c119497, k2, ← c119498] at e_1959_17
  have hr := e_1959_17
  simp only [e_1959_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1959 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15781)) := by
  have c119499 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119500 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119501 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119502 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119503 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119504 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1959_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_18
  simp only [← c119499, ← c119500, ← c119501] at e_1959_18
  have e_1959_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1959 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_19
  simp only [← c119502, ← c119503, k2, ← c119504] at e_1959_19
  have hr := e_1959_19
  simp only [e_1959_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1960 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15782)) := by
  have c119505 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119506 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119507 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119508 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119509 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119510 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1960_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_0
  simp only [← c119505, ← c119506, ← c119507] at e_1960_0
  have e_1960_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_1
  simp only [← c119508, ← c119509, k2, ← c119510] at e_1960_1
  have hr := e_1960_1
  simp only [e_1960_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1960 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15783)) := by
  have c119511 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119512 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119513 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119514 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119515 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119516 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1960_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_2
  simp only [← c119511, ← c119512, ← c119513] at e_1960_2
  have e_1960_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_3
  simp only [← c119514, ← c119515, k2, ← c119516] at e_1960_3
  have hr := e_1960_3
  simp only [e_1960_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1960 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15784)) := by
  have c119517 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119518 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119519 := (publicBatchWrapper86_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119520 := (publicBatchWrapper86_copies3735 a h).1
  have c119521 := (publicBatchWrapper86_copies3735 a h).2.1
  have c119522 := (publicBatchWrapper86_copies3735 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1960_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_4
  simp only [← c119517, ← c119518, ← c119519] at e_1960_4
  have e_1960_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_5
  simp only [← c119520, ← c119521, k2, ← c119522] at e_1960_5
  have hr := e_1960_5
  simp only [e_1960_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1960 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15785)) := by
  have c119523 := (publicBatchWrapper86_copies3735 a h).2.2.2.1
  have c119524 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.1
  have c119525 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.1
  have c119526 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.1
  have c119527 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.1
  have c119528 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1960_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_6
  simp only [← c119523, ← c119524, ← c119525] at e_1960_6
  have e_1960_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_7
  simp only [← c119526, ← c119527, k2, ← c119528] at e_1960_7
  have hr := e_1960_7
  simp only [e_1960_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1960 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15786)) := by
  have c119529 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.1
  have c119530 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119531 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119532 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119533 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119534 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1960_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_8
  simp only [← c119529, ← c119530, ← c119531] at e_1960_8
  have e_1960_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_9
  simp only [← c119532, ← c119533, k2, ← c119534] at e_1960_9
  have hr := e_1960_9
  simp only [e_1960_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1960 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15787)) := by
  have c119535 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119536 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119537 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119538 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119539 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119540 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1960_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_10
  simp only [← c119535, ← c119536, ← c119537] at e_1960_10
  have e_1960_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_11
  simp only [← c119538, ← c119539, k2, ← c119540] at e_1960_11
  have hr := e_1960_11
  simp only [e_1960_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1960 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15788)) := by
  have c119541 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119542 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119543 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119544 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119545 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119546 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1960_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_12
  simp only [← c119541, ← c119542, ← c119543] at e_1960_12
  have e_1960_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_13
  simp only [← c119544, ← c119545, k2, ← c119546] at e_1960_13
  have hr := e_1960_13
  simp only [e_1960_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1960 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15789)) := by
  have c119547 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119548 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119549 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119550 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119551 := (publicBatchWrapper86_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119552 := (publicBatchWrapper86_copies3736 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1960_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_14
  simp only [← c119547, ← c119548, ← c119549] at e_1960_14
  have e_1960_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_15
  simp only [← c119550, ← c119551, k2, ← c119552] at e_1960_15
  have hr := e_1960_15
  simp only [e_1960_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1960 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15790)) := by
  have c119553 := (publicBatchWrapper86_copies3736 a h).2.1
  have c119554 := (publicBatchWrapper86_copies3736 a h).2.2.1
  have c119555 := (publicBatchWrapper86_copies3736 a h).2.2.2.1
  have c119556 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.1
  have c119557 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.1
  have c119558 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1960_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_16
  simp only [← c119553, ← c119554, ← c119555] at e_1960_16
  have e_1960_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_17
  simp only [← c119556, ← c119557, k2, ← c119558] at e_1960_17
  have hr := e_1960_17
  simp only [e_1960_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1960 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15791)) := by
  have c119559 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.1
  have c119560 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.1
  have c119561 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.1
  have c119562 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119563 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119564 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1960_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_18
  simp only [← c119559, ← c119560, ← c119561] at e_1960_18
  have e_1960_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1960 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_19
  simp only [← c119562, ← c119563, k2, ← c119564] at e_1960_19
  have hr := e_1960_19
  simp only [e_1960_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1961 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15792)) := by
  have c119565 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119566 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119567 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119568 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119569 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119570 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1961_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_0
  simp only [← c119565, ← c119566, ← c119567] at e_1961_0
  have e_1961_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_1
  simp only [← c119568, ← c119569, k2, ← c119570] at e_1961_1
  have hr := e_1961_1
  simp only [e_1961_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1961 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15793)) := by
  have c119571 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119572 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119573 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119574 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119575 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119576 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1961_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_2
  simp only [← c119571, ← c119572, ← c119573] at e_1961_2
  have e_1961_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_3
  simp only [← c119574, ← c119575, k2, ← c119576] at e_1961_3
  have hr := e_1961_3
  simp only [e_1961_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1961 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15794)) := by
  have c119577 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119578 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119579 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119580 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119581 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119582 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1961_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_4
  simp only [← c119577, ← c119578, ← c119579] at e_1961_4
  have e_1961_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_5
  simp only [← c119580, ← c119581, k2, ← c119582] at e_1961_5
  have hr := e_1961_5
  simp only [e_1961_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1961 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15795)) := by
  have c119583 := (publicBatchWrapper86_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119584 := (publicBatchWrapper86_copies3737 a h).1
  have c119585 := (publicBatchWrapper86_copies3737 a h).2.1
  have c119586 := (publicBatchWrapper86_copies3737 a h).2.2.1
  have c119587 := (publicBatchWrapper86_copies3737 a h).2.2.2.1
  have c119588 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1961_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_6
  simp only [← c119583, ← c119584, ← c119585] at e_1961_6
  have e_1961_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_7
  simp only [← c119586, ← c119587, k2, ← c119588] at e_1961_7
  have hr := e_1961_7
  simp only [e_1961_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1961 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15796)) := by
  have c119589 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.1
  have c119590 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.1
  have c119591 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.1
  have c119592 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.1
  have c119593 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.1
  have c119594 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1961_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_8
  simp only [← c119589, ← c119590, ← c119591] at e_1961_8
  have e_1961_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_9
  simp only [← c119592, ← c119593, k2, ← c119594] at e_1961_9
  have hr := e_1961_9
  simp only [e_1961_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1961 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15797)) := by
  have c119595 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119596 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119597 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119598 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119599 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119600 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1961_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_10
  simp only [← c119595, ← c119596, ← c119597] at e_1961_10
  have e_1961_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_11
  simp only [← c119598, ← c119599, k2, ← c119600] at e_1961_11
  have hr := e_1961_11
  simp only [e_1961_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1961 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15798)) := by
  have c119601 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119602 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119603 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119604 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119605 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119606 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1961_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_12
  simp only [← c119601, ← c119602, ← c119603] at e_1961_12
  have e_1961_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_13
  simp only [← c119604, ← c119605, k2, ← c119606] at e_1961_13
  have hr := e_1961_13
  simp only [e_1961_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1961 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15799)) := by
  have c119607 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119608 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119609 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119610 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119611 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119612 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1961_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_14
  simp only [← c119607, ← c119608, ← c119609] at e_1961_14
  have e_1961_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_15
  simp only [← c119610, ← c119611, k2, ← c119612] at e_1961_15
  have hr := e_1961_15
  simp only [e_1961_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1961 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15800)) := by
  have c119613 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119614 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119615 := (publicBatchWrapper86_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119616 := (publicBatchWrapper86_copies3738 a h).1
  have c119617 := (publicBatchWrapper86_copies3738 a h).2.1
  have c119618 := (publicBatchWrapper86_copies3738 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1961_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_16
  simp only [← c119613, ← c119614, ← c119615] at e_1961_16
  have e_1961_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_17
  simp only [← c119616, ← c119617, k2, ← c119618] at e_1961_17
  have hr := e_1961_17
  simp only [e_1961_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1961 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15801)) := by
  have c119619 := (publicBatchWrapper86_copies3738 a h).2.2.2.1
  have c119620 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.1
  have c119621 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.1
  have c119622 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.1
  have c119623 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.1
  have c119624 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1961_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_18
  simp only [← c119619, ← c119620, ← c119621] at e_1961_18
  have e_1961_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1961 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_19
  simp only [← c119622, ← c119623, k2, ← c119624] at e_1961_19
  have hr := e_1961_19
  simp only [e_1961_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1962 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15970)) := by
  have c119625 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.1
  have c119626 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119627 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119628 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119629 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119630 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1962_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_0
  simp only [← c119625, ← c119626, ← c119627] at e_1962_0
  have e_1962_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_1
  simp only [← c119628, ← c119629, k2, ← c119630] at e_1962_1
  have hr := e_1962_1
  simp only [e_1962_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1962 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15971)) := by
  have c119631 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119632 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119633 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119634 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119635 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119636 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1962_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_2
  simp only [← c119631, ← c119632, ← c119633] at e_1962_2
  have e_1962_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_3
  simp only [← c119634, ← c119635, k2, ← c119636] at e_1962_3
  have hr := e_1962_3
  simp only [e_1962_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1962 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15972)) := by
  have c119637 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119638 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119639 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119640 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119641 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119642 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1962_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_4
  simp only [← c119637, ← c119638, ← c119639] at e_1962_4
  have e_1962_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_5
  simp only [← c119640, ← c119641, k2, ← c119642] at e_1962_5
  have hr := e_1962_5
  simp only [e_1962_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1962 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15973)) := by
  have c119643 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119644 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119645 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119646 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119647 := (publicBatchWrapper86_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119648 := (publicBatchWrapper86_copies3739 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1962_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_6
  simp only [← c119643, ← c119644, ← c119645] at e_1962_6
  have e_1962_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_7
  simp only [← c119646, ← c119647, k2, ← c119648] at e_1962_7
  have hr := e_1962_7
  simp only [e_1962_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1962 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15974)) := by
  have c119649 := (publicBatchWrapper86_copies3739 a h).2.1
  have c119650 := (publicBatchWrapper86_copies3739 a h).2.2.1
  have c119651 := (publicBatchWrapper86_copies3739 a h).2.2.2.1
  have c119652 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.1
  have c119653 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.1
  have c119654 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1962_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_8
  simp only [← c119649, ← c119650, ← c119651] at e_1962_8
  have e_1962_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_9
  simp only [← c119652, ← c119653, k2, ← c119654] at e_1962_9
  have hr := e_1962_9
  simp only [e_1962_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1962 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15975)) := by
  have c119655 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.1
  have c119656 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.1
  have c119657 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.1
  have c119658 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119659 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119660 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1962_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_10
  simp only [← c119655, ← c119656, ← c119657] at e_1962_10
  have e_1962_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_11
  simp only [← c119658, ← c119659, k2, ← c119660] at e_1962_11
  have hr := e_1962_11
  simp only [e_1962_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1962 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15976)) := by
  have c119661 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119662 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119663 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119664 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119665 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119666 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1962_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_12
  simp only [← c119661, ← c119662, ← c119663] at e_1962_12
  have e_1962_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_13
  simp only [← c119664, ← c119665, k2, ← c119666] at e_1962_13
  have hr := e_1962_13
  simp only [e_1962_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1962 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15977)) := by
  have c119667 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119668 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119669 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119670 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119671 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119672 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1962_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_14
  simp only [← c119667, ← c119668, ← c119669] at e_1962_14
  have e_1962_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_15
  simp only [← c119670, ← c119671, k2, ← c119672] at e_1962_15
  have hr := e_1962_15
  simp only [e_1962_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1962 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15978)) := by
  have c119673 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119674 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119675 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119676 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119677 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119678 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1962_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_16
  simp only [← c119673, ← c119674, ← c119675] at e_1962_16
  have e_1962_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_17
  simp only [← c119676, ← c119677, k2, ← c119678] at e_1962_17
  have hr := e_1962_17
  simp only [e_1962_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1962 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15979)) := by
  have c119679 := (publicBatchWrapper86_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119680 := (publicBatchWrapper86_copies3740 a h).1
  have c119681 := (publicBatchWrapper86_copies3740 a h).2.1
  have c119682 := (publicBatchWrapper86_copies3740 a h).2.2.1
  have c119683 := (publicBatchWrapper86_copies3740 a h).2.2.2.1
  have c119684 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1962_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_18
  simp only [← c119679, ← c119680, ← c119681] at e_1962_18
  have e_1962_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1962 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_19
  simp only [← c119682, ← c119683, k2, ← c119684] at e_1962_19
  have hr := e_1962_19
  simp only [e_1962_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1963 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15980)) := by
  have c119685 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.1
  have c119686 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.1
  have c119687 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.1
  have c119688 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.1
  have c119689 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.1
  have c119690 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1963_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_0
  simp only [← c119685, ← c119686, ← c119687] at e_1963_0
  have e_1963_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_1
  simp only [← c119688, ← c119689, k2, ← c119690] at e_1963_1
  have hr := e_1963_1
  simp only [e_1963_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1963 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15981)) := by
  have c119691 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119692 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119693 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119694 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119695 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119696 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1963_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_2
  simp only [← c119691, ← c119692, ← c119693] at e_1963_2
  have e_1963_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_3
  simp only [← c119694, ← c119695, k2, ← c119696] at e_1963_3
  have hr := e_1963_3
  simp only [e_1963_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1963 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15982)) := by
  have c119697 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119698 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119699 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119700 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119701 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119702 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1963_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_4
  simp only [← c119697, ← c119698, ← c119699] at e_1963_4
  have e_1963_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_5
  simp only [← c119700, ← c119701, k2, ← c119702] at e_1963_5
  have hr := e_1963_5
  simp only [e_1963_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1963 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15983)) := by
  have c119703 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119704 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119705 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119706 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119707 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119708 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1963_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_6
  simp only [← c119703, ← c119704, ← c119705] at e_1963_6
  have e_1963_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_7
  simp only [← c119706, ← c119707, k2, ← c119708] at e_1963_7
  have hr := e_1963_7
  simp only [e_1963_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1963 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15984)) := by
  have c119709 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119710 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119711 := (publicBatchWrapper86_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119712 := (publicBatchWrapper86_copies3741 a h).1
  have c119713 := (publicBatchWrapper86_copies3741 a h).2.1
  have c119714 := (publicBatchWrapper86_copies3741 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1963_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_8
  simp only [← c119709, ← c119710, ← c119711] at e_1963_8
  have e_1963_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_9
  simp only [← c119712, ← c119713, k2, ← c119714] at e_1963_9
  have hr := e_1963_9
  simp only [e_1963_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1963 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15985)) := by
  have c119715 := (publicBatchWrapper86_copies3741 a h).2.2.2.1
  have c119716 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.1
  have c119717 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.1
  have c119718 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.1
  have c119719 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.1
  have c119720 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1963_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_10
  simp only [← c119715, ← c119716, ← c119717] at e_1963_10
  have e_1963_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_11
  simp only [← c119718, ← c119719, k2, ← c119720] at e_1963_11
  have hr := e_1963_11
  simp only [e_1963_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1963 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15986)) := by
  have c119721 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.1
  have c119722 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119723 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119724 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119725 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119726 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1963_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_12
  simp only [← c119721, ← c119722, ← c119723] at e_1963_12
  have e_1963_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_13
  simp only [← c119724, ← c119725, k2, ← c119726] at e_1963_13
  have hr := e_1963_13
  simp only [e_1963_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1963 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15987)) := by
  have c119727 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119728 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119729 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119730 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119731 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119732 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1963_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_14
  simp only [← c119727, ← c119728, ← c119729] at e_1963_14
  have e_1963_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_15
  simp only [← c119730, ← c119731, k2, ← c119732] at e_1963_15
  have hr := e_1963_15
  simp only [e_1963_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1963 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15988)) := by
  have c119733 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119734 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119735 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119736 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119737 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119738 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1963_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_16
  simp only [← c119733, ← c119734, ← c119735] at e_1963_16
  have e_1963_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_17
  simp only [← c119736, ← c119737, k2, ← c119738] at e_1963_17
  have hr := e_1963_17
  simp only [e_1963_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1963 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15989)) := by
  have c119739 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119740 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119741 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119742 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119743 := (publicBatchWrapper86_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119744 := (publicBatchWrapper86_copies3742 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1963_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_18
  simp only [← c119739, ← c119740, ← c119741] at e_1963_18
  have e_1963_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1963 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_19
  simp only [← c119742, ← c119743, k2, ← c119744] at e_1963_19
  have hr := e_1963_19
  simp only [e_1963_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1964 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15990)) := by
  have c119745 := (publicBatchWrapper86_copies3742 a h).2.1
  have c119746 := (publicBatchWrapper86_copies3742 a h).2.2.1
  have c119747 := (publicBatchWrapper86_copies3742 a h).2.2.2.1
  have c119748 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.1
  have c119749 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.1
  have c119750 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1964_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_0
  simp only [← c119745, ← c119746, ← c119747] at e_1964_0
  have e_1964_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_1
  simp only [← c119748, ← c119749, k2, ← c119750] at e_1964_1
  have hr := e_1964_1
  simp only [e_1964_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1964 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15991)) := by
  have c119751 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.1
  have c119752 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.1
  have c119753 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.1
  have c119754 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119755 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119756 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1964_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_2
  simp only [← c119751, ← c119752, ← c119753] at e_1964_2
  have e_1964_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_3
  simp only [← c119754, ← c119755, k2, ← c119756] at e_1964_3
  have hr := e_1964_3
  simp only [e_1964_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1964 23) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15992)) := by
  have c119757 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119758 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119759 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119760 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119761 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119762 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1964_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_4
  simp only [← c119757, ← c119758, ← c119759] at e_1964_4
  have e_1964_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_5
  simp only [← c119760, ← c119761, k2, ← c119762] at e_1964_5
  have hr := e_1964_5
  simp only [e_1964_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1964 31) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15993)) := by
  have c119763 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119764 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119765 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119766 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119767 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119768 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1964_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_6
  simp only [← c119763, ← c119764, ← c119765] at e_1964_6
  have e_1964_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_7
  simp only [← c119766, ← c119767, k2, ← c119768] at e_1964_7
  have hr := e_1964_7
  simp only [e_1964_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1964 39) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15994)) := by
  have c119769 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119770 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119771 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119772 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119773 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119774 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1964_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_8
  simp only [← c119769, ← c119770, ← c119771] at e_1964_8
  have e_1964_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_9
  simp only [← c119772, ← c119773, k2, ← c119774] at e_1964_9
  have hr := e_1964_9
  simp only [e_1964_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1964 47) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15995)) := by
  have c119775 := (publicBatchWrapper86_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119776 := (publicBatchWrapper86_copies3743 a h).1
  have c119777 := (publicBatchWrapper86_copies3743 a h).2.1
  have c119778 := (publicBatchWrapper86_copies3743 a h).2.2.1
  have c119779 := (publicBatchWrapper86_copies3743 a h).2.2.2.1
  have c119780 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1964_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_10
  simp only [← c119775, ← c119776, ← c119777] at e_1964_10
  have e_1964_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_11
  simp only [← c119778, ← c119779, k2, ← c119780] at e_1964_11
  have hr := e_1964_11
  simp only [e_1964_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1964 55) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15996)) := by
  have c119781 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.1
  have c119782 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.1
  have c119783 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.1
  have c119784 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.1
  have c119785 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.1
  have c119786 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1964_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_12
  simp only [← c119781, ← c119782, ← c119783] at e_1964_12
  have e_1964_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_13
  simp only [← c119784, ← c119785, k2, ← c119786] at e_1964_13
  have hr := e_1964_13
  simp only [e_1964_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1964 63) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15997)) := by
  have c119787 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119788 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119789 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119790 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119791 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119792 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1964_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_14
  simp only [← c119787, ← c119788, ← c119789] at e_1964_14
  have e_1964_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_15
  simp only [← c119790, ← c119791, k2, ← c119792] at e_1964_15
  have hr := e_1964_15
  simp only [e_1964_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1964 71) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15998)) := by
  have c119793 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119794 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119795 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119796 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119797 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119798 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1964_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_16
  simp only [← c119793, ← c119794, ← c119795] at e_1964_16
  have e_1964_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_17
  simp only [← c119796, ← c119797, k2, ← c119798] at e_1964_17
  have hr := e_1964_17
  simp only [e_1964_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1964 79) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 15999)) := by
  have c119799 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119800 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119801 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119802 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119803 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119804 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1964_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_18
  simp only [← c119799, ← c119800, ← c119801] at e_1964_18
  have e_1964_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1964 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_19
  simp only [← c119802, ← c119803, k2, ← c119804] at e_1964_19
  have hr := e_1964_19
  simp only [e_1964_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1965 7) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 16000)) := by
  have c119805 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119806 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119807 := (publicBatchWrapper86_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119808 := (publicBatchWrapper86_copies3744 a h).1
  have c119809 := (publicBatchWrapper86_copies3744 a h).2.1
  have c119810 := (publicBatchWrapper86_copies3744 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1965_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_0
  simp only [← c119805, ← c119806, ← c119807] at e_1965_0
  have e_1965_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_1
  simp only [← c119808, ← c119809, k2, ← c119810] at e_1965_1
  have hr := e_1965_1
  simp only [e_1965_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1965 15) = bselect (a (.wire 77 63)) (a (.virt 200)) (a (.virt 16001)) := by
  have c119811 := (publicBatchWrapper86_copies3744 a h).2.2.2.1
  have c119812 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.1
  have c119813 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.1
  have c119814 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.1
  have c119815 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.1
  have c119816 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1965_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_2
  simp only [← c119811, ← c119812, ← c119813] at e_1965_2
  have e_1965_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_3
  simp only [← c119814, ← c119815, k2, ← c119816] at e_1965_3
  have hr := e_1965_3
  simp only [e_1965_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1965 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16170)) := by
  have c119817 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.1
  have c119818 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119819 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119820 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119821 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119822 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1965_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_4
  simp only [← c119817, ← c119818, ← c119819] at e_1965_4
  have e_1965_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_5
  simp only [← c119820, ← c119821, k2, ← c119822] at e_1965_5
  have hr := e_1965_5
  simp only [e_1965_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1965 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16171)) := by
  have c119823 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119824 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119825 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119826 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119827 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119828 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1965_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_6
  simp only [← c119823, ← c119824, ← c119825] at e_1965_6
  have e_1965_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_7
  simp only [← c119826, ← c119827, k2, ← c119828] at e_1965_7
  have hr := e_1965_7
  simp only [e_1965_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1965 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16172)) := by
  have c119829 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119830 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119831 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119832 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119833 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119834 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1965_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_8
  simp only [← c119829, ← c119830, ← c119831] at e_1965_8
  have e_1965_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_9
  simp only [← c119832, ← c119833, k2, ← c119834] at e_1965_9
  have hr := e_1965_9
  simp only [e_1965_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1965 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16173)) := by
  have c119835 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119836 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119837 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119838 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119839 := (publicBatchWrapper86_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119840 := (publicBatchWrapper86_copies3745 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1965_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_10
  simp only [← c119835, ← c119836, ← c119837] at e_1965_10
  have e_1965_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_11
  simp only [← c119838, ← c119839, k2, ← c119840] at e_1965_11
  have hr := e_1965_11
  simp only [e_1965_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1965 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16174)) := by
  have c119841 := (publicBatchWrapper86_copies3745 a h).2.1
  have c119842 := (publicBatchWrapper86_copies3745 a h).2.2.1
  have c119843 := (publicBatchWrapper86_copies3745 a h).2.2.2.1
  have c119844 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.1
  have c119845 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.1
  have c119846 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1965_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_12
  simp only [← c119841, ← c119842, ← c119843] at e_1965_12
  have e_1965_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_13
  simp only [← c119844, ← c119845, k2, ← c119846] at e_1965_13
  have hr := e_1965_13
  simp only [e_1965_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1965 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16175)) := by
  have c119847 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.1
  have c119848 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.1
  have c119849 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.1
  have c119850 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119851 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119852 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1965_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_14
  simp only [← c119847, ← c119848, ← c119849] at e_1965_14
  have e_1965_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_15
  simp only [← c119850, ← c119851, k2, ← c119852] at e_1965_15
  have hr := e_1965_15
  simp only [e_1965_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1965 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16176)) := by
  have c119853 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119854 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119855 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119856 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119857 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119858 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1965_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_16
  simp only [← c119853, ← c119854, ← c119855] at e_1965_16
  have e_1965_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_17
  simp only [← c119856, ← c119857, k2, ← c119858] at e_1965_17
  have hr := e_1965_17
  simp only [e_1965_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1965 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16177)) := by
  have c119859 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119860 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119861 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119862 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119863 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119864 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1965_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_18
  simp only [← c119859, ← c119860, ← c119861] at e_1965_18
  have e_1965_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1965 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_19
  simp only [← c119862, ← c119863, k2, ← c119864] at e_1965_19
  have hr := e_1965_19
  simp only [e_1965_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1966 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16178)) := by
  have c119865 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119866 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119867 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119868 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119869 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119870 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1966_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_0
  simp only [← c119865, ← c119866, ← c119867] at e_1966_0
  have e_1966_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_1
  simp only [← c119868, ← c119869, k2, ← c119870] at e_1966_1
  have hr := e_1966_1
  simp only [e_1966_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1966 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16179)) := by
  have c119871 := (publicBatchWrapper86_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119872 := (publicBatchWrapper86_copies3746 a h).1
  have c119873 := (publicBatchWrapper86_copies3746 a h).2.1
  have c119874 := (publicBatchWrapper86_copies3746 a h).2.2.1
  have c119875 := (publicBatchWrapper86_copies3746 a h).2.2.2.1
  have c119876 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1966_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_2
  simp only [← c119871, ← c119872, ← c119873] at e_1966_2
  have e_1966_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_3
  simp only [← c119874, ← c119875, k2, ← c119876] at e_1966_3
  have hr := e_1966_3
  simp only [e_1966_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1966 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16180)) := by
  have c119877 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.1
  have c119878 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.1
  have c119879 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.1
  have c119880 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.1
  have c119881 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.1
  have c119882 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1966_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_4
  simp only [← c119877, ← c119878, ← c119879] at e_1966_4
  have e_1966_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_5
  simp only [← c119880, ← c119881, k2, ← c119882] at e_1966_5
  have hr := e_1966_5
  simp only [e_1966_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1966 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16181)) := by
  have c119883 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119884 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119885 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119886 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119887 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119888 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1966_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_6
  simp only [← c119883, ← c119884, ← c119885] at e_1966_6
  have e_1966_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_7
  simp only [← c119886, ← c119887, k2, ← c119888] at e_1966_7
  have hr := e_1966_7
  simp only [e_1966_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1966 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16182)) := by
  have c119889 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119890 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119891 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119892 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119893 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119894 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1966_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_8
  simp only [← c119889, ← c119890, ← c119891] at e_1966_8
  have e_1966_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_9
  simp only [← c119892, ← c119893, k2, ← c119894] at e_1966_9
  have hr := e_1966_9
  simp only [e_1966_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1966 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16183)) := by
  have c119895 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119896 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119897 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119898 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119899 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119900 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1966_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_10
  simp only [← c119895, ← c119896, ← c119897] at e_1966_10
  have e_1966_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_11
  simp only [← c119898, ← c119899, k2, ← c119900] at e_1966_11
  have hr := e_1966_11
  simp only [e_1966_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1966 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16184)) := by
  have c119901 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119902 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119903 := (publicBatchWrapper86_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119904 := (publicBatchWrapper86_copies3747 a h).1
  have c119905 := (publicBatchWrapper86_copies3747 a h).2.1
  have c119906 := (publicBatchWrapper86_copies3747 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1966_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_12
  simp only [← c119901, ← c119902, ← c119903] at e_1966_12
  have e_1966_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_13
  simp only [← c119904, ← c119905, k2, ← c119906] at e_1966_13
  have hr := e_1966_13
  simp only [e_1966_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1966 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16185)) := by
  have c119907 := (publicBatchWrapper86_copies3747 a h).2.2.2.1
  have c119908 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.1
  have c119909 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.1
  have c119910 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.1
  have c119911 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.1
  have c119912 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1966_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_14
  simp only [← c119907, ← c119908, ← c119909] at e_1966_14
  have e_1966_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_15
  simp only [← c119910, ← c119911, k2, ← c119912] at e_1966_15
  have hr := e_1966_15
  simp only [e_1966_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1966 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16186)) := by
  have c119913 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.1
  have c119914 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119915 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119916 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119917 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119918 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1966_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_16
  simp only [← c119913, ← c119914, ← c119915] at e_1966_16
  have e_1966_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_17
  simp only [← c119916, ← c119917, k2, ← c119918] at e_1966_17
  have hr := e_1966_17
  simp only [e_1966_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1966 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16187)) := by
  have c119919 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119920 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119921 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119922 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119923 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119924 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1966_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_18
  simp only [← c119919, ← c119920, ← c119921] at e_1966_18
  have e_1966_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1966 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_19
  simp only [← c119922, ← c119923, k2, ← c119924] at e_1966_19
  have hr := e_1966_19
  simp only [e_1966_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1967 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16188)) := by
  have c119925 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119926 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119927 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119928 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119929 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119930 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1967_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_0
  simp only [← c119925, ← c119926, ← c119927] at e_1967_0
  have e_1967_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_1
  simp only [← c119928, ← c119929, k2, ← c119930] at e_1967_1
  have hr := e_1967_1
  simp only [e_1967_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1967 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16189)) := by
  have c119931 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119932 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119933 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119934 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119935 := (publicBatchWrapper86_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119936 := (publicBatchWrapper86_copies3748 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1967_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_2
  simp only [← c119931, ← c119932, ← c119933] at e_1967_2
  have e_1967_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_3
  simp only [← c119934, ← c119935, k2, ← c119936] at e_1967_3
  have hr := e_1967_3
  simp only [e_1967_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1967 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16190)) := by
  have c119937 := (publicBatchWrapper86_copies3748 a h).2.1
  have c119938 := (publicBatchWrapper86_copies3748 a h).2.2.1
  have c119939 := (publicBatchWrapper86_copies3748 a h).2.2.2.1
  have c119940 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.1
  have c119941 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.1
  have c119942 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1967_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_4
  simp only [← c119937, ← c119938, ← c119939] at e_1967_4
  have e_1967_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_5
  simp only [← c119940, ← c119941, k2, ← c119942] at e_1967_5
  have hr := e_1967_5
  simp only [e_1967_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1967 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16191)) := by
  have c119943 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.1
  have c119944 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.1
  have c119945 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.1
  have c119946 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119947 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119948 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1967_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_6
  simp only [← c119943, ← c119944, ← c119945] at e_1967_6
  have e_1967_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_7
  simp only [← c119946, ← c119947, k2, ← c119948] at e_1967_7
  have hr := e_1967_7
  simp only [e_1967_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1967 39) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16192)) := by
  have c119949 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119950 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119951 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119952 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119953 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119954 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1967_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_8
  simp only [← c119949, ← c119950, ← c119951] at e_1967_8
  have e_1967_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_9
  simp only [← c119952, ← c119953, k2, ← c119954] at e_1967_9
  have hr := e_1967_9
  simp only [e_1967_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1967 47) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16193)) := by
  have c119955 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119956 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119957 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119958 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119959 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119960 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1967_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_10
  simp only [← c119955, ← c119956, ← c119957] at e_1967_10
  have e_1967_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_11
  simp only [← c119958, ← c119959, k2, ← c119960] at e_1967_11
  have hr := e_1967_11
  simp only [e_1967_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1967 55) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16194)) := by
  have c119961 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119962 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119963 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119964 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119965 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119966 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1967_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_12
  simp only [← c119961, ← c119962, ← c119963] at e_1967_12
  have e_1967_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_13
  simp only [← c119964, ← c119965, k2, ← c119966] at e_1967_13
  have hr := e_1967_13
  simp only [e_1967_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1967 63) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16195)) := by
  have c119967 := (publicBatchWrapper86_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119968 := (publicBatchWrapper86_copies3749 a h).1
  have c119969 := (publicBatchWrapper86_copies3749 a h).2.1
  have c119970 := (publicBatchWrapper86_copies3749 a h).2.2.1
  have c119971 := (publicBatchWrapper86_copies3749 a h).2.2.2.1
  have c119972 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1967_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_14
  simp only [← c119967, ← c119968, ← c119969] at e_1967_14
  have e_1967_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_15
  simp only [← c119970, ← c119971, k2, ← c119972] at e_1967_15
  have hr := e_1967_15
  simp only [e_1967_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1967 71) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16196)) := by
  have c119973 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.1
  have c119974 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.1
  have c119975 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.1
  have c119976 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.1
  have c119977 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.1
  have c119978 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1967_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_16
  simp only [← c119973, ← c119974, ← c119975] at e_1967_16
  have e_1967_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_17
  simp only [← c119976, ← c119977, k2, ← c119978] at e_1967_17
  have hr := e_1967_17
  simp only [e_1967_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1967 79) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16197)) := by
  have c119979 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119980 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119981 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119982 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119983 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119984 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1967_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_18
  simp only [← c119979, ← c119980, ← c119981] at e_1967_18
  have e_1967_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1967 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_19
  simp only [← c119982, ← c119983, k2, ← c119984] at e_1967_19
  have hr := e_1967_19
  simp only [e_1967_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1968 7) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16198)) := by
  have c119985 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119986 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119987 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119988 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119989 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119990 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1968_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_0
  simp only [← c119985, ← c119986, ← c119987] at e_1968_0
  have e_1968_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_1
  simp only [← c119988, ← c119989, k2, ← c119990] at e_1968_1
  have hr := e_1968_1
  simp only [e_1968_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1968 15) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16199)) := by
  have c119991 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119992 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119993 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119994 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119995 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119996 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1968_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_2
  simp only [← c119991, ← c119992, ← c119993] at e_1968_2
  have e_1968_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_3
  simp only [← c119994, ← c119995, k2, ← c119996] at e_1968_3
  have hr := e_1968_3
  simp only [e_1968_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1968 23) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16200)) := by
  have c119997 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119998 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119999 := (publicBatchWrapper86_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120000 := (publicBatchWrapper86_copies3750 a h).1
  have c120001 := (publicBatchWrapper86_copies3750 a h).2.1
  have c120002 := (publicBatchWrapper86_copies3750 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1968_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_4
  simp only [← c119997, ← c119998, ← c119999] at e_1968_4
  have e_1968_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_5
  simp only [← c120000, ← c120001, k2, ← c120002] at e_1968_5
  have hr := e_1968_5
  simp only [e_1968_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1968 31) = bselect (a (.wire 79 27)) (a (.virt 200)) (a (.virt 16201)) := by
  have c120003 := (publicBatchWrapper86_copies3750 a h).2.2.2.1
  have c120004 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.1
  have c120005 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.1
  have c120006 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.1
  have c120007 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.1
  have c120008 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1968_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_6
  simp only [← c120003, ← c120004, ← c120005] at e_1968_6
  have e_1968_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_7
  simp only [← c120006, ← c120007, k2, ← c120008] at e_1968_7
  have hr := e_1968_7
  simp only [e_1968_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1968 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16370)) := by
  have c120009 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.1
  have c120010 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120011 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120012 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120013 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120014 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1968_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_8
  simp only [← c120009, ← c120010, ← c120011] at e_1968_8
  have e_1968_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_9
  simp only [← c120012, ← c120013, k2, ← c120014] at e_1968_9
  have hr := e_1968_9
  simp only [e_1968_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1968 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16371)) := by
  have c120015 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120016 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120017 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120018 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120019 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120020 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1968_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_10
  simp only [← c120015, ← c120016, ← c120017] at e_1968_10
  have e_1968_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_11
  simp only [← c120018, ← c120019, k2, ← c120020] at e_1968_11
  have hr := e_1968_11
  simp only [e_1968_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1968 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16372)) := by
  have c120021 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120022 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120023 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120024 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120025 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120026 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1968_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_12
  simp only [← c120021, ← c120022, ← c120023] at e_1968_12
  have e_1968_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_13
  simp only [← c120024, ← c120025, k2, ← c120026] at e_1968_13
  have hr := e_1968_13
  simp only [e_1968_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1968 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16373)) := by
  have c120027 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120028 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120029 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120030 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120031 := (publicBatchWrapper86_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120032 := (publicBatchWrapper86_copies3751 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1968_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_14
  simp only [← c120027, ← c120028, ← c120029] at e_1968_14
  have e_1968_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_15
  simp only [← c120030, ← c120031, k2, ← c120032] at e_1968_15
  have hr := e_1968_15
  simp only [e_1968_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1968 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16374)) := by
  have c120033 := (publicBatchWrapper86_copies3751 a h).2.1
  have c120034 := (publicBatchWrapper86_copies3751 a h).2.2.1
  have c120035 := (publicBatchWrapper86_copies3751 a h).2.2.2.1
  have c120036 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.1
  have c120037 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.1
  have c120038 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1968_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_16
  simp only [← c120033, ← c120034, ← c120035] at e_1968_16
  have e_1968_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_17
  simp only [← c120036, ← c120037, k2, ← c120038] at e_1968_17
  have hr := e_1968_17
  simp only [e_1968_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1968 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16375)) := by
  have c120039 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.1
  have c120040 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.1
  have c120041 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.1
  have c120042 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120043 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120044 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1968_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_18
  simp only [← c120039, ← c120040, ← c120041] at e_1968_18
  have e_1968_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1968 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_19
  simp only [← c120042, ← c120043, k2, ← c120044] at e_1968_19
  have hr := e_1968_19
  simp only [e_1968_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1969 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16376)) := by
  have c120045 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120046 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120047 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120048 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120049 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120050 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1969_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_0
  simp only [← c120045, ← c120046, ← c120047] at e_1969_0
  have e_1969_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_1
  simp only [← c120048, ← c120049, k2, ← c120050] at e_1969_1
  have hr := e_1969_1
  simp only [e_1969_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1969 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16377)) := by
  have c120051 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120052 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120053 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120054 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120055 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120056 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1969_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_2
  simp only [← c120051, ← c120052, ← c120053] at e_1969_2
  have e_1969_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_3
  simp only [← c120054, ← c120055, k2, ← c120056] at e_1969_3
  have hr := e_1969_3
  simp only [e_1969_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1969 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16378)) := by
  have c120057 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120058 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120059 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120060 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120061 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120062 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1969_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_4
  simp only [← c120057, ← c120058, ← c120059] at e_1969_4
  have e_1969_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_5
  simp only [← c120060, ← c120061, k2, ← c120062] at e_1969_5
  have hr := e_1969_5
  simp only [e_1969_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1969 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16379)) := by
  have c120063 := (publicBatchWrapper86_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120064 := (publicBatchWrapper86_copies3752 a h).1
  have c120065 := (publicBatchWrapper86_copies3752 a h).2.1
  have c120066 := (publicBatchWrapper86_copies3752 a h).2.2.1
  have c120067 := (publicBatchWrapper86_copies3752 a h).2.2.2.1
  have c120068 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1969_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_6
  simp only [← c120063, ← c120064, ← c120065] at e_1969_6
  have e_1969_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_7
  simp only [← c120066, ← c120067, k2, ← c120068] at e_1969_7
  have hr := e_1969_7
  simp only [e_1969_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1969 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16380)) := by
  have c120069 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.1
  have c120070 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.1
  have c120071 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.1
  have c120072 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.1
  have c120073 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.1
  have c120074 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1969_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_8
  simp only [← c120069, ← c120070, ← c120071] at e_1969_8
  have e_1969_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_9
  simp only [← c120072, ← c120073, k2, ← c120074] at e_1969_9
  have hr := e_1969_9
  simp only [e_1969_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1969 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16381)) := by
  have c120075 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120076 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120077 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120078 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120079 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120080 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1969_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_10
  simp only [← c120075, ← c120076, ← c120077] at e_1969_10
  have e_1969_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_11
  simp only [← c120078, ← c120079, k2, ← c120080] at e_1969_11
  have hr := e_1969_11
  simp only [e_1969_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1969 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16382)) := by
  have c120081 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120082 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120083 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120084 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120085 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120086 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1969_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_12
  simp only [← c120081, ← c120082, ← c120083] at e_1969_12
  have e_1969_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_13
  simp only [← c120084, ← c120085, k2, ← c120086] at e_1969_13
  have hr := e_1969_13
  simp only [e_1969_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1969 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16383)) := by
  have c120087 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120088 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120089 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120090 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120091 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120092 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1969_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_14
  simp only [← c120087, ← c120088, ← c120089] at e_1969_14
  have e_1969_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_15
  simp only [← c120090, ← c120091, k2, ← c120092] at e_1969_15
  have hr := e_1969_15
  simp only [e_1969_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1969 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16384)) := by
  have c120093 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120094 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120095 := (publicBatchWrapper86_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120096 := (publicBatchWrapper86_copies3753 a h).1
  have c120097 := (publicBatchWrapper86_copies3753 a h).2.1
  have c120098 := (publicBatchWrapper86_copies3753 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1969_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_16
  simp only [← c120093, ← c120094, ← c120095] at e_1969_16
  have e_1969_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_17
  simp only [← c120096, ← c120097, k2, ← c120098] at e_1969_17
  have hr := e_1969_17
  simp only [e_1969_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1969 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16385)) := by
  have c120099 := (publicBatchWrapper86_copies3753 a h).2.2.2.1
  have c120100 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.1
  have c120101 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.1
  have c120102 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.1
  have c120103 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.1
  have c120104 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1969_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_18
  simp only [← c120099, ← c120100, ← c120101] at e_1969_18
  have e_1969_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1969 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_19
  simp only [← c120102, ← c120103, k2, ← c120104] at e_1969_19
  have hr := e_1969_19
  simp only [e_1969_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1970 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16386)) := by
  have c120105 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.1
  have c120106 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120107 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120108 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120109 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120110 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1970_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_0
  simp only [← c120105, ← c120106, ← c120107] at e_1970_0
  have e_1970_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_1
  simp only [← c120108, ← c120109, k2, ← c120110] at e_1970_1
  have hr := e_1970_1
  simp only [e_1970_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1970 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16387)) := by
  have c120111 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120112 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120113 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120114 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120115 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120116 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1970_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_2
  simp only [← c120111, ← c120112, ← c120113] at e_1970_2
  have e_1970_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_3
  simp only [← c120114, ← c120115, k2, ← c120116] at e_1970_3
  have hr := e_1970_3
  simp only [e_1970_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1970 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16388)) := by
  have c120117 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120118 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120119 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120120 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120121 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120122 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1970_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_4
  simp only [← c120117, ← c120118, ← c120119] at e_1970_4
  have e_1970_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_5
  simp only [← c120120, ← c120121, k2, ← c120122] at e_1970_5
  have hr := e_1970_5
  simp only [e_1970_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1970 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16389)) := by
  have c120123 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120124 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120125 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120126 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120127 := (publicBatchWrapper86_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120128 := (publicBatchWrapper86_copies3754 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1970_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_6
  simp only [← c120123, ← c120124, ← c120125] at e_1970_6
  have e_1970_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_7
  simp only [← c120126, ← c120127, k2, ← c120128] at e_1970_7
  have hr := e_1970_7
  simp only [e_1970_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1970 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16390)) := by
  have c120129 := (publicBatchWrapper86_copies3754 a h).2.1
  have c120130 := (publicBatchWrapper86_copies3754 a h).2.2.1
  have c120131 := (publicBatchWrapper86_copies3754 a h).2.2.2.1
  have c120132 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.1
  have c120133 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.1
  have c120134 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1970_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_8
  simp only [← c120129, ← c120130, ← c120131] at e_1970_8
  have e_1970_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_9
  simp only [← c120132, ← c120133, k2, ← c120134] at e_1970_9
  have hr := e_1970_9
  simp only [e_1970_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1970 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16391)) := by
  have c120135 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.1
  have c120136 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.1
  have c120137 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.1
  have c120138 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120139 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120140 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1970_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_10
  simp only [← c120135, ← c120136, ← c120137] at e_1970_10
  have e_1970_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_11
  simp only [← c120138, ← c120139, k2, ← c120140] at e_1970_11
  have hr := e_1970_11
  simp only [e_1970_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1970 55) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16392)) := by
  have c120141 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120142 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120143 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120144 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120145 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120146 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1970_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_12
  simp only [← c120141, ← c120142, ← c120143] at e_1970_12
  have e_1970_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_13
  simp only [← c120144, ← c120145, k2, ← c120146] at e_1970_13
  have hr := e_1970_13
  simp only [e_1970_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1970 63) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16393)) := by
  have c120147 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120148 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120149 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120150 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120151 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120152 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1970_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_14
  simp only [← c120147, ← c120148, ← c120149] at e_1970_14
  have e_1970_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_15
  simp only [← c120150, ← c120151, k2, ← c120152] at e_1970_15
  have hr := e_1970_15
  simp only [e_1970_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1970 71) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16394)) := by
  have c120153 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120154 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120155 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120156 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120157 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120158 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1970_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_16
  simp only [← c120153, ← c120154, ← c120155] at e_1970_16
  have e_1970_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_17
  simp only [← c120156, ← c120157, k2, ← c120158] at e_1970_17
  have hr := e_1970_17
  simp only [e_1970_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1970 79) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16395)) := by
  have c120159 := (publicBatchWrapper86_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120160 := (publicBatchWrapper86_copies3755 a h).1
  have c120161 := (publicBatchWrapper86_copies3755 a h).2.1
  have c120162 := (publicBatchWrapper86_copies3755 a h).2.2.1
  have c120163 := (publicBatchWrapper86_copies3755 a h).2.2.2.1
  have c120164 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1970_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_18
  simp only [← c120159, ← c120160, ← c120161] at e_1970_18
  have e_1970_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1970 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_19
  simp only [← c120162, ← c120163, k2, ← c120164] at e_1970_19
  have hr := e_1970_19
  simp only [e_1970_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1971 7) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16396)) := by
  have c120165 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.1
  have c120166 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.1
  have c120167 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.1
  have c120168 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.1
  have c120169 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.1
  have c120170 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1971_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_0
  simp only [← c120165, ← c120166, ← c120167] at e_1971_0
  have e_1971_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_1
  simp only [← c120168, ← c120169, k2, ← c120170] at e_1971_1
  have hr := e_1971_1
  simp only [e_1971_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1971 15) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16397)) := by
  have c120171 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120172 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120173 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120174 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120175 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120176 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1971_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_2
  simp only [← c120171, ← c120172, ← c120173] at e_1971_2
  have e_1971_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_3
  simp only [← c120174, ← c120175, k2, ← c120176] at e_1971_3
  have hr := e_1971_3
  simp only [e_1971_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1971 23) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16398)) := by
  have c120177 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120178 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120179 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120180 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120181 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120182 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1971_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_4
  simp only [← c120177, ← c120178, ← c120179] at e_1971_4
  have e_1971_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_5
  simp only [← c120180, ← c120181, k2, ← c120182] at e_1971_5
  have hr := e_1971_5
  simp only [e_1971_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1971 31) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16399)) := by
  have c120183 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120184 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120185 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120186 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120187 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120188 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1971_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_6
  simp only [← c120183, ← c120184, ← c120185] at e_1971_6
  have e_1971_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_7
  simp only [← c120186, ← c120187, k2, ← c120188] at e_1971_7
  have hr := e_1971_7
  simp only [e_1971_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1971 39) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16400)) := by
  have c120189 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120190 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120191 := (publicBatchWrapper86_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120192 := (publicBatchWrapper86_copies3756 a h).1
  have c120193 := (publicBatchWrapper86_copies3756 a h).2.1
  have c120194 := (publicBatchWrapper86_copies3756 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1971_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_8
  simp only [← c120189, ← c120190, ← c120191] at e_1971_8
  have e_1971_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_9
  simp only [← c120192, ← c120193, k2, ← c120194] at e_1971_9
  have hr := e_1971_9
  simp only [e_1971_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1971 47) = bselect (a (.wire 79 71)) (a (.virt 200)) (a (.virt 16401)) := by
  have c120195 := (publicBatchWrapper86_copies3756 a h).2.2.2.1
  have c120196 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.1
  have c120197 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.1
  have c120198 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.1
  have c120199 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.1
  have c120200 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1971_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_10
  simp only [← c120195, ← c120196, ← c120197] at e_1971_10
  have e_1971_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_11
  simp only [← c120198, ← c120199, k2, ← c120200] at e_1971_11
  have hr := e_1971_11
  simp only [e_1971_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1971 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16570)) := by
  have c120201 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.1
  have c120202 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120203 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120204 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120205 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120206 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1971_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_12
  simp only [← c120201, ← c120202, ← c120203] at e_1971_12
  have e_1971_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_13
  simp only [← c120204, ← c120205, k2, ← c120206] at e_1971_13
  have hr := e_1971_13
  simp only [e_1971_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1971 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16571)) := by
  have c120207 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120208 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120209 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120210 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120211 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120212 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1971_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_14
  simp only [← c120207, ← c120208, ← c120209] at e_1971_14
  have e_1971_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_15
  simp only [← c120210, ← c120211, k2, ← c120212] at e_1971_15
  have hr := e_1971_15
  simp only [e_1971_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1971 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16572)) := by
  have c120213 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120214 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120215 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120216 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120217 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120218 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1971_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_16
  simp only [← c120213, ← c120214, ← c120215] at e_1971_16
  have e_1971_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_17
  simp only [← c120216, ← c120217, k2, ← c120218] at e_1971_17
  have hr := e_1971_17
  simp only [e_1971_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1971 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16573)) := by
  have c120219 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120220 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120221 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120222 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120223 := (publicBatchWrapper86_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120224 := (publicBatchWrapper86_copies3757 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1971_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_18
  simp only [← c120219, ← c120220, ← c120221] at e_1971_18
  have e_1971_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1971 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_19
  simp only [← c120222, ← c120223, k2, ← c120224] at e_1971_19
  have hr := e_1971_19
  simp only [e_1971_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1972 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16574)) := by
  have c120225 := (publicBatchWrapper86_copies3757 a h).2.1
  have c120226 := (publicBatchWrapper86_copies3757 a h).2.2.1
  have c120227 := (publicBatchWrapper86_copies3757 a h).2.2.2.1
  have c120228 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.1
  have c120229 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.1
  have c120230 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1972_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_0
  simp only [← c120225, ← c120226, ← c120227] at e_1972_0
  have e_1972_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_1
  simp only [← c120228, ← c120229, k2, ← c120230] at e_1972_1
  have hr := e_1972_1
  simp only [e_1972_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1972 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16575)) := by
  have c120231 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.1
  have c120232 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.1
  have c120233 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.1
  have c120234 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120235 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120236 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1972_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_2
  simp only [← c120231, ← c120232, ← c120233] at e_1972_2
  have e_1972_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_3
  simp only [← c120234, ← c120235, k2, ← c120236] at e_1972_3
  have hr := e_1972_3
  simp only [e_1972_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1972 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16576)) := by
  have c120237 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120238 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120239 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120240 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120241 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120242 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1972_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_4
  simp only [← c120237, ← c120238, ← c120239] at e_1972_4
  have e_1972_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_5
  simp only [← c120240, ← c120241, k2, ← c120242] at e_1972_5
  have hr := e_1972_5
  simp only [e_1972_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1972 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16577)) := by
  have c120243 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120244 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120245 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120246 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120247 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120248 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1972_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_6
  simp only [← c120243, ← c120244, ← c120245] at e_1972_6
  have e_1972_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_7
  simp only [← c120246, ← c120247, k2, ← c120248] at e_1972_7
  have hr := e_1972_7
  simp only [e_1972_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1972 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16578)) := by
  have c120249 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120250 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120251 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120252 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120253 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120254 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1972_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_8
  simp only [← c120249, ← c120250, ← c120251] at e_1972_8
  have e_1972_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_9
  simp only [← c120252, ← c120253, k2, ← c120254] at e_1972_9
  have hr := e_1972_9
  simp only [e_1972_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1972 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16579)) := by
  have c120255 := (publicBatchWrapper86_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120256 := (publicBatchWrapper86_copies3758 a h).1
  have c120257 := (publicBatchWrapper86_copies3758 a h).2.1
  have c120258 := (publicBatchWrapper86_copies3758 a h).2.2.1
  have c120259 := (publicBatchWrapper86_copies3758 a h).2.2.2.1
  have c120260 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1972_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_10
  simp only [← c120255, ← c120256, ← c120257] at e_1972_10
  have e_1972_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_11
  simp only [← c120258, ← c120259, k2, ← c120260] at e_1972_11
  have hr := e_1972_11
  simp only [e_1972_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1972 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16580)) := by
  have c120261 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.1
  have c120262 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.1
  have c120263 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.1
  have c120264 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.1
  have c120265 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.1
  have c120266 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1972_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_12
  simp only [← c120261, ← c120262, ← c120263] at e_1972_12
  have e_1972_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_13
  simp only [← c120264, ← c120265, k2, ← c120266] at e_1972_13
  have hr := e_1972_13
  simp only [e_1972_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1972 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16581)) := by
  have c120267 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120268 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120269 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120270 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120271 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120272 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1972_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_14
  simp only [← c120267, ← c120268, ← c120269] at e_1972_14
  have e_1972_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_15
  simp only [← c120270, ← c120271, k2, ← c120272] at e_1972_15
  have hr := e_1972_15
  simp only [e_1972_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1972 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16582)) := by
  have c120273 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120274 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120275 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120276 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120277 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120278 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1972_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_16
  simp only [← c120273, ← c120274, ← c120275] at e_1972_16
  have e_1972_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_17
  simp only [← c120276, ← c120277, k2, ← c120278] at e_1972_17
  have hr := e_1972_17
  simp only [e_1972_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1972 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16583)) := by
  have c120279 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120280 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120281 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120282 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120283 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120284 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1972_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_18
  simp only [← c120279, ← c120280, ← c120281] at e_1972_18
  have e_1972_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1972 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_19
  simp only [← c120282, ← c120283, k2, ← c120284] at e_1972_19
  have hr := e_1972_19
  simp only [e_1972_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1973 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16584)) := by
  have c120285 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120286 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120287 := (publicBatchWrapper86_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120288 := (publicBatchWrapper86_copies3759 a h).1
  have c120289 := (publicBatchWrapper86_copies3759 a h).2.1
  have c120290 := (publicBatchWrapper86_copies3759 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1973_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_0
  simp only [← c120285, ← c120286, ← c120287] at e_1973_0
  have e_1973_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_1
  simp only [← c120288, ← c120289, k2, ← c120290] at e_1973_1
  have hr := e_1973_1
  simp only [e_1973_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1973 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16585)) := by
  have c120291 := (publicBatchWrapper86_copies3759 a h).2.2.2.1
  have c120292 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.1
  have c120293 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.1
  have c120294 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.1
  have c120295 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.1
  have c120296 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1973_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_2
  simp only [← c120291, ← c120292, ← c120293] at e_1973_2
  have e_1973_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_3
  simp only [← c120294, ← c120295, k2, ← c120296] at e_1973_3
  have hr := e_1973_3
  simp only [e_1973_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1973 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16586)) := by
  have c120297 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.1
  have c120298 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120299 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120300 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120301 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120302 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1973_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_4
  simp only [← c120297, ← c120298, ← c120299] at e_1973_4
  have e_1973_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_5
  simp only [← c120300, ← c120301, k2, ← c120302] at e_1973_5
  have hr := e_1973_5
  simp only [e_1973_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1973 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16587)) := by
  have c120303 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120304 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120305 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120306 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120307 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120308 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1973_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_6
  simp only [← c120303, ← c120304, ← c120305] at e_1973_6
  have e_1973_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_7
  simp only [← c120306, ← c120307, k2, ← c120308] at e_1973_7
  have hr := e_1973_7
  simp only [e_1973_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1973 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16588)) := by
  have c120309 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120310 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120311 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120312 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120313 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120314 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1973_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_8
  simp only [← c120309, ← c120310, ← c120311] at e_1973_8
  have e_1973_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_9
  simp only [← c120312, ← c120313, k2, ← c120314] at e_1973_9
  have hr := e_1973_9
  simp only [e_1973_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1973 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16589)) := by
  have c120315 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120316 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120317 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120318 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120319 := (publicBatchWrapper86_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120320 := (publicBatchWrapper86_copies3760 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1973_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_10
  simp only [← c120315, ← c120316, ← c120317] at e_1973_10
  have e_1973_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_11
  simp only [← c120318, ← c120319, k2, ← c120320] at e_1973_11
  have hr := e_1973_11
  simp only [e_1973_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1973 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16590)) := by
  have c120321 := (publicBatchWrapper86_copies3760 a h).2.1
  have c120322 := (publicBatchWrapper86_copies3760 a h).2.2.1
  have c120323 := (publicBatchWrapper86_copies3760 a h).2.2.2.1
  have c120324 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.1
  have c120325 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.1
  have c120326 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1973_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_12
  simp only [← c120321, ← c120322, ← c120323] at e_1973_12
  have e_1973_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_13
  simp only [← c120324, ← c120325, k2, ← c120326] at e_1973_13
  have hr := e_1973_13
  simp only [e_1973_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1973 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16591)) := by
  have c120327 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.1
  have c120328 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.1
  have c120329 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.1
  have c120330 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120331 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120332 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1973_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_14
  simp only [← c120327, ← c120328, ← c120329] at e_1973_14
  have e_1973_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_15
  simp only [← c120330, ← c120331, k2, ← c120332] at e_1973_15
  have hr := e_1973_15
  simp only [e_1973_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1973 71) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16592)) := by
  have c120333 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120334 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120335 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120336 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120337 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120338 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1973_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_16
  simp only [← c120333, ← c120334, ← c120335] at e_1973_16
  have e_1973_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_17
  simp only [← c120336, ← c120337, k2, ← c120338] at e_1973_17
  have hr := e_1973_17
  simp only [e_1973_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1973 79) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16593)) := by
  have c120339 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120340 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120341 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120342 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120343 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120344 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1973_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_18
  simp only [← c120339, ← c120340, ← c120341] at e_1973_18
  have e_1973_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1973 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_19
  simp only [← c120342, ← c120343, k2, ← c120344] at e_1973_19
  have hr := e_1973_19
  simp only [e_1973_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 7) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16594)) := by
  have c120345 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120346 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120347 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120348 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120349 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120350 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_0
  simp only [← c120345, ← c120346, ← c120347] at e_1974_0
  have e_1974_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_1
  simp only [← c120348, ← c120349, k2, ← c120350] at e_1974_1
  have hr := e_1974_1
  simp only [e_1974_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 15) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16595)) := by
  have c120351 := (publicBatchWrapper86_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120352 := (publicBatchWrapper86_copies3761 a h).1
  have c120353 := (publicBatchWrapper86_copies3761 a h).2.1
  have c120354 := (publicBatchWrapper86_copies3761 a h).2.2.1
  have c120355 := (publicBatchWrapper86_copies3761 a h).2.2.2.1
  have c120356 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_2
  simp only [← c120351, ← c120352, ← c120353] at e_1974_2
  have e_1974_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_3
  simp only [← c120354, ← c120355, k2, ← c120356] at e_1974_3
  have hr := e_1974_3
  simp only [e_1974_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 23) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16596)) := by
  have c120357 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.1
  have c120358 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.1
  have c120359 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.1
  have c120360 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.1
  have c120361 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.1
  have c120362 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_4
  simp only [← c120357, ← c120358, ← c120359] at e_1974_4
  have e_1974_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_5
  simp only [← c120360, ← c120361, k2, ← c120362] at e_1974_5
  have hr := e_1974_5
  simp only [e_1974_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 31) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16597)) := by
  have c120363 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120364 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120365 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120366 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120367 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120368 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_6
  simp only [← c120363, ← c120364, ← c120365] at e_1974_6
  have e_1974_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_7
  simp only [← c120366, ← c120367, k2, ← c120368] at e_1974_7
  have hr := e_1974_7
  simp only [e_1974_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 39) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16598)) := by
  have c120369 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120370 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120371 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120372 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120373 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120374 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_8
  simp only [← c120369, ← c120370, ← c120371] at e_1974_8
  have e_1974_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_9
  simp only [← c120372, ← c120373, k2, ← c120374] at e_1974_9
  have hr := e_1974_9
  simp only [e_1974_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 47) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16599)) := by
  have c120375 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120376 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120377 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120378 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120379 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120380 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_10
  simp only [← c120375, ← c120376, ← c120377] at e_1974_10
  have e_1974_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_11
  simp only [← c120378, ← c120379, k2, ← c120380] at e_1974_11
  have hr := e_1974_11
  simp only [e_1974_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 55) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16600)) := by
  have c120381 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120382 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120383 := (publicBatchWrapper86_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120384 := (publicBatchWrapper86_copies3762 a h).1
  have c120385 := (publicBatchWrapper86_copies3762 a h).2.1
  have c120386 := (publicBatchWrapper86_copies3762 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_12
  simp only [← c120381, ← c120382, ← c120383] at e_1974_12
  have e_1974_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_13
  simp only [← c120384, ← c120385, k2, ← c120386] at e_1974_13
  have hr := e_1974_13
  simp only [e_1974_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 63) = bselect (a (.wire 80 35)) (a (.virt 200)) (a (.virt 16601)) := by
  have c120387 := (publicBatchWrapper86_copies3762 a h).2.2.2.1
  have c120388 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.1
  have c120389 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.1
  have c120390 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.1
  have c120391 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.1
  have c120392 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_14
  simp only [← c120387, ← c120388, ← c120389] at e_1974_14
  have e_1974_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_15
  simp only [← c120390, ← c120391, k2, ← c120392] at e_1974_15
  have hr := e_1974_15
  simp only [e_1974_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16770)) := by
  have c120393 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.1
  have c120394 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120395 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120396 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120397 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120398 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_16
  simp only [← c120393, ← c120394, ← c120395] at e_1974_16
  have e_1974_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_17
  simp only [← c120396, ← c120397, k2, ← c120398] at e_1974_17
  have hr := e_1974_17
  simp only [e_1974_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1974 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16771)) := by
  have c120399 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120400 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120401 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120402 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120403 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120404 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1974_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_18
  simp only [← c120399, ← c120400, ← c120401] at e_1974_18
  have e_1974_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1974 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1974_19
  simp only [← c120402, ← c120403, k2, ← c120404] at e_1974_19
  have hr := e_1974_19
  simp only [e_1974_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16772)) := by
  have c120405 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120406 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120407 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120408 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120409 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120410 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_0
  simp only [← c120405, ← c120406, ← c120407] at e_1975_0
  have e_1975_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_1
  simp only [← c120408, ← c120409, k2, ← c120410] at e_1975_1
  have hr := e_1975_1
  simp only [e_1975_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16773)) := by
  have c120411 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120412 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120413 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120414 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120415 := (publicBatchWrapper86_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120416 := (publicBatchWrapper86_copies3763 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_2
  simp only [← c120411, ← c120412, ← c120413] at e_1975_2
  have e_1975_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_3
  simp only [← c120414, ← c120415, k2, ← c120416] at e_1975_3
  have hr := e_1975_3
  simp only [e_1975_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16774)) := by
  have c120417 := (publicBatchWrapper86_copies3763 a h).2.1
  have c120418 := (publicBatchWrapper86_copies3763 a h).2.2.1
  have c120419 := (publicBatchWrapper86_copies3763 a h).2.2.2.1
  have c120420 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.1
  have c120421 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.1
  have c120422 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_4
  simp only [← c120417, ← c120418, ← c120419] at e_1975_4
  have e_1975_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_5
  simp only [← c120420, ← c120421, k2, ← c120422] at e_1975_5
  have hr := e_1975_5
  simp only [e_1975_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16775)) := by
  have c120423 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.1
  have c120424 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.1
  have c120425 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.1
  have c120426 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120427 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120428 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_6
  simp only [← c120423, ← c120424, ← c120425] at e_1975_6
  have e_1975_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_7
  simp only [← c120426, ← c120427, k2, ← c120428] at e_1975_7
  have hr := e_1975_7
  simp only [e_1975_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16776)) := by
  have c120429 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120430 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120431 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120432 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120433 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120434 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_8
  simp only [← c120429, ← c120430, ← c120431] at e_1975_8
  have e_1975_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_9
  simp only [← c120432, ← c120433, k2, ← c120434] at e_1975_9
  have hr := e_1975_9
  simp only [e_1975_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16777)) := by
  have c120435 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120436 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120437 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120438 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120439 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120440 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_10
  simp only [← c120435, ← c120436, ← c120437] at e_1975_10
  have e_1975_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_11
  simp only [← c120438, ← c120439, k2, ← c120440] at e_1975_11
  have hr := e_1975_11
  simp only [e_1975_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16778)) := by
  have c120441 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120442 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120443 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120444 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120445 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120446 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_12
  simp only [← c120441, ← c120442, ← c120443] at e_1975_12
  have e_1975_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_13
  simp only [← c120444, ← c120445, k2, ← c120446] at e_1975_13
  have hr := e_1975_13
  simp only [e_1975_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16779)) := by
  have c120447 := (publicBatchWrapper86_copies3763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120448 := (publicBatchWrapper86_copies3764 a h).1
  have c120449 := (publicBatchWrapper86_copies3764 a h).2.1
  have c120450 := (publicBatchWrapper86_copies3764 a h).2.2.1
  have c120451 := (publicBatchWrapper86_copies3764 a h).2.2.2.1
  have c120452 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_14
  simp only [← c120447, ← c120448, ← c120449] at e_1975_14
  have e_1975_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_15
  simp only [← c120450, ← c120451, k2, ← c120452] at e_1975_15
  have hr := e_1975_15
  simp only [e_1975_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16780)) := by
  have c120453 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.1
  have c120454 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.1
  have c120455 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.1
  have c120456 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.1
  have c120457 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.1
  have c120458 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_16
  simp only [← c120453, ← c120454, ← c120455] at e_1975_16
  have e_1975_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_17
  simp only [← c120456, ← c120457, k2, ← c120458] at e_1975_17
  have hr := e_1975_17
  simp only [e_1975_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1975 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16781)) := by
  have c120459 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120460 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120461 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120462 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120463 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120464 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1975_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_18
  simp only [← c120459, ← c120460, ← c120461] at e_1975_18
  have e_1975_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1975 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1975_19
  simp only [← c120462, ← c120463, k2, ← c120464] at e_1975_19
  have hr := e_1975_19
  simp only [e_1975_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16782)) := by
  have c120465 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120466 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120467 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120468 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120469 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120470 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_0
  simp only [← c120465, ← c120466, ← c120467] at e_1976_0
  have e_1976_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_1
  simp only [← c120468, ← c120469, k2, ← c120470] at e_1976_1
  have hr := e_1976_1
  simp only [e_1976_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16783)) := by
  have c120471 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120472 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120473 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120474 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120475 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120476 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_2
  simp only [← c120471, ← c120472, ← c120473] at e_1976_2
  have e_1976_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_3
  simp only [← c120474, ← c120475, k2, ← c120476] at e_1976_3
  have hr := e_1976_3
  simp only [e_1976_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16784)) := by
  have c120477 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120478 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120479 := (publicBatchWrapper86_copies3764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120480 := (publicBatchWrapper86_copies3765 a h).1
  have c120481 := (publicBatchWrapper86_copies3765 a h).2.1
  have c120482 := (publicBatchWrapper86_copies3765 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_4
  simp only [← c120477, ← c120478, ← c120479] at e_1976_4
  have e_1976_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_5
  simp only [← c120480, ← c120481, k2, ← c120482] at e_1976_5
  have hr := e_1976_5
  simp only [e_1976_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16785)) := by
  have c120483 := (publicBatchWrapper86_copies3765 a h).2.2.2.1
  have c120484 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.1
  have c120485 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.1
  have c120486 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.1
  have c120487 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.1
  have c120488 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_6
  simp only [← c120483, ← c120484, ← c120485] at e_1976_6
  have e_1976_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_7
  simp only [← c120486, ← c120487, k2, ← c120488] at e_1976_7
  have hr := e_1976_7
  simp only [e_1976_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16786)) := by
  have c120489 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.1
  have c120490 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120491 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120492 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120493 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120494 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_8
  simp only [← c120489, ← c120490, ← c120491] at e_1976_8
  have e_1976_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_9
  simp only [← c120492, ← c120493, k2, ← c120494] at e_1976_9
  have hr := e_1976_9
  simp only [e_1976_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16787)) := by
  have c120495 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120496 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120497 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120498 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120499 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120500 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_10
  simp only [← c120495, ← c120496, ← c120497] at e_1976_10
  have e_1976_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_11
  simp only [← c120498, ← c120499, k2, ← c120500] at e_1976_11
  have hr := e_1976_11
  simp only [e_1976_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16788)) := by
  have c120501 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120502 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120503 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120504 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120505 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120506 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_12
  simp only [← c120501, ← c120502, ← c120503] at e_1976_12
  have e_1976_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_13
  simp only [← c120504, ← c120505, k2, ← c120506] at e_1976_13
  have hr := e_1976_13
  simp only [e_1976_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16789)) := by
  have c120507 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120508 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120509 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120510 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120511 := (publicBatchWrapper86_copies3765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120512 := (publicBatchWrapper86_copies3766 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_14
  simp only [← c120507, ← c120508, ← c120509] at e_1976_14
  have e_1976_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_15
  simp only [← c120510, ← c120511, k2, ← c120512] at e_1976_15
  have hr := e_1976_15
  simp only [e_1976_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16790)) := by
  have c120513 := (publicBatchWrapper86_copies3766 a h).2.1
  have c120514 := (publicBatchWrapper86_copies3766 a h).2.2.1
  have c120515 := (publicBatchWrapper86_copies3766 a h).2.2.2.1
  have c120516 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.1
  have c120517 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.1
  have c120518 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_16
  simp only [← c120513, ← c120514, ← c120515] at e_1976_16
  have e_1976_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_17
  simp only [← c120516, ← c120517, k2, ← c120518] at e_1976_17
  have hr := e_1976_17
  simp only [e_1976_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1976 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16791)) := by
  have c120519 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.1
  have c120520 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.1
  have c120521 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.1
  have c120522 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120523 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120524 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1976_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_18
  simp only [← c120519, ← c120520, ← c120521] at e_1976_18
  have e_1976_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1976 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1976_19
  simp only [← c120522, ← c120523, k2, ← c120524] at e_1976_19
  have hr := e_1976_19
  simp only [e_1976_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16792)) := by
  have c120525 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120526 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120527 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120528 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120529 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120530 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_0
  simp only [← c120525, ← c120526, ← c120527] at e_1977_0
  have e_1977_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_1
  simp only [← c120528, ← c120529, k2, ← c120530] at e_1977_1
  have hr := e_1977_1
  simp only [e_1977_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16793)) := by
  have c120531 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120532 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120533 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120534 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120535 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120536 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_2
  simp only [← c120531, ← c120532, ← c120533] at e_1977_2
  have e_1977_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_3
  simp only [← c120534, ← c120535, k2, ← c120536] at e_1977_3
  have hr := e_1977_3
  simp only [e_1977_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16794)) := by
  have c120537 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120538 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120539 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120540 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120541 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120542 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_4
  simp only [← c120537, ← c120538, ← c120539] at e_1977_4
  have e_1977_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_5
  simp only [← c120540, ← c120541, k2, ← c120542] at e_1977_5
  have hr := e_1977_5
  simp only [e_1977_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16795)) := by
  have c120543 := (publicBatchWrapper86_copies3766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120544 := (publicBatchWrapper86_copies3767 a h).1
  have c120545 := (publicBatchWrapper86_copies3767 a h).2.1
  have c120546 := (publicBatchWrapper86_copies3767 a h).2.2.1
  have c120547 := (publicBatchWrapper86_copies3767 a h).2.2.2.1
  have c120548 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_6
  simp only [← c120543, ← c120544, ← c120545] at e_1977_6
  have e_1977_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_7
  simp only [← c120546, ← c120547, k2, ← c120548] at e_1977_7
  have hr := e_1977_7
  simp only [e_1977_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16796)) := by
  have c120549 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.1
  have c120550 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.1
  have c120551 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.1
  have c120552 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.1
  have c120553 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.1
  have c120554 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_8
  simp only [← c120549, ← c120550, ← c120551] at e_1977_8
  have e_1977_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_9
  simp only [← c120552, ← c120553, k2, ← c120554] at e_1977_9
  have hr := e_1977_9
  simp only [e_1977_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16797)) := by
  have c120555 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120556 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120557 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120558 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120559 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120560 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_10
  simp only [← c120555, ← c120556, ← c120557] at e_1977_10
  have e_1977_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_11
  simp only [← c120558, ← c120559, k2, ← c120560] at e_1977_11
  have hr := e_1977_11
  simp only [e_1977_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16798)) := by
  have c120561 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120562 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120563 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120564 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120565 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120566 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_12
  simp only [← c120561, ← c120562, ← c120563] at e_1977_12
  have e_1977_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_13
  simp only [← c120564, ← c120565, k2, ← c120566] at e_1977_13
  have hr := e_1977_13
  simp only [e_1977_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16799)) := by
  have c120567 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120568 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120569 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120570 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120571 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120572 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_14
  simp only [← c120567, ← c120568, ← c120569] at e_1977_14
  have e_1977_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_15
  simp only [← c120570, ← c120571, k2, ← c120572] at e_1977_15
  have hr := e_1977_15
  simp only [e_1977_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16800)) := by
  have c120573 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120574 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120575 := (publicBatchWrapper86_copies3767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120576 := (publicBatchWrapper86_copies3768 a h).1
  have c120577 := (publicBatchWrapper86_copies3768 a h).2.1
  have c120578 := (publicBatchWrapper86_copies3768 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_16
  simp only [← c120573, ← c120574, ← c120575] at e_1977_16
  have e_1977_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_17
  simp only [← c120576, ← c120577, k2, ← c120578] at e_1977_17
  have hr := e_1977_17
  simp only [e_1977_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1977 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16801)) := by
  have c120579 := (publicBatchWrapper86_copies3768 a h).2.2.2.1
  have c120580 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.1
  have c120581 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.1
  have c120582 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.1
  have c120583 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.1
  have c120584 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1977_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_18
  simp only [← c120579, ← c120580, ← c120581] at e_1977_18
  have e_1977_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1977 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1977_19
  simp only [← c120582, ← c120583, k2, ← c120584] at e_1977_19
  have hr := e_1977_19
  simp only [e_1977_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16970)) := by
  have c120585 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.1
  have c120586 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120587 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120588 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120589 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120590 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_0
  simp only [← c120585, ← c120586, ← c120587] at e_1978_0
  have e_1978_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_1
  simp only [← c120588, ← c120589, k2, ← c120590] at e_1978_1
  have hr := e_1978_1
  simp only [e_1978_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16971)) := by
  have c120591 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120592 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120593 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120594 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120595 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120596 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_2
  simp only [← c120591, ← c120592, ← c120593] at e_1978_2
  have e_1978_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_3
  simp only [← c120594, ← c120595, k2, ← c120596] at e_1978_3
  have hr := e_1978_3
  simp only [e_1978_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16972)) := by
  have c120597 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120598 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120599 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120600 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120601 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120602 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_4
  simp only [← c120597, ← c120598, ← c120599] at e_1978_4
  have e_1978_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_5
  simp only [← c120600, ← c120601, k2, ← c120602] at e_1978_5
  have hr := e_1978_5
  simp only [e_1978_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16973)) := by
  have c120603 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120604 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120605 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120606 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120607 := (publicBatchWrapper86_copies3768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120608 := (publicBatchWrapper86_copies3769 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_6
  simp only [← c120603, ← c120604, ← c120605] at e_1978_6
  have e_1978_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_7
  simp only [← c120606, ← c120607, k2, ← c120608] at e_1978_7
  have hr := e_1978_7
  simp only [e_1978_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16974)) := by
  have c120609 := (publicBatchWrapper86_copies3769 a h).2.1
  have c120610 := (publicBatchWrapper86_copies3769 a h).2.2.1
  have c120611 := (publicBatchWrapper86_copies3769 a h).2.2.2.1
  have c120612 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.1
  have c120613 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.1
  have c120614 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_8
  simp only [← c120609, ← c120610, ← c120611] at e_1978_8
  have e_1978_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_9
  simp only [← c120612, ← c120613, k2, ← c120614] at e_1978_9
  have hr := e_1978_9
  simp only [e_1978_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16975)) := by
  have c120615 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.1
  have c120616 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.1
  have c120617 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.1
  have c120618 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120619 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120620 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_10
  simp only [← c120615, ← c120616, ← c120617] at e_1978_10
  have e_1978_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_11
  simp only [← c120618, ← c120619, k2, ← c120620] at e_1978_11
  have hr := e_1978_11
  simp only [e_1978_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16976)) := by
  have c120621 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120622 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120623 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120624 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120625 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120626 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_12
  simp only [← c120621, ← c120622, ← c120623] at e_1978_12
  have e_1978_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_13
  simp only [← c120624, ← c120625, k2, ← c120626] at e_1978_13
  have hr := e_1978_13
  simp only [e_1978_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16977)) := by
  have c120627 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120628 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120629 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120630 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120631 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120632 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_14
  simp only [← c120627, ← c120628, ← c120629] at e_1978_14
  have e_1978_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_15
  simp only [← c120630, ← c120631, k2, ← c120632] at e_1978_15
  have hr := e_1978_15
  simp only [e_1978_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16978)) := by
  have c120633 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120634 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120635 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120636 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120637 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120638 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_16
  simp only [← c120633, ← c120634, ← c120635] at e_1978_16
  have e_1978_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_17
  simp only [← c120636, ← c120637, k2, ← c120638] at e_1978_17
  have hr := e_1978_17
  simp only [e_1978_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1978 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16979)) := by
  have c120639 := (publicBatchWrapper86_copies3769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120640 := (publicBatchWrapper86_copies3770 a h).1
  have c120641 := (publicBatchWrapper86_copies3770 a h).2.1
  have c120642 := (publicBatchWrapper86_copies3770 a h).2.2.1
  have c120643 := (publicBatchWrapper86_copies3770 a h).2.2.2.1
  have c120644 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1978_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_18
  simp only [← c120639, ← c120640, ← c120641] at e_1978_18
  have e_1978_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1978 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1978_19
  simp only [← c120642, ← c120643, k2, ← c120644] at e_1978_19
  have hr := e_1978_19
  simp only [e_1978_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16980)) := by
  have c120645 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.1
  have c120646 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.1
  have c120647 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.1
  have c120648 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.1
  have c120649 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.1
  have c120650 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_0
  simp only [← c120645, ← c120646, ← c120647] at e_1979_0
  have e_1979_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_1
  simp only [← c120648, ← c120649, k2, ← c120650] at e_1979_1
  have hr := e_1979_1
  simp only [e_1979_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16981)) := by
  have c120651 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120652 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120653 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120654 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120655 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120656 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_2
  simp only [← c120651, ← c120652, ← c120653] at e_1979_2
  have e_1979_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_3
  simp only [← c120654, ← c120655, k2, ← c120656] at e_1979_3
  have hr := e_1979_3
  simp only [e_1979_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16982)) := by
  have c120657 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120658 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120659 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120660 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120661 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120662 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_4
  simp only [← c120657, ← c120658, ← c120659] at e_1979_4
  have e_1979_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_5
  simp only [← c120660, ← c120661, k2, ← c120662] at e_1979_5
  have hr := e_1979_5
  simp only [e_1979_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16983)) := by
  have c120663 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120664 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120665 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120666 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120667 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120668 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_6
  simp only [← c120663, ← c120664, ← c120665] at e_1979_6
  have e_1979_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_7
  simp only [← c120666, ← c120667, k2, ← c120668] at e_1979_7
  have hr := e_1979_7
  simp only [e_1979_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16984)) := by
  have c120669 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120670 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120671 := (publicBatchWrapper86_copies3770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120672 := (publicBatchWrapper86_copies3771 a h).1
  have c120673 := (publicBatchWrapper86_copies3771 a h).2.1
  have c120674 := (publicBatchWrapper86_copies3771 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_8
  simp only [← c120669, ← c120670, ← c120671] at e_1979_8
  have e_1979_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_9
  simp only [← c120672, ← c120673, k2, ← c120674] at e_1979_9
  have hr := e_1979_9
  simp only [e_1979_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16985)) := by
  have c120675 := (publicBatchWrapper86_copies3771 a h).2.2.2.1
  have c120676 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.1
  have c120677 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.1
  have c120678 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.1
  have c120679 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.1
  have c120680 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_10
  simp only [← c120675, ← c120676, ← c120677] at e_1979_10
  have e_1979_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_11
  simp only [← c120678, ← c120679, k2, ← c120680] at e_1979_11
  have hr := e_1979_11
  simp only [e_1979_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16986)) := by
  have c120681 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.1
  have c120682 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120683 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120684 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120685 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120686 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_12
  simp only [← c120681, ← c120682, ← c120683] at e_1979_12
  have e_1979_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_13
  simp only [← c120684, ← c120685, k2, ← c120686] at e_1979_13
  have hr := e_1979_13
  simp only [e_1979_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16987)) := by
  have c120687 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120688 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120689 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120690 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120691 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120692 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_14
  simp only [← c120687, ← c120688, ← c120689] at e_1979_14
  have e_1979_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_15
  simp only [← c120690, ← c120691, k2, ← c120692] at e_1979_15
  have hr := e_1979_15
  simp only [e_1979_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16988)) := by
  have c120693 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120694 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120695 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120696 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120697 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120698 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_16
  simp only [← c120693, ← c120694, ← c120695] at e_1979_16
  have e_1979_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_17
  simp only [← c120696, ← c120697, k2, ← c120698] at e_1979_17
  have hr := e_1979_17
  simp only [e_1979_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1979 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16989)) := by
  have c120699 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120700 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120701 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120702 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120703 := (publicBatchWrapper86_copies3771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120704 := (publicBatchWrapper86_copies3772 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1979_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_18
  simp only [← c120699, ← c120700, ← c120701] at e_1979_18
  have e_1979_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1979 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1979_19
  simp only [← c120702, ← c120703, k2, ← c120704] at e_1979_19
  have hr := e_1979_19
  simp only [e_1979_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16990)) := by
  have c120705 := (publicBatchWrapper86_copies3772 a h).2.1
  have c120706 := (publicBatchWrapper86_copies3772 a h).2.2.1
  have c120707 := (publicBatchWrapper86_copies3772 a h).2.2.2.1
  have c120708 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.1
  have c120709 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.1
  have c120710 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_0
  simp only [← c120705, ← c120706, ← c120707] at e_1980_0
  have e_1980_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_1
  simp only [← c120708, ← c120709, k2, ← c120710] at e_1980_1
  have hr := e_1980_1
  simp only [e_1980_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16991)) := by
  have c120711 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.1
  have c120712 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.1
  have c120713 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.1
  have c120714 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120715 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120716 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_2
  simp only [← c120711, ← c120712, ← c120713] at e_1980_2
  have e_1980_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_3
  simp only [← c120714, ← c120715, k2, ← c120716] at e_1980_3
  have hr := e_1980_3
  simp only [e_1980_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16992)) := by
  have c120717 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120718 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120719 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120720 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120721 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120722 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_4
  simp only [← c120717, ← c120718, ← c120719] at e_1980_4
  have e_1980_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_5
  simp only [← c120720, ← c120721, k2, ← c120722] at e_1980_5
  have hr := e_1980_5
  simp only [e_1980_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16993)) := by
  have c120723 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120724 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120725 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120726 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120727 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120728 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_6
  simp only [← c120723, ← c120724, ← c120725] at e_1980_6
  have e_1980_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_7
  simp only [← c120726, ← c120727, k2, ← c120728] at e_1980_7
  have hr := e_1980_7
  simp only [e_1980_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16994)) := by
  have c120729 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120730 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120731 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120732 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120733 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120734 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_8
  simp only [← c120729, ← c120730, ← c120731] at e_1980_8
  have e_1980_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_9
  simp only [← c120732, ← c120733, k2, ← c120734] at e_1980_9
  have hr := e_1980_9
  simp only [e_1980_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16995)) := by
  have c120735 := (publicBatchWrapper86_copies3772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120736 := (publicBatchWrapper86_copies3773 a h).1
  have c120737 := (publicBatchWrapper86_copies3773 a h).2.1
  have c120738 := (publicBatchWrapper86_copies3773 a h).2.2.1
  have c120739 := (publicBatchWrapper86_copies3773 a h).2.2.2.1
  have c120740 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_10
  simp only [← c120735, ← c120736, ← c120737] at e_1980_10
  have e_1980_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_11
  simp only [← c120738, ← c120739, k2, ← c120740] at e_1980_11
  have hr := e_1980_11
  simp only [e_1980_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16996)) := by
  have c120741 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.1
  have c120742 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.1
  have c120743 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.1
  have c120744 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.1
  have c120745 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.1
  have c120746 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_12
  simp only [← c120741, ← c120742, ← c120743] at e_1980_12
  have e_1980_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_13
  simp only [← c120744, ← c120745, k2, ← c120746] at e_1980_13
  have hr := e_1980_13
  simp only [e_1980_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16997)) := by
  have c120747 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120748 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120749 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120750 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120751 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120752 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_14
  simp only [← c120747, ← c120748, ← c120749] at e_1980_14
  have e_1980_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_15
  simp only [← c120750, ← c120751, k2, ← c120752] at e_1980_15
  have hr := e_1980_15
  simp only [e_1980_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16998)) := by
  have c120753 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120754 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120755 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120756 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120757 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120758 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_16
  simp only [← c120753, ← c120754, ← c120755] at e_1980_16
  have e_1980_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_17
  simp only [← c120756, ← c120757, k2, ← c120758] at e_1980_17
  have hr := e_1980_17
  simp only [e_1980_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1980 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16999)) := by
  have c120759 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120760 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120761 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120762 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120763 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120764 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1980_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_18
  simp only [← c120759, ← c120760, ← c120761] at e_1980_18
  have e_1980_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1980 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1980_19
  simp only [← c120762, ← c120763, k2, ← c120764] at e_1980_19
  have hr := e_1980_19
  simp only [e_1980_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 17000)) := by
  have c120765 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120766 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120767 := (publicBatchWrapper86_copies3773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120768 := (publicBatchWrapper86_copies3774 a h).1
  have c120769 := (publicBatchWrapper86_copies3774 a h).2.1
  have c120770 := (publicBatchWrapper86_copies3774 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_0
  simp only [← c120765, ← c120766, ← c120767] at e_1981_0
  have e_1981_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_1
  simp only [← c120768, ← c120769, k2, ← c120770] at e_1981_1
  have hr := e_1981_1
  simp only [e_1981_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 17001)) := by
  have c120771 := (publicBatchWrapper86_copies3774 a h).2.2.2.1
  have c120772 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.1
  have c120773 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.1
  have c120774 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.1
  have c120775 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.1
  have c120776 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_2
  simp only [← c120771, ← c120772, ← c120773] at e_1981_2
  have e_1981_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_3
  simp only [← c120774, ← c120775, k2, ← c120776] at e_1981_3
  have hr := e_1981_3
  simp only [e_1981_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17170)) := by
  have c120777 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.1
  have c120778 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120779 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120780 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120781 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120782 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_4
  simp only [← c120777, ← c120778, ← c120779] at e_1981_4
  have e_1981_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_5
  simp only [← c120780, ← c120781, k2, ← c120782] at e_1981_5
  have hr := e_1981_5
  simp only [e_1981_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17171)) := by
  have c120783 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120784 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120785 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120786 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120787 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120788 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_6
  simp only [← c120783, ← c120784, ← c120785] at e_1981_6
  have e_1981_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_7
  simp only [← c120786, ← c120787, k2, ← c120788] at e_1981_7
  have hr := e_1981_7
  simp only [e_1981_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17172)) := by
  have c120789 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120790 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120791 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120792 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120793 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120794 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_8
  simp only [← c120789, ← c120790, ← c120791] at e_1981_8
  have e_1981_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_9
  simp only [← c120792, ← c120793, k2, ← c120794] at e_1981_9
  have hr := e_1981_9
  simp only [e_1981_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17173)) := by
  have c120795 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120796 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120797 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120798 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120799 := (publicBatchWrapper86_copies3774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120800 := (publicBatchWrapper86_copies3775 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_10
  simp only [← c120795, ← c120796, ← c120797] at e_1981_10
  have e_1981_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_11
  simp only [← c120798, ← c120799, k2, ← c120800] at e_1981_11
  have hr := e_1981_11
  simp only [e_1981_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17174)) := by
  have c120801 := (publicBatchWrapper86_copies3775 a h).2.1
  have c120802 := (publicBatchWrapper86_copies3775 a h).2.2.1
  have c120803 := (publicBatchWrapper86_copies3775 a h).2.2.2.1
  have c120804 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.1
  have c120805 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.1
  have c120806 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_12
  simp only [← c120801, ← c120802, ← c120803] at e_1981_12
  have e_1981_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_13
  simp only [← c120804, ← c120805, k2, ← c120806] at e_1981_13
  have hr := e_1981_13
  simp only [e_1981_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17175)) := by
  have c120807 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.1
  have c120808 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.1
  have c120809 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.1
  have c120810 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120811 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120812 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_14
  simp only [← c120807, ← c120808, ← c120809] at e_1981_14
  have e_1981_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_15
  simp only [← c120810, ← c120811, k2, ← c120812] at e_1981_15
  have hr := e_1981_15
  simp only [e_1981_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17176)) := by
  have c120813 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120814 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120815 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120816 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120817 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120818 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_16
  simp only [← c120813, ← c120814, ← c120815] at e_1981_16
  have e_1981_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_17
  simp only [← c120816, ← c120817, k2, ← c120818] at e_1981_17
  have hr := e_1981_17
  simp only [e_1981_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1981 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17177)) := by
  have c120819 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120820 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120821 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120822 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120823 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120824 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1981_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_18
  simp only [← c120819, ← c120820, ← c120821] at e_1981_18
  have e_1981_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1981 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1981_19
  simp only [← c120822, ← c120823, k2, ← c120824] at e_1981_19
  have hr := e_1981_19
  simp only [e_1981_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17178)) := by
  have c120825 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120826 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120827 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120828 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120829 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120830 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_0
  simp only [← c120825, ← c120826, ← c120827] at e_1982_0
  have e_1982_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_1
  simp only [← c120828, ← c120829, k2, ← c120830] at e_1982_1
  have hr := e_1982_1
  simp only [e_1982_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17179)) := by
  have c120831 := (publicBatchWrapper86_copies3775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120832 := (publicBatchWrapper86_copies3776 a h).1
  have c120833 := (publicBatchWrapper86_copies3776 a h).2.1
  have c120834 := (publicBatchWrapper86_copies3776 a h).2.2.1
  have c120835 := (publicBatchWrapper86_copies3776 a h).2.2.2.1
  have c120836 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_2
  simp only [← c120831, ← c120832, ← c120833] at e_1982_2
  have e_1982_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_3
  simp only [← c120834, ← c120835, k2, ← c120836] at e_1982_3
  have hr := e_1982_3
  simp only [e_1982_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17180)) := by
  have c120837 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.1
  have c120838 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.1
  have c120839 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.1
  have c120840 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.1
  have c120841 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.1
  have c120842 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_4
  simp only [← c120837, ← c120838, ← c120839] at e_1982_4
  have e_1982_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_5
  simp only [← c120840, ← c120841, k2, ← c120842] at e_1982_5
  have hr := e_1982_5
  simp only [e_1982_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17181)) := by
  have c120843 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120844 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120845 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120846 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120847 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120848 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_6
  simp only [← c120843, ← c120844, ← c120845] at e_1982_6
  have e_1982_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_7
  simp only [← c120846, ← c120847, k2, ← c120848] at e_1982_7
  have hr := e_1982_7
  simp only [e_1982_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17182)) := by
  have c120849 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120850 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120851 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120852 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120853 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120854 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_8
  simp only [← c120849, ← c120850, ← c120851] at e_1982_8
  have e_1982_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_9
  simp only [← c120852, ← c120853, k2, ← c120854] at e_1982_9
  have hr := e_1982_9
  simp only [e_1982_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17183)) := by
  have c120855 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120856 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120857 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120858 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120859 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120860 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_10
  simp only [← c120855, ← c120856, ← c120857] at e_1982_10
  have e_1982_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_11
  simp only [← c120858, ← c120859, k2, ← c120860] at e_1982_11
  have hr := e_1982_11
  simp only [e_1982_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17184)) := by
  have c120861 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120862 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120863 := (publicBatchWrapper86_copies3776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120864 := (publicBatchWrapper86_copies3777 a h).1
  have c120865 := (publicBatchWrapper86_copies3777 a h).2.1
  have c120866 := (publicBatchWrapper86_copies3777 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_12
  simp only [← c120861, ← c120862, ← c120863] at e_1982_12
  have e_1982_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_13
  simp only [← c120864, ← c120865, k2, ← c120866] at e_1982_13
  have hr := e_1982_13
  simp only [e_1982_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17185)) := by
  have c120867 := (publicBatchWrapper86_copies3777 a h).2.2.2.1
  have c120868 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.1
  have c120869 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.1
  have c120870 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.1
  have c120871 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.1
  have c120872 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_14
  simp only [← c120867, ← c120868, ← c120869] at e_1982_14
  have e_1982_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_15
  simp only [← c120870, ← c120871, k2, ← c120872] at e_1982_15
  have hr := e_1982_15
  simp only [e_1982_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17186)) := by
  have c120873 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.1
  have c120874 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120875 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120876 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120877 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120878 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_16
  simp only [← c120873, ← c120874, ← c120875] at e_1982_16
  have e_1982_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_17
  simp only [← c120876, ← c120877, k2, ← c120878] at e_1982_17
  have hr := e_1982_17
  simp only [e_1982_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1982 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17187)) := by
  have c120879 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120880 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120881 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120882 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120883 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120884 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1982_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_18
  simp only [← c120879, ← c120880, ← c120881] at e_1982_18
  have e_1982_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1982 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1982_19
  simp only [← c120882, ← c120883, k2, ← c120884] at e_1982_19
  have hr := e_1982_19
  simp only [e_1982_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17188)) := by
  have c120885 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120886 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120887 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120888 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120889 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120890 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_0
  simp only [← c120885, ← c120886, ← c120887] at e_1983_0
  have e_1983_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_1
  simp only [← c120888, ← c120889, k2, ← c120890] at e_1983_1
  have hr := e_1983_1
  simp only [e_1983_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17189)) := by
  have c120891 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120892 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120893 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120894 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120895 := (publicBatchWrapper86_copies3777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120896 := (publicBatchWrapper86_copies3778 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_2
  simp only [← c120891, ← c120892, ← c120893] at e_1983_2
  have e_1983_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_3
  simp only [← c120894, ← c120895, k2, ← c120896] at e_1983_3
  have hr := e_1983_3
  simp only [e_1983_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17190)) := by
  have c120897 := (publicBatchWrapper86_copies3778 a h).2.1
  have c120898 := (publicBatchWrapper86_copies3778 a h).2.2.1
  have c120899 := (publicBatchWrapper86_copies3778 a h).2.2.2.1
  have c120900 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.1
  have c120901 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.1
  have c120902 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_4
  simp only [← c120897, ← c120898, ← c120899] at e_1983_4
  have e_1983_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_5
  simp only [← c120900, ← c120901, k2, ← c120902] at e_1983_5
  have hr := e_1983_5
  simp only [e_1983_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17191)) := by
  have c120903 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.1
  have c120904 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.1
  have c120905 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.1
  have c120906 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120907 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120908 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_6
  simp only [← c120903, ← c120904, ← c120905] at e_1983_6
  have e_1983_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_7
  simp only [← c120906, ← c120907, k2, ← c120908] at e_1983_7
  have hr := e_1983_7
  simp only [e_1983_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17192)) := by
  have c120909 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120910 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120911 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120912 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120913 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120914 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_8
  simp only [← c120909, ← c120910, ← c120911] at e_1983_8
  have e_1983_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_9
  simp only [← c120912, ← c120913, k2, ← c120914] at e_1983_9
  have hr := e_1983_9
  simp only [e_1983_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17193)) := by
  have c120915 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120916 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120917 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120918 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120919 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120920 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_10
  simp only [← c120915, ← c120916, ← c120917] at e_1983_10
  have e_1983_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_11
  simp only [← c120918, ← c120919, k2, ← c120920] at e_1983_11
  have hr := e_1983_11
  simp only [e_1983_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17194)) := by
  have c120921 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120922 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120923 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120924 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120925 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120926 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_12
  simp only [← c120921, ← c120922, ← c120923] at e_1983_12
  have e_1983_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_13
  simp only [← c120924, ← c120925, k2, ← c120926] at e_1983_13
  have hr := e_1983_13
  simp only [e_1983_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17195)) := by
  have c120927 := (publicBatchWrapper86_copies3778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120928 := (publicBatchWrapper86_copies3779 a h).1
  have c120929 := (publicBatchWrapper86_copies3779 a h).2.1
  have c120930 := (publicBatchWrapper86_copies3779 a h).2.2.1
  have c120931 := (publicBatchWrapper86_copies3779 a h).2.2.2.1
  have c120932 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_14
  simp only [← c120927, ← c120928, ← c120929] at e_1983_14
  have e_1983_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_15
  simp only [← c120930, ← c120931, k2, ← c120932] at e_1983_15
  have hr := e_1983_15
  simp only [e_1983_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17196)) := by
  have c120933 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.1
  have c120934 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.1
  have c120935 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.1
  have c120936 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.1
  have c120937 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.1
  have c120938 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_16
  simp only [← c120933, ← c120934, ← c120935] at e_1983_16
  have e_1983_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_17
  simp only [← c120936, ← c120937, k2, ← c120938] at e_1983_17
  have hr := e_1983_17
  simp only [e_1983_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1983 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17197)) := by
  have c120939 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120940 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120941 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120942 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120943 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120944 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1983_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_18
  simp only [← c120939, ← c120940, ← c120941] at e_1983_18
  have e_1983_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row1983 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1983_19
  simp only [← c120942, ← c120943, k2, ← c120944] at e_1983_19
  have hr := e_1983_19
  simp only [e_1983_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1984 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17198)) := by
  have c120945 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120946 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120947 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120948 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120949 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120950 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1984_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_0
  simp only [← c120945, ← c120946, ← c120947] at e_1984_0
  have e_1984_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_1
  simp only [← c120948, ← c120949, k2, ← c120950] at e_1984_1
  have hr := e_1984_1
  simp only [e_1984_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1984 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17199)) := by
  have c120951 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120952 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120953 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120954 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120955 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120956 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1984_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_2
  simp only [← c120951, ← c120952, ← c120953] at e_1984_2
  have e_1984_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_3
  simp only [← c120954, ← c120955, k2, ← c120956] at e_1984_3
  have hr := e_1984_3
  simp only [e_1984_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1984 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17200)) := by
  have c120957 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120958 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120959 := (publicBatchWrapper86_copies3779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120960 := (publicBatchWrapper86_copies3780 a h).1
  have c120961 := (publicBatchWrapper86_copies3780 a h).2.1
  have c120962 := (publicBatchWrapper86_copies3780 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1984_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_4
  simp only [← c120957, ← c120958, ← c120959] at e_1984_4
  have e_1984_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_5
  simp only [← c120960, ← c120961, k2, ← c120962] at e_1984_5
  have hr := e_1984_5
  simp only [e_1984_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f19435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1984 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17201)) := by
  have c120963 := (publicBatchWrapper86_copies3780 a h).2.2.2.1
  have c120964 := (publicBatchWrapper86_copies3780 a h).2.2.2.2.1
  have c120965 := (publicBatchWrapper86_copies3780 a h).2.2.2.2.2.1
  have c120966 := (publicBatchWrapper86_copies3780 a h).2.2.2.2.2.2.1
  have c120967 := (publicBatchWrapper86_copies3780 a h).2.2.2.2.2.2.2.1
  have c120968 := (publicBatchWrapper86_copies3780 a h).2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1984_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_6
  simp only [← c120963, ← c120964, ← c120965] at e_1984_6
  have e_1984_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row1984 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1984_7
  simp only [← c120966, ← c120967, k2, ← c120968] at e_1984_7
  have hr := e_1984_7
  simp only [e_1984_6] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
