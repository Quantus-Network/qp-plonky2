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

theorem publicBatchWrapper86_f2870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16405)) (a (.wire 165 15)) (a (.virt 18883)) (a (.virt 18884)) := by
  have c21417 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.1
  have c21418 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21419 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21420 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21421 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21422 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21423 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21424 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21425 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21426 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21427 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21428 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21429 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21430 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21431 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21432 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21433 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_324_18 := arithEq_of_rows h (row := 324) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_324_18
  simp only [← c21417, k3, ← c21418, k3, ← c21419] at e_324_18
  have e_324_19 := arithEq_of_rows h (row := 324) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_324_19
  simp only [← c21420, ← c21421, k3, ← c21422] at e_324_19
  have e_325_1 := arithEq_of_rows h (row := 325) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_1
  simp only [← c21423, ← c21424, ← c21425] at e_325_1
  have e_325_2 := arithEq_of_rows h (row := 325) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_2
  simp only [← c21426, ← c21427, ← c21428] at e_325_2
  have e_327_0 := arithEq_of_rows h (row := 327) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_0
  simp only [← c21429, ← c21430, k3, ← c21431] at e_327_0
  refine ⟨?_, ?_⟩
  · have hc := e_325_1
    simp only [e_324_19] at hc
    linear_combination c21432.trans k2 - hc
  · have hc := e_327_0
    simp only [e_324_18, e_325_2, e_324_19] at hc
    linear_combination c21433.trans k2 - hc

theorem publicBatchWrapper86_f2871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16406)) (a (.wire 165 23)) (a (.virt 18885)) (a (.virt 18886)) := by
  have c21434 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21435 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21436 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21437 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21438 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21439 := (publicBatchWrapper86_copies669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21440 := (publicBatchWrapper86_copies670 a h).1
  have c21441 := (publicBatchWrapper86_copies670 a h).2.1
  have c21442 := (publicBatchWrapper86_copies670 a h).2.2.1
  have c21443 := (publicBatchWrapper86_copies670 a h).2.2.2.1
  have c21444 := (publicBatchWrapper86_copies670 a h).2.2.2.2.1
  have c21445 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.1
  have c21446 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.1
  have c21447 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.1
  have c21448 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.1
  have c21449 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.1
  have c21450 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_325_3 := arithEq_of_rows h (row := 325) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_3
  simp only [← c21440, ← c21441, ← c21442] at e_325_3
  have e_325_4 := arithEq_of_rows h (row := 325) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_4
  simp only [← c21443, ← c21444, ← c21445] at e_325_4
  have e_327_1 := arithEq_of_rows h (row := 327) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_1
  simp only [← c21434, k3, ← c21435, k3, ← c21436] at e_327_1
  have e_327_2 := arithEq_of_rows h (row := 327) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_2
  simp only [← c21437, ← c21438, k3, ← c21439] at e_327_2
  have e_327_3 := arithEq_of_rows h (row := 327) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_3
  simp only [← c21446, ← c21447, k3, ← c21448] at e_327_3
  refine ⟨?_, ?_⟩
  · have hc := e_325_3
    simp only [e_327_2] at hc
    linear_combination c21449.trans k2 - hc
  · have hc := e_327_3
    simp only [e_327_1, e_325_4, e_327_2] at hc
    linear_combination c21450.trans k2 - hc

theorem publicBatchWrapper86_f2872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16407)) (a (.wire 165 31)) (a (.virt 18887)) (a (.virt 18888)) := by
  have c21451 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21452 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21453 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21454 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21455 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21456 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21457 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21458 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21459 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21460 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21461 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21462 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21463 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21464 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21465 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21466 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21467 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_325_5 := arithEq_of_rows h (row := 325) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_5
  simp only [← c21457, ← c21458, ← c21459] at e_325_5
  have e_325_6 := arithEq_of_rows h (row := 325) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_6
  simp only [← c21460, ← c21461, ← c21462] at e_325_6
  have e_327_4 := arithEq_of_rows h (row := 327) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_4
  simp only [← c21451, k3, ← c21452, k3, ← c21453] at e_327_4
  have e_327_5 := arithEq_of_rows h (row := 327) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_5
  simp only [← c21454, ← c21455, k3, ← c21456] at e_327_5
  have e_327_6 := arithEq_of_rows h (row := 327) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_6
  simp only [← c21463, ← c21464, k3, ← c21465] at e_327_6
  refine ⟨?_, ?_⟩
  · have hc := e_325_5
    simp only [e_327_5] at hc
    linear_combination c21466.trans k2 - hc
  · have hc := e_327_6
    simp only [e_327_4, e_325_6, e_327_5] at hc
    linear_combination c21467.trans k2 - hc

theorem publicBatchWrapper86_f2873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16408)) (a (.wire 165 39)) (a (.virt 18889)) (a (.virt 18890)) := by
  have c21468 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21469 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21470 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21471 := (publicBatchWrapper86_copies670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21472 := (publicBatchWrapper86_copies671 a h).1
  have c21473 := (publicBatchWrapper86_copies671 a h).2.1
  have c21474 := (publicBatchWrapper86_copies671 a h).2.2.1
  have c21475 := (publicBatchWrapper86_copies671 a h).2.2.2.1
  have c21476 := (publicBatchWrapper86_copies671 a h).2.2.2.2.1
  have c21477 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.1
  have c21478 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.1
  have c21479 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.1
  have c21480 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.1
  have c21481 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.1
  have c21482 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21483 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21484 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_325_7 := arithEq_of_rows h (row := 325) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_7
  simp only [← c21474, ← c21475, ← c21476] at e_325_7
  have e_325_8 := arithEq_of_rows h (row := 325) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_8
  simp only [← c21477, ← c21478, ← c21479] at e_325_8
  have e_327_7 := arithEq_of_rows h (row := 327) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_7
  simp only [← c21468, k3, ← c21469, k3, ← c21470] at e_327_7
  have e_327_8 := arithEq_of_rows h (row := 327) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_8
  simp only [← c21471, ← c21472, k3, ← c21473] at e_327_8
  have e_327_9 := arithEq_of_rows h (row := 327) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_9
  simp only [← c21480, ← c21481, k3, ← c21482] at e_327_9
  refine ⟨?_, ?_⟩
  · have hc := e_325_7
    simp only [e_327_8] at hc
    linear_combination c21483.trans k2 - hc
  · have hc := e_327_9
    simp only [e_327_7, e_325_8, e_327_8] at hc
    linear_combination c21484.trans k2 - hc

theorem publicBatchWrapper86_f2874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 325 39) = band (a (.virt 18883)) (a (.virt 18885)) := by
  have c21485 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21486 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21487 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_325_9 := arithEq_of_rows h (row := 325) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_9
  simp only [← c21485, ← c21486, ← c21487] at e_325_9
  have hr := e_325_9
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 325 43) = band (a (.virt 18887)) (a (.virt 18889)) := by
  have c21488 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21489 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21490 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_325_10 := arithEq_of_rows h (row := 325) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_10
  simp only [← c21488, ← c21489, ← c21490] at e_325_10
  have hr := e_325_10
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 325 47) = band (a (.wire 325 39)) (a (.wire 325 43)) := by
  have c21491 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21492 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21493 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_325_11 := arithEq_of_rows h (row := 325) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_11
  simp only [← c21491, ← c21492, ← c21493] at e_325_11
  have hr := e_325_11
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2877 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 7) = bor (a (.wire 80 35)) (a (.wire 325 47)) := by
  have c21494 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21495 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21496 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21497 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21498 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21499 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_319_13 := arithEq_of_rows h (row := 319) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_13
  simp only [← c21494, ← c21495, ← c21496] at e_319_13
  have e_326_1 := arithEq_of_rows h (row := 326) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_1
  simp only [← c21497, ← c21498, k3, ← c21499] at e_326_1
  have hr := e_326_1
  simp only [e_319_13] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2878 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 7) = a (.virt 201) := by
  have c21500 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21500

theorem publicBatchWrapper86_f2879 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16603)) (a (.wire 165 55)) (a (.virt 18891)) (a (.virt 18892)) := by
  have c21501 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21502 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21503 := (publicBatchWrapper86_copies671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21504 := (publicBatchWrapper86_copies672 a h).1
  have c21505 := (publicBatchWrapper86_copies672 a h).2.1
  have c21506 := (publicBatchWrapper86_copies672 a h).2.2.1
  have c21507 := (publicBatchWrapper86_copies672 a h).2.2.2.1
  have c21508 := (publicBatchWrapper86_copies672 a h).2.2.2.2.1
  have c21509 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.1
  have c21510 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.1
  have c21511 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.1
  have c21512 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.1
  have c21513 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.1
  have c21514 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21515 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21516 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21517 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_325_12 := arithEq_of_rows h (row := 325) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_12
  simp only [← c21507, ← c21508, ← c21509] at e_325_12
  have e_325_13 := arithEq_of_rows h (row := 325) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_13
  simp only [← c21510, ← c21511, ← c21512] at e_325_13
  have e_327_10 := arithEq_of_rows h (row := 327) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_10
  simp only [← c21501, k3, ← c21502, k3, ← c21503] at e_327_10
  have e_327_11 := arithEq_of_rows h (row := 327) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_11
  simp only [← c21504, ← c21505, k3, ← c21506] at e_327_11
  have e_327_12 := arithEq_of_rows h (row := 327) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_12
  simp only [← c21513, ← c21514, k3, ← c21515] at e_327_12
  refine ⟨?_, ?_⟩
  · have hc := e_325_12
    simp only [e_327_11] at hc
    linear_combination c21516.trans k2 - hc
  · have hc := e_327_12
    simp only [e_327_10, e_325_13, e_327_11] at hc
    linear_combination c21517.trans k2 - hc

theorem publicBatchWrapper86_f2880 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 11) = bor (a (.wire 80 79)) (a (.virt 18891)) := by
  have c21518 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21519 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21520 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21521 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21522 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21523 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_319_14 := arithEq_of_rows h (row := 319) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_14
  simp only [← c21518, ← c21519, ← c21520] at e_319_14
  have e_326_2 := arithEq_of_rows h (row := 326) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_2
  simp only [← c21521, ← c21522, k3, ← c21523] at e_326_2
  have hr := e_326_2
  simp only [e_319_14] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2881 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 11) = a (.virt 201) := by
  have c21524 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21524

theorem publicBatchWrapper86_f2882 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16604)) (a (.wire 165 63)) (a (.virt 18893)) (a (.virt 18894)) := by
  have c21525 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21526 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21527 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21528 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21529 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21530 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21531 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21532 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21533 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21534 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21535 := (publicBatchWrapper86_copies672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21536 := (publicBatchWrapper86_copies673 a h).1
  have c21537 := (publicBatchWrapper86_copies673 a h).2.1
  have c21538 := (publicBatchWrapper86_copies673 a h).2.2.1
  have c21539 := (publicBatchWrapper86_copies673 a h).2.2.2.1
  have c21540 := (publicBatchWrapper86_copies673 a h).2.2.2.2.1
  have c21541 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_325_14 := arithEq_of_rows h (row := 325) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_14
  simp only [← c21531, ← c21532, ← c21533] at e_325_14
  have e_325_15 := arithEq_of_rows h (row := 325) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_15
  simp only [← c21534, ← c21535, ← c21536] at e_325_15
  have e_327_13 := arithEq_of_rows h (row := 327) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_13
  simp only [← c21525, k3, ← c21526, k3, ← c21527] at e_327_13
  have e_327_14 := arithEq_of_rows h (row := 327) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_14
  simp only [← c21528, ← c21529, k3, ← c21530] at e_327_14
  have e_327_15 := arithEq_of_rows h (row := 327) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_15
  simp only [← c21537, ← c21538, k3, ← c21539] at e_327_15
  refine ⟨?_, ?_⟩
  · have hc := e_325_14
    simp only [e_327_14] at hc
    linear_combination c21540.trans k2 - hc
  · have hc := e_327_15
    simp only [e_327_13, e_325_15, e_327_14] at hc
    linear_combination c21541.trans k2 - hc

theorem publicBatchWrapper86_f2883 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 15) = bor (a (.wire 80 79)) (a (.virt 18893)) := by
  have c21542 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.1
  have c21543 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.1
  have c21544 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.1
  have c21545 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.1
  have c21546 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21547 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_319_15 := arithEq_of_rows h (row := 319) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_15
  simp only [← c21542, ← c21543, ← c21544] at e_319_15
  have e_326_3 := arithEq_of_rows h (row := 326) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_3
  simp only [← c21545, ← c21546, k3, ← c21547] at e_326_3
  have hr := e_326_3
  simp only [e_319_15] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2884 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 15) = a (.virt 201) := by
  have c21548 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21548

theorem publicBatchWrapper86_f2885 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16605)) (a (.wire 165 15)) (a (.virt 18895)) (a (.virt 18896)) := by
  have c21549 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21550 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21551 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21552 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21553 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21554 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21555 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21556 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21557 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21558 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21559 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21560 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21561 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21562 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21563 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21564 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21565 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_325_16 := arithEq_of_rows h (row := 325) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_16
  simp only [← c21555, ← c21556, ← c21557] at e_325_16
  have e_325_17 := arithEq_of_rows h (row := 325) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_17
  simp only [← c21558, ← c21559, ← c21560] at e_325_17
  have e_327_16 := arithEq_of_rows h (row := 327) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_16
  simp only [← c21549, k3, ← c21550, k3, ← c21551] at e_327_16
  have e_327_17 := arithEq_of_rows h (row := 327) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_17
  simp only [← c21552, ← c21553, k3, ← c21554] at e_327_17
  have e_327_18 := arithEq_of_rows h (row := 327) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_18
  simp only [← c21561, ← c21562, k3, ← c21563] at e_327_18
  refine ⟨?_, ?_⟩
  · have hc := e_325_16
    simp only [e_327_17] at hc
    linear_combination c21564.trans k2 - hc
  · have hc := e_327_18
    simp only [e_327_16, e_325_17, e_327_17] at hc
    linear_combination c21565.trans k2 - hc

theorem publicBatchWrapper86_f2886 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16606)) (a (.wire 165 23)) (a (.virt 18897)) (a (.virt 18898)) := by
  have c21566 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21567 := (publicBatchWrapper86_copies673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21568 := (publicBatchWrapper86_copies674 a h).1
  have c21569 := (publicBatchWrapper86_copies674 a h).2.1
  have c21570 := (publicBatchWrapper86_copies674 a h).2.2.1
  have c21571 := (publicBatchWrapper86_copies674 a h).2.2.2.1
  have c21572 := (publicBatchWrapper86_copies674 a h).2.2.2.2.1
  have c21573 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.1
  have c21574 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.1
  have c21575 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.1
  have c21576 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.1
  have c21577 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.1
  have c21578 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21579 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21580 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21581 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21582 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_325_18 := arithEq_of_rows h (row := 325) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_18
  simp only [← c21572, ← c21573, ← c21574] at e_325_18
  have e_325_19 := arithEq_of_rows h (row := 325) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_325_19
  simp only [← c21575, ← c21576, ← c21577] at e_325_19
  have e_327_19 := arithEq_of_rows h (row := 327) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_327_19
  simp only [← c21566, k3, ← c21567, k3, ← c21568] at e_327_19
  have e_328_0 := arithEq_of_rows h (row := 328) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_0
  simp only [← c21569, ← c21570, k3, ← c21571] at e_328_0
  have e_328_1 := arithEq_of_rows h (row := 328) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_1
  simp only [← c21578, ← c21579, k3, ← c21580] at e_328_1
  refine ⟨?_, ?_⟩
  · have hc := e_325_18
    simp only [e_328_0] at hc
    linear_combination c21581.trans k2 - hc
  · have hc := e_328_1
    simp only [e_327_19, e_325_19, e_328_0] at hc
    linear_combination c21582.trans k2 - hc

theorem publicBatchWrapper86_f2887 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16607)) (a (.wire 165 31)) (a (.virt 18899)) (a (.virt 18900)) := by
  have c21583 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21584 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21585 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21586 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21587 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21588 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21589 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21590 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21591 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21592 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21593 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21594 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21595 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21596 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21597 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21598 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21599 := (publicBatchWrapper86_copies674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_328_2 := arithEq_of_rows h (row := 328) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_2
  simp only [← c21583, k3, ← c21584, k3, ← c21585] at e_328_2
  have e_328_3 := arithEq_of_rows h (row := 328) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_3
  simp only [← c21586, ← c21587, k3, ← c21588] at e_328_3
  have e_328_4 := arithEq_of_rows h (row := 328) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_4
  simp only [← c21595, ← c21596, k3, ← c21597] at e_328_4
  have e_329_0 := arithEq_of_rows h (row := 329) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_0
  simp only [← c21589, ← c21590, ← c21591] at e_329_0
  have e_329_1 := arithEq_of_rows h (row := 329) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_1
  simp only [← c21592, ← c21593, ← c21594] at e_329_1
  refine ⟨?_, ?_⟩
  · have hc := e_329_0
    simp only [e_328_3] at hc
    linear_combination c21598.trans k2 - hc
  · have hc := e_328_4
    simp only [e_328_2, e_329_1, e_328_3] at hc
    linear_combination c21599.trans k2 - hc

theorem publicBatchWrapper86_f2888 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16608)) (a (.wire 165 39)) (a (.virt 18901)) (a (.virt 18902)) := by
  have c21600 := (publicBatchWrapper86_copies675 a h).1
  have c21601 := (publicBatchWrapper86_copies675 a h).2.1
  have c21602 := (publicBatchWrapper86_copies675 a h).2.2.1
  have c21603 := (publicBatchWrapper86_copies675 a h).2.2.2.1
  have c21604 := (publicBatchWrapper86_copies675 a h).2.2.2.2.1
  have c21605 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.1
  have c21606 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.1
  have c21607 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.1
  have c21608 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.1
  have c21609 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.1
  have c21610 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21611 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21612 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21613 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21614 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21615 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21616 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_328_5 := arithEq_of_rows h (row := 328) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_5
  simp only [← c21600, k3, ← c21601, k3, ← c21602] at e_328_5
  have e_328_6 := arithEq_of_rows h (row := 328) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_6
  simp only [← c21603, ← c21604, k3, ← c21605] at e_328_6
  have e_328_7 := arithEq_of_rows h (row := 328) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_7
  simp only [← c21612, ← c21613, k3, ← c21614] at e_328_7
  have e_329_2 := arithEq_of_rows h (row := 329) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_2
  simp only [← c21606, ← c21607, ← c21608] at e_329_2
  have e_329_3 := arithEq_of_rows h (row := 329) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_3
  simp only [← c21609, ← c21610, ← c21611] at e_329_3
  refine ⟨?_, ?_⟩
  · have hc := e_329_2
    simp only [e_328_6] at hc
    linear_combination c21615.trans k2 - hc
  · have hc := e_328_7
    simp only [e_328_5, e_329_3, e_328_6] at hc
    linear_combination c21616.trans k2 - hc

theorem publicBatchWrapper86_f2889 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 329 19) = band (a (.virt 18895)) (a (.virt 18897)) := by
  have c21617 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21618 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21619 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_329_4 := arithEq_of_rows h (row := 329) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_4
  simp only [← c21617, ← c21618, ← c21619] at e_329_4
  have hr := e_329_4
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2890 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 329 23) = band (a (.virt 18899)) (a (.virt 18901)) := by
  have c21620 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21621 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21622 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_329_5 := arithEq_of_rows h (row := 329) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_5
  simp only [← c21620, ← c21621, ← c21622] at e_329_5
  have hr := e_329_5
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2891 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 329 27) = band (a (.wire 329 19)) (a (.wire 329 23)) := by
  have c21623 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21624 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21625 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_329_6 := arithEq_of_rows h (row := 329) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_6
  simp only [← c21623, ← c21624, ← c21625] at e_329_6
  have hr := e_329_6
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2892 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 19) = bor (a (.wire 80 79)) (a (.wire 329 27)) := by
  have c21626 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21627 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21628 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21629 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21630 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21631 := (publicBatchWrapper86_copies675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_319_16 := arithEq_of_rows h (row := 319) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_16
  simp only [← c21626, ← c21627, ← c21628] at e_319_16
  have e_326_4 := arithEq_of_rows h (row := 326) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_4
  simp only [← c21629, ← c21630, k3, ← c21631] at e_326_4
  have hr := e_326_4
  simp only [e_319_16] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2893 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 19) = a (.virt 201) := by
  have c21632 := (publicBatchWrapper86_copies676 a h).1
  exact c21632

theorem publicBatchWrapper86_f2894 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16803)) (a (.wire 165 55)) (a (.virt 18903)) (a (.virt 18904)) := by
  have c21633 := (publicBatchWrapper86_copies676 a h).2.1
  have c21634 := (publicBatchWrapper86_copies676 a h).2.2.1
  have c21635 := (publicBatchWrapper86_copies676 a h).2.2.2.1
  have c21636 := (publicBatchWrapper86_copies676 a h).2.2.2.2.1
  have c21637 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.1
  have c21638 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.1
  have c21639 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.1
  have c21640 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.1
  have c21641 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.1
  have c21642 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21643 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21644 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21645 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21646 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21647 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21648 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21649 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_328_8 := arithEq_of_rows h (row := 328) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_8
  simp only [← c21633, k3, ← c21634, k3, ← c21635] at e_328_8
  have e_328_9 := arithEq_of_rows h (row := 328) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_9
  simp only [← c21636, ← c21637, k3, ← c21638] at e_328_9
  have e_328_10 := arithEq_of_rows h (row := 328) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_10
  simp only [← c21645, ← c21646, k3, ← c21647] at e_328_10
  have e_329_7 := arithEq_of_rows h (row := 329) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_7
  simp only [← c21639, ← c21640, ← c21641] at e_329_7
  have e_329_8 := arithEq_of_rows h (row := 329) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_8
  simp only [← c21642, ← c21643, ← c21644] at e_329_8
  refine ⟨?_, ?_⟩
  · have hc := e_329_7
    simp only [e_328_9] at hc
    linear_combination c21648.trans k2 - hc
  · have hc := e_328_10
    simp only [e_328_8, e_329_8, e_328_9] at hc
    linear_combination c21649.trans k2 - hc

theorem publicBatchWrapper86_f2895 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 23) = bor (a (.wire 82 43)) (a (.virt 18903)) := by
  have c21650 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21651 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21652 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21653 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21654 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21655 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_319_17 := arithEq_of_rows h (row := 319) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_17
  simp only [← c21650, ← c21651, ← c21652] at e_319_17
  have e_326_5 := arithEq_of_rows h (row := 326) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_5
  simp only [← c21653, ← c21654, k3, ← c21655] at e_326_5
  have hr := e_326_5
  simp only [e_319_17] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2896 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 23) = a (.virt 201) := by
  have c21656 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21656

theorem publicBatchWrapper86_f2897 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16804)) (a (.wire 165 63)) (a (.virt 18905)) (a (.virt 18906)) := by
  have c21657 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21658 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21659 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21660 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21661 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21662 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21663 := (publicBatchWrapper86_copies676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21664 := (publicBatchWrapper86_copies677 a h).1
  have c21665 := (publicBatchWrapper86_copies677 a h).2.1
  have c21666 := (publicBatchWrapper86_copies677 a h).2.2.1
  have c21667 := (publicBatchWrapper86_copies677 a h).2.2.2.1
  have c21668 := (publicBatchWrapper86_copies677 a h).2.2.2.2.1
  have c21669 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.1
  have c21670 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.1
  have c21671 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.1
  have c21672 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.1
  have c21673 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_328_11 := arithEq_of_rows h (row := 328) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_11
  simp only [← c21657, k3, ← c21658, k3, ← c21659] at e_328_11
  have e_328_12 := arithEq_of_rows h (row := 328) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_12
  simp only [← c21660, ← c21661, k3, ← c21662] at e_328_12
  have e_328_13 := arithEq_of_rows h (row := 328) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_13
  simp only [← c21669, ← c21670, k3, ← c21671] at e_328_13
  have e_329_9 := arithEq_of_rows h (row := 329) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_9
  simp only [← c21663, ← c21664, ← c21665] at e_329_9
  have e_329_10 := arithEq_of_rows h (row := 329) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_10
  simp only [← c21666, ← c21667, ← c21668] at e_329_10
  refine ⟨?_, ?_⟩
  · have hc := e_329_9
    simp only [e_328_12] at hc
    linear_combination c21672.trans k2 - hc
  · have hc := e_328_13
    simp only [e_328_11, e_329_10, e_328_12] at hc
    linear_combination c21673.trans k2 - hc

theorem publicBatchWrapper86_f2898 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 27) = bor (a (.wire 82 43)) (a (.virt 18905)) := by
  have c21674 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21675 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21676 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21677 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21678 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21679 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_319_18 := arithEq_of_rows h (row := 319) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_18
  simp only [← c21674, ← c21675, ← c21676] at e_319_18
  have e_326_6 := arithEq_of_rows h (row := 326) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_6
  simp only [← c21677, ← c21678, k3, ← c21679] at e_326_6
  have hr := e_326_6
  simp only [e_319_18] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2899 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 27) = a (.virt 201) := by
  have c21680 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21680

theorem publicBatchWrapper86_f2900 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16805)) (a (.wire 165 15)) (a (.virt 18907)) (a (.virt 18908)) := by
  have c21681 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21682 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21683 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21684 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21685 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21686 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21687 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21688 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21689 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21690 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21691 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21692 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21693 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21694 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21695 := (publicBatchWrapper86_copies677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21696 := (publicBatchWrapper86_copies678 a h).1
  have c21697 := (publicBatchWrapper86_copies678 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_328_14 := arithEq_of_rows h (row := 328) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_14
  simp only [← c21681, k3, ← c21682, k3, ← c21683] at e_328_14
  have e_328_15 := arithEq_of_rows h (row := 328) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_15
  simp only [← c21684, ← c21685, k3, ← c21686] at e_328_15
  have e_328_16 := arithEq_of_rows h (row := 328) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_16
  simp only [← c21693, ← c21694, k3, ← c21695] at e_328_16
  have e_329_11 := arithEq_of_rows h (row := 329) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_11
  simp only [← c21687, ← c21688, ← c21689] at e_329_11
  have e_329_12 := arithEq_of_rows h (row := 329) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_12
  simp only [← c21690, ← c21691, ← c21692] at e_329_12
  refine ⟨?_, ?_⟩
  · have hc := e_329_11
    simp only [e_328_15] at hc
    linear_combination c21696.trans k2 - hc
  · have hc := e_328_16
    simp only [e_328_14, e_329_12, e_328_15] at hc
    linear_combination c21697.trans k2 - hc

theorem publicBatchWrapper86_f2901 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16806)) (a (.wire 165 23)) (a (.virt 18909)) (a (.virt 18910)) := by
  have c21698 := (publicBatchWrapper86_copies678 a h).2.2.1
  have c21699 := (publicBatchWrapper86_copies678 a h).2.2.2.1
  have c21700 := (publicBatchWrapper86_copies678 a h).2.2.2.2.1
  have c21701 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.1
  have c21702 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.1
  have c21703 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.1
  have c21704 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.1
  have c21705 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.1
  have c21706 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21707 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21708 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21709 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21710 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21711 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21712 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21713 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21714 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_328_17 := arithEq_of_rows h (row := 328) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_17
  simp only [← c21698, k3, ← c21699, k3, ← c21700] at e_328_17
  have e_328_18 := arithEq_of_rows h (row := 328) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_18
  simp only [← c21701, ← c21702, k3, ← c21703] at e_328_18
  have e_328_19 := arithEq_of_rows h (row := 328) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_328_19
  simp only [← c21710, ← c21711, k3, ← c21712] at e_328_19
  have e_329_13 := arithEq_of_rows h (row := 329) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_13
  simp only [← c21704, ← c21705, ← c21706] at e_329_13
  have e_329_14 := arithEq_of_rows h (row := 329) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_14
  simp only [← c21707, ← c21708, ← c21709] at e_329_14
  refine ⟨?_, ?_⟩
  · have hc := e_329_13
    simp only [e_328_18] at hc
    linear_combination c21713.trans k2 - hc
  · have hc := e_328_19
    simp only [e_328_17, e_329_14, e_328_18] at hc
    linear_combination c21714.trans k2 - hc

theorem publicBatchWrapper86_f2902 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16807)) (a (.wire 165 31)) (a (.virt 18911)) (a (.virt 18912)) := by
  have c21715 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21716 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21717 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21718 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21719 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21720 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21721 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21722 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21723 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21724 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21725 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21726 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21727 := (publicBatchWrapper86_copies678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21728 := (publicBatchWrapper86_copies679 a h).1
  have c21729 := (publicBatchWrapper86_copies679 a h).2.1
  have c21730 := (publicBatchWrapper86_copies679 a h).2.2.1
  have c21731 := (publicBatchWrapper86_copies679 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_329_15 := arithEq_of_rows h (row := 329) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_15
  simp only [← c21721, ← c21722, ← c21723] at e_329_15
  have e_329_16 := arithEq_of_rows h (row := 329) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_16
  simp only [← c21724, ← c21725, ← c21726] at e_329_16
  have e_330_0 := arithEq_of_rows h (row := 330) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_0
  simp only [← c21715, k3, ← c21716, k3, ← c21717] at e_330_0
  have e_330_1 := arithEq_of_rows h (row := 330) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_1
  simp only [← c21718, ← c21719, k3, ← c21720] at e_330_1
  have e_330_2 := arithEq_of_rows h (row := 330) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_2
  simp only [← c21727, ← c21728, k3, ← c21729] at e_330_2
  refine ⟨?_, ?_⟩
  · have hc := e_329_15
    simp only [e_330_1] at hc
    linear_combination c21730.trans k2 - hc
  · have hc := e_330_2
    simp only [e_330_0, e_329_16, e_330_1] at hc
    linear_combination c21731.trans k2 - hc

theorem publicBatchWrapper86_f2903 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 16808)) (a (.wire 165 39)) (a (.virt 18913)) (a (.virt 18914)) := by
  have c21732 := (publicBatchWrapper86_copies679 a h).2.2.2.2.1
  have c21733 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.1
  have c21734 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.1
  have c21735 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.1
  have c21736 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.1
  have c21737 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.1
  have c21738 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21739 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21740 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21741 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21742 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21743 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21744 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21745 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21746 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21747 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21748 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_329_17 := arithEq_of_rows h (row := 329) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_17
  simp only [← c21738, ← c21739, ← c21740] at e_329_17
  have e_329_18 := arithEq_of_rows h (row := 329) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_18
  simp only [← c21741, ← c21742, ← c21743] at e_329_18
  have e_330_3 := arithEq_of_rows h (row := 330) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_3
  simp only [← c21732, k3, ← c21733, k3, ← c21734] at e_330_3
  have e_330_4 := arithEq_of_rows h (row := 330) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_4
  simp only [← c21735, ← c21736, k3, ← c21737] at e_330_4
  have e_330_5 := arithEq_of_rows h (row := 330) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_5
  simp only [← c21744, ← c21745, k3, ← c21746] at e_330_5
  refine ⟨?_, ?_⟩
  · have hc := e_329_17
    simp only [e_330_4] at hc
    linear_combination c21747.trans k2 - hc
  · have hc := e_330_5
    simp only [e_330_3, e_329_18, e_330_4] at hc
    linear_combination c21748.trans k2 - hc

theorem publicBatchWrapper86_f2904 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 329 79) = band (a (.virt 18907)) (a (.virt 18909)) := by
  have c21749 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21750 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21751 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_329_19 := arithEq_of_rows h (row := 329) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_329_19
  simp only [← c21749, ← c21750, ← c21751] at e_329_19
  have hr := e_329_19
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2905 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 331 3) = band (a (.virt 18911)) (a (.virt 18913)) := by
  have c21752 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21753 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21754 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_331_0 := arithEq_of_rows h (row := 331) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_0
  simp only [← c21752, ← c21753, ← c21754] at e_331_0
  have hr := e_331_0
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2906 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 331 7) = band (a (.wire 329 79)) (a (.wire 331 3)) := by
  have c21755 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21756 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21757 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_331_1 := arithEq_of_rows h (row := 331) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_1
  simp only [← c21755, ← c21756, ← c21757] at e_331_1
  have hr := e_331_1
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2907 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 31) = bor (a (.wire 82 43)) (a (.wire 331 7)) := by
  have c21758 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21759 := (publicBatchWrapper86_copies679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21760 := (publicBatchWrapper86_copies680 a h).1
  have c21761 := (publicBatchWrapper86_copies680 a h).2.1
  have c21762 := (publicBatchWrapper86_copies680 a h).2.2.1
  have c21763 := (publicBatchWrapper86_copies680 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_319_19 := arithEq_of_rows h (row := 319) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_319_19
  simp only [← c21758, ← c21759, ← c21760] at e_319_19
  have e_326_7 := arithEq_of_rows h (row := 326) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_7
  simp only [← c21761, ← c21762, k3, ← c21763] at e_326_7
  have hr := e_326_7
  simp only [e_319_19] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2908 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 31) = a (.virt 201) := by
  have c21764 := (publicBatchWrapper86_copies680 a h).2.2.2.2.1
  exact c21764

theorem publicBatchWrapper86_f2909 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 17003)) (a (.wire 165 55)) (a (.virt 18915)) (a (.virt 18916)) := by
  have c21765 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.1
  have c21766 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.1
  have c21767 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.1
  have c21768 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.1
  have c21769 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.1
  have c21770 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21771 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21772 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21773 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21774 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21775 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21776 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21777 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21778 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21779 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21780 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21781 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_330_6 := arithEq_of_rows h (row := 330) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_6
  simp only [← c21765, k3, ← c21766, k3, ← c21767] at e_330_6
  have e_330_7 := arithEq_of_rows h (row := 330) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_7
  simp only [← c21768, ← c21769, k3, ← c21770] at e_330_7
  have e_330_8 := arithEq_of_rows h (row := 330) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_8
  simp only [← c21777, ← c21778, k3, ← c21779] at e_330_8
  have e_331_2 := arithEq_of_rows h (row := 331) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_2
  simp only [← c21771, ← c21772, ← c21773] at e_331_2
  have e_331_3 := arithEq_of_rows h (row := 331) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_3
  simp only [← c21774, ← c21775, ← c21776] at e_331_3
  refine ⟨?_, ?_⟩
  · have hc := e_331_2
    simp only [e_330_7] at hc
    linear_combination c21780.trans k2 - hc
  · have hc := e_330_8
    simp only [e_330_6, e_331_3, e_330_7] at hc
    linear_combination c21781.trans k2 - hc

theorem publicBatchWrapper86_f2910 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 35) = bor (a (.wire 84 7)) (a (.virt 18915)) := by
  have c21782 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21783 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21784 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21785 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21786 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21787 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_326_8 := arithEq_of_rows h (row := 326) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_8
  simp only [← c21785, ← c21786, k3, ← c21787] at e_326_8
  have e_332_0 := arithEq_of_rows h (row := 332) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_0
  simp only [← c21782, ← c21783, ← c21784] at e_332_0
  have hr := e_326_8
  simp only [e_332_0] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2911 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 35) = a (.virt 201) := by
  have c21788 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21788

theorem publicBatchWrapper86_f2912 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 17004)) (a (.wire 165 63)) (a (.virt 18917)) (a (.virt 18918)) := by
  have c21789 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21790 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21791 := (publicBatchWrapper86_copies680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21792 := (publicBatchWrapper86_copies681 a h).1
  have c21793 := (publicBatchWrapper86_copies681 a h).2.1
  have c21794 := (publicBatchWrapper86_copies681 a h).2.2.1
  have c21795 := (publicBatchWrapper86_copies681 a h).2.2.2.1
  have c21796 := (publicBatchWrapper86_copies681 a h).2.2.2.2.1
  have c21797 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.1
  have c21798 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.1
  have c21799 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.1
  have c21800 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.1
  have c21801 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.1
  have c21802 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21803 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21804 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21805 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_330_9 := arithEq_of_rows h (row := 330) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_9
  simp only [← c21789, k3, ← c21790, k3, ← c21791] at e_330_9
  have e_330_10 := arithEq_of_rows h (row := 330) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_10
  simp only [← c21792, ← c21793, k3, ← c21794] at e_330_10
  have e_330_11 := arithEq_of_rows h (row := 330) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_11
  simp only [← c21801, ← c21802, k3, ← c21803] at e_330_11
  have e_331_4 := arithEq_of_rows h (row := 331) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_4
  simp only [← c21795, ← c21796, ← c21797] at e_331_4
  have e_331_5 := arithEq_of_rows h (row := 331) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_5
  simp only [← c21798, ← c21799, ← c21800] at e_331_5
  refine ⟨?_, ?_⟩
  · have hc := e_331_4
    simp only [e_330_10] at hc
    linear_combination c21804.trans k2 - hc
  · have hc := e_330_11
    simp only [e_330_9, e_331_5, e_330_10] at hc
    linear_combination c21805.trans k2 - hc

theorem publicBatchWrapper86_f2913 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 39) = bor (a (.wire 84 7)) (a (.virt 18917)) := by
  have c21806 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21807 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21808 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21809 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21810 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21811 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_326_9 := arithEq_of_rows h (row := 326) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_9
  simp only [← c21809, ← c21810, k3, ← c21811] at e_326_9
  have e_332_1 := arithEq_of_rows h (row := 332) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_1
  simp only [← c21806, ← c21807, ← c21808] at e_332_1
  have hr := e_326_9
  simp only [e_332_1] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2914 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 39) = a (.virt 201) := by
  have c21812 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  exact c21812

theorem publicBatchWrapper86_f2915 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 17005)) (a (.wire 165 15)) (a (.virt 18919)) (a (.virt 18920)) := by
  have c21813 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21814 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21815 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21816 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21817 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21818 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21819 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21820 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21821 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21822 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21823 := (publicBatchWrapper86_copies681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21824 := (publicBatchWrapper86_copies682 a h).1
  have c21825 := (publicBatchWrapper86_copies682 a h).2.1
  have c21826 := (publicBatchWrapper86_copies682 a h).2.2.1
  have c21827 := (publicBatchWrapper86_copies682 a h).2.2.2.1
  have c21828 := (publicBatchWrapper86_copies682 a h).2.2.2.2.1
  have c21829 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_330_12 := arithEq_of_rows h (row := 330) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_12
  simp only [← c21813, k3, ← c21814, k3, ← c21815] at e_330_12
  have e_330_13 := arithEq_of_rows h (row := 330) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_13
  simp only [← c21816, ← c21817, k3, ← c21818] at e_330_13
  have e_330_14 := arithEq_of_rows h (row := 330) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_14
  simp only [← c21825, ← c21826, k3, ← c21827] at e_330_14
  have e_331_6 := arithEq_of_rows h (row := 331) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_6
  simp only [← c21819, ← c21820, ← c21821] at e_331_6
  have e_331_7 := arithEq_of_rows h (row := 331) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_7
  simp only [← c21822, ← c21823, ← c21824] at e_331_7
  refine ⟨?_, ?_⟩
  · have hc := e_331_6
    simp only [e_330_13] at hc
    linear_combination c21828.trans k2 - hc
  · have hc := e_330_14
    simp only [e_330_12, e_331_7, e_330_13] at hc
    linear_combination c21829.trans k2 - hc

theorem publicBatchWrapper86_f2916 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 17006)) (a (.wire 165 23)) (a (.virt 18921)) (a (.virt 18922)) := by
  have c21830 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.1
  have c21831 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.1
  have c21832 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.1
  have c21833 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.1
  have c21834 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21835 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21836 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21837 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21838 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21839 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21840 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21841 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21842 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21843 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21844 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21845 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21846 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_330_15 := arithEq_of_rows h (row := 330) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_15
  simp only [← c21830, k3, ← c21831, k3, ← c21832] at e_330_15
  have e_330_16 := arithEq_of_rows h (row := 330) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_16
  simp only [← c21833, ← c21834, k3, ← c21835] at e_330_16
  have e_330_17 := arithEq_of_rows h (row := 330) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_17
  simp only [← c21842, ← c21843, k3, ← c21844] at e_330_17
  have e_331_8 := arithEq_of_rows h (row := 331) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_8
  simp only [← c21836, ← c21837, ← c21838] at e_331_8
  have e_331_9 := arithEq_of_rows h (row := 331) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_9
  simp only [← c21839, ← c21840, ← c21841] at e_331_9
  refine ⟨?_, ?_⟩
  · have hc := e_331_8
    simp only [e_330_16] at hc
    linear_combination c21845.trans k2 - hc
  · have hc := e_330_17
    simp only [e_330_15, e_331_9, e_330_16] at hc
    linear_combination c21846.trans k2 - hc

theorem publicBatchWrapper86_f2917 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 17007)) (a (.wire 165 31)) (a (.virt 18923)) (a (.virt 18924)) := by
  have c21847 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21848 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21849 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21850 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21851 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21852 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21853 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21854 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21855 := (publicBatchWrapper86_copies682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21856 := (publicBatchWrapper86_copies683 a h).1
  have c21857 := (publicBatchWrapper86_copies683 a h).2.1
  have c21858 := (publicBatchWrapper86_copies683 a h).2.2.1
  have c21859 := (publicBatchWrapper86_copies683 a h).2.2.2.1
  have c21860 := (publicBatchWrapper86_copies683 a h).2.2.2.2.1
  have c21861 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.1
  have c21862 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.1
  have c21863 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_330_18 := arithEq_of_rows h (row := 330) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_18
  simp only [← c21847, k3, ← c21848, k3, ← c21849] at e_330_18
  have e_330_19 := arithEq_of_rows h (row := 330) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_330_19
  simp only [← c21850, ← c21851, k3, ← c21852] at e_330_19
  have e_331_10 := arithEq_of_rows h (row := 331) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_10
  simp only [← c21853, ← c21854, ← c21855] at e_331_10
  have e_331_11 := arithEq_of_rows h (row := 331) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_11
  simp only [← c21856, ← c21857, ← c21858] at e_331_11
  have e_333_0 := arithEq_of_rows h (row := 333) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_0
  simp only [← c21859, ← c21860, k3, ← c21861] at e_333_0
  refine ⟨?_, ?_⟩
  · have hc := e_331_10
    simp only [e_330_19] at hc
    linear_combination c21862.trans k2 - hc
  · have hc := e_333_0
    simp only [e_330_18, e_331_11, e_330_19] at hc
    linear_combination c21863.trans k2 - hc

theorem publicBatchWrapper86_f2918 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    IsEqual (a (.virt 17008)) (a (.wire 165 39)) (a (.virt 18925)) (a (.virt 18926)) := by
  have c21864 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.1
  have c21865 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.1
  have c21866 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21867 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21868 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21869 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21870 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21871 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21872 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21873 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21874 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21875 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21876 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21877 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21878 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21879 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21880 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_331_12 := arithEq_of_rows h (row := 331) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_12
  simp only [← c21870, ← c21871, ← c21872] at e_331_12
  have e_331_13 := arithEq_of_rows h (row := 331) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_13
  simp only [← c21873, ← c21874, ← c21875] at e_331_13
  have e_333_1 := arithEq_of_rows h (row := 333) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_1
  simp only [← c21864, k3, ← c21865, k3, ← c21866] at e_333_1
  have e_333_2 := arithEq_of_rows h (row := 333) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_2
  simp only [← c21867, ← c21868, k3, ← c21869] at e_333_2
  have e_333_3 := arithEq_of_rows h (row := 333) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_3
  simp only [← c21876, ← c21877, k3, ← c21878] at e_333_3
  refine ⟨?_, ?_⟩
  · have hc := e_331_12
    simp only [e_333_2] at hc
    linear_combination c21879.trans k2 - hc
  · have hc := e_333_3
    simp only [e_333_1, e_331_13, e_333_2] at hc
    linear_combination c21880.trans k2 - hc

theorem publicBatchWrapper86_f2919 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 331 59) = band (a (.virt 18919)) (a (.virt 18921)) := by
  have c21881 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21882 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21883 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_331_14 := arithEq_of_rows h (row := 331) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_14
  simp only [← c21881, ← c21882, ← c21883] at e_331_14
  have hr := e_331_14
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2920 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 331 63) = band (a (.virt 18923)) (a (.virt 18925)) := by
  have c21884 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21885 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21886 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have e_331_15 := arithEq_of_rows h (row := 331) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_15
  simp only [← c21884, ← c21885, ← c21886] at e_331_15
  have hr := e_331_15
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2921 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 331 67) = band (a (.wire 331 59)) (a (.wire 331 63)) := by
  have c21887 := (publicBatchWrapper86_copies683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21888 := (publicBatchWrapper86_copies684 a h).1
  have c21889 := (publicBatchWrapper86_copies684 a h).2.1
  have e_331_16 := arithEq_of_rows h (row := 331) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_331_16
  simp only [← c21887, ← c21888, ← c21889] at e_331_16
  have hr := e_331_16
  simp only [band]
  linear_combination hr

theorem publicBatchWrapper86_f2922 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 43) = bor (a (.wire 84 7)) (a (.wire 331 67)) := by
  have c21890 := (publicBatchWrapper86_copies684 a h).2.2.1
  have c21891 := (publicBatchWrapper86_copies684 a h).2.2.2.1
  have c21892 := (publicBatchWrapper86_copies684 a h).2.2.2.2.1
  have c21893 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.1
  have c21894 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.1
  have c21895 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_326_10 := arithEq_of_rows h (row := 326) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_326_10
  simp only [← c21893, ← c21894, k3, ← c21895] at e_326_10
  have e_332_2 := arithEq_of_rows h (row := 332) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_332_2
  simp only [← c21890, ← c21891, ← c21892] at e_332_2
  have hr := e_326_10
  simp only [e_332_2] at hr
  simp only [bor]
  linear_combination hr

theorem publicBatchWrapper86_f2923 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 326 43) = a (.virt 201) := by
  have c21896 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.1
  exact c21896

theorem publicBatchWrapper86_f2924 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 333 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 8)) := by
  have c21897 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.1
  have c21898 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21899 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21900 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21901 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21902 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_333_4 := arithEq_of_rows h (row := 333) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_4
  simp only [← c21897, ← c21898, ← c21899] at e_333_4
  have e_333_5 := arithEq_of_rows h (row := 333) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_5
  simp only [← c21900, ← c21901, k2, ← c21902] at e_333_5
  have hr := e_333_5
  simp only [e_333_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2925 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 333 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 9)) := by
  have c21903 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21904 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21905 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21906 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21907 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21908 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_333_6 := arithEq_of_rows h (row := 333) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_6
  simp only [← c21903, ← c21904, ← c21905] at e_333_6
  have e_333_7 := arithEq_of_rows h (row := 333) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_7
  simp only [← c21906, ← c21907, k2, ← c21908] at e_333_7
  have hr := e_333_7
  simp only [e_333_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2926 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 333 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 10)) := by
  have c21909 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21910 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21911 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21912 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21913 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21914 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_333_8 := arithEq_of_rows h (row := 333) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_8
  simp only [← c21909, ← c21910, ← c21911] at e_333_8
  have e_333_9 := arithEq_of_rows h (row := 333) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_9
  simp only [← c21912, ← c21913, k2, ← c21914] at e_333_9
  have hr := e_333_9
  simp only [e_333_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2927 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 333 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 11)) := by
  have c21915 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21916 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21917 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21918 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21919 := (publicBatchWrapper86_copies684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21920 := (publicBatchWrapper86_copies685 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_333_10 := arithEq_of_rows h (row := 333) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_10
  simp only [← c21915, ← c21916, ← c21917] at e_333_10
  have e_333_11 := arithEq_of_rows h (row := 333) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_11
  simp only [← c21918, ← c21919, k2, ← c21920] at e_333_11
  have hr := e_333_11
  simp only [e_333_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2928 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 333 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 12)) := by
  have c21921 := (publicBatchWrapper86_copies685 a h).2.1
  have c21922 := (publicBatchWrapper86_copies685 a h).2.2.1
  have c21923 := (publicBatchWrapper86_copies685 a h).2.2.2.1
  have c21924 := (publicBatchWrapper86_copies685 a h).2.2.2.2.1
  have c21925 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.1
  have c21926 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_333_12 := arithEq_of_rows h (row := 333) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_12
  simp only [← c21921, ← c21922, ← c21923] at e_333_12
  have e_333_13 := arithEq_of_rows h (row := 333) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_13
  simp only [← c21924, ← c21925, k2, ← c21926] at e_333_13
  have hr := e_333_13
  simp only [e_333_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2929 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 333 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 13)) := by
  have c21927 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.1
  have c21928 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.1
  have c21929 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.1
  have c21930 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21931 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21932 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_333_14 := arithEq_of_rows h (row := 333) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_14
  simp only [← c21927, ← c21928, ← c21929] at e_333_14
  have e_333_15 := arithEq_of_rows h (row := 333) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_15
  simp only [← c21930, ← c21931, k2, ← c21932] at e_333_15
  have hr := e_333_15
  simp only [e_333_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2930 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 333 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 14)) := by
  have c21933 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21934 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21935 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21936 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21937 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21938 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_333_16 := arithEq_of_rows h (row := 333) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_16
  simp only [← c21933, ← c21934, ← c21935] at e_333_16
  have e_333_17 := arithEq_of_rows h (row := 333) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_17
  simp only [← c21936, ← c21937, k2, ← c21938] at e_333_17
  have hr := e_333_17
  simp only [e_333_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2931 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 333 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 15)) := by
  have c21939 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21940 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21941 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21942 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21943 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21944 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_333_18 := arithEq_of_rows h (row := 333) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_18
  simp only [← c21939, ← c21940, ← c21941] at e_333_18
  have e_333_19 := arithEq_of_rows h (row := 333) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_333_19
  simp only [← c21942, ← c21943, k2, ← c21944] at e_333_19
  have hr := e_333_19
  simp only [e_333_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2932 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 334 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 16)) := by
  have c21945 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21946 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21947 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21948 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21949 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21950 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_334_0 := arithEq_of_rows h (row := 334) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_0
  simp only [← c21945, ← c21946, ← c21947] at e_334_0
  have e_334_1 := arithEq_of_rows h (row := 334) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_1
  simp only [← c21948, ← c21949, k2, ← c21950] at e_334_1
  have hr := e_334_1
  simp only [e_334_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2933 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 334 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 17)) := by
  have c21951 := (publicBatchWrapper86_copies685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21952 := (publicBatchWrapper86_copies686 a h).1
  have c21953 := (publicBatchWrapper86_copies686 a h).2.1
  have c21954 := (publicBatchWrapper86_copies686 a h).2.2.1
  have c21955 := (publicBatchWrapper86_copies686 a h).2.2.2.1
  have c21956 := (publicBatchWrapper86_copies686 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_334_2 := arithEq_of_rows h (row := 334) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_2
  simp only [← c21951, ← c21952, ← c21953] at e_334_2
  have e_334_3 := arithEq_of_rows h (row := 334) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_3
  simp only [← c21954, ← c21955, k2, ← c21956] at e_334_3
  have hr := e_334_3
  simp only [e_334_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2934 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 334 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 18)) := by
  have c21957 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.1
  have c21958 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.1
  have c21959 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.1
  have c21960 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.1
  have c21961 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.1
  have c21962 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_334_4 := arithEq_of_rows h (row := 334) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_4
  simp only [← c21957, ← c21958, ← c21959] at e_334_4
  have e_334_5 := arithEq_of_rows h (row := 334) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_5
  simp only [← c21960, ← c21961, k2, ← c21962] at e_334_5
  have hr := e_334_5
  simp only [e_334_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2935 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 334 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 19)) := by
  have c21963 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21964 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21965 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21966 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21967 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21968 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_334_6 := arithEq_of_rows h (row := 334) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_6
  simp only [← c21963, ← c21964, ← c21965] at e_334_6
  have e_334_7 := arithEq_of_rows h (row := 334) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_7
  simp only [← c21966, ← c21967, k2, ← c21968] at e_334_7
  have hr := e_334_7
  simp only [e_334_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2936 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 334 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 20)) := by
  have c21969 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21970 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21971 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21972 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21973 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21974 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_334_8 := arithEq_of_rows h (row := 334) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_8
  simp only [← c21969, ← c21970, ← c21971] at e_334_8
  have e_334_9 := arithEq_of_rows h (row := 334) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_9
  simp only [← c21972, ← c21973, k2, ← c21974] at e_334_9
  have hr := e_334_9
  simp only [e_334_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2937 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 334 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 21)) := by
  have c21975 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21976 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21977 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21978 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21979 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21980 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_334_10 := arithEq_of_rows h (row := 334) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_10
  simp only [← c21975, ← c21976, ← c21977] at e_334_10
  have e_334_11 := arithEq_of_rows h (row := 334) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_11
  simp only [← c21978, ← c21979, k2, ← c21980] at e_334_11
  have hr := e_334_11
  simp only [e_334_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2938 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 334 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 22)) := by
  have c21981 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21982 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21983 := (publicBatchWrapper86_copies686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c21984 := (publicBatchWrapper86_copies687 a h).1
  have c21985 := (publicBatchWrapper86_copies687 a h).2.1
  have c21986 := (publicBatchWrapper86_copies687 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_334_12 := arithEq_of_rows h (row := 334) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_12
  simp only [← c21981, ← c21982, ← c21983] at e_334_12
  have e_334_13 := arithEq_of_rows h (row := 334) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_13
  simp only [← c21984, ← c21985, k2, ← c21986] at e_334_13
  have hr := e_334_13
  simp only [e_334_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2939 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 334 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 23)) := by
  have c21987 := (publicBatchWrapper86_copies687 a h).2.2.2.1
  have c21988 := (publicBatchWrapper86_copies687 a h).2.2.2.2.1
  have c21989 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.1
  have c21990 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.1
  have c21991 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.1
  have c21992 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_334_14 := arithEq_of_rows h (row := 334) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_14
  simp only [← c21987, ← c21988, ← c21989] at e_334_14
  have e_334_15 := arithEq_of_rows h (row := 334) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_15
  simp only [← c21990, ← c21991, k2, ← c21992] at e_334_15
  have hr := e_334_15
  simp only [e_334_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2940 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 334 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 24)) := by
  have c21993 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.1
  have c21994 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.1
  have c21995 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c21996 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21997 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c21998 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_334_16 := arithEq_of_rows h (row := 334) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_16
  simp only [← c21993, ← c21994, ← c21995] at e_334_16
  have e_334_17 := arithEq_of_rows h (row := 334) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_17
  simp only [← c21996, ← c21997, k2, ← c21998] at e_334_17
  have hr := e_334_17
  simp only [e_334_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2941 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 334 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 25)) := by
  have c21999 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22000 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22001 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22002 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22003 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22004 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_334_18 := arithEq_of_rows h (row := 334) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_18
  simp only [← c21999, ← c22000, ← c22001] at e_334_18
  have e_334_19 := arithEq_of_rows h (row := 334) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_334_19
  simp only [← c22002, ← c22003, k2, ← c22004] at e_334_19
  have hr := e_334_19
  simp only [e_334_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2942 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 335 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 26)) := by
  have c22005 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22006 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22007 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22008 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22009 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22010 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_335_0 := arithEq_of_rows h (row := 335) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_0
  simp only [← c22005, ← c22006, ← c22007] at e_335_0
  have e_335_1 := arithEq_of_rows h (row := 335) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_1
  simp only [← c22008, ← c22009, k2, ← c22010] at e_335_1
  have hr := e_335_1
  simp only [e_335_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2943 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 335 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 27)) := by
  have c22011 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22012 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22013 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22014 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22015 := (publicBatchWrapper86_copies687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22016 := (publicBatchWrapper86_copies688 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_335_2 := arithEq_of_rows h (row := 335) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_2
  simp only [← c22011, ← c22012, ← c22013] at e_335_2
  have e_335_3 := arithEq_of_rows h (row := 335) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_3
  simp only [← c22014, ← c22015, k2, ← c22016] at e_335_3
  have hr := e_335_3
  simp only [e_335_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2944 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 335 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 28)) := by
  have c22017 := (publicBatchWrapper86_copies688 a h).2.1
  have c22018 := (publicBatchWrapper86_copies688 a h).2.2.1
  have c22019 := (publicBatchWrapper86_copies688 a h).2.2.2.1
  have c22020 := (publicBatchWrapper86_copies688 a h).2.2.2.2.1
  have c22021 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.1
  have c22022 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_335_4 := arithEq_of_rows h (row := 335) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_4
  simp only [← c22017, ← c22018, ← c22019] at e_335_4
  have e_335_5 := arithEq_of_rows h (row := 335) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_5
  simp only [← c22020, ← c22021, k2, ← c22022] at e_335_5
  have hr := e_335_5
  simp only [e_335_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2945 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 335 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 29)) := by
  have c22023 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.1
  have c22024 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.1
  have c22025 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.1
  have c22026 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22027 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22028 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_335_6 := arithEq_of_rows h (row := 335) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_6
  simp only [← c22023, ← c22024, ← c22025] at e_335_6
  have e_335_7 := arithEq_of_rows h (row := 335) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_7
  simp only [← c22026, ← c22027, k2, ← c22028] at e_335_7
  have hr := e_335_7
  simp only [e_335_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2946 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 335 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 30)) := by
  have c22029 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22030 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22031 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22032 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22033 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22034 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_335_8 := arithEq_of_rows h (row := 335) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_8
  simp only [← c22029, ← c22030, ← c22031] at e_335_8
  have e_335_9 := arithEq_of_rows h (row := 335) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_9
  simp only [← c22032, ← c22033, k2, ← c22034] at e_335_9
  have hr := e_335_9
  simp only [e_335_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2947 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 335 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 31)) := by
  have c22035 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22036 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22037 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22038 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22039 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22040 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_335_10 := arithEq_of_rows h (row := 335) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_10
  simp only [← c22035, ← c22036, ← c22037] at e_335_10
  have e_335_11 := arithEq_of_rows h (row := 335) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_11
  simp only [← c22038, ← c22039, k2, ← c22040] at e_335_11
  have hr := e_335_11
  simp only [e_335_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2948 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 335 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 32)) := by
  have c22041 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22042 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22043 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22044 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22045 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22046 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_335_12 := arithEq_of_rows h (row := 335) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_12
  simp only [← c22041, ← c22042, ← c22043] at e_335_12
  have e_335_13 := arithEq_of_rows h (row := 335) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_13
  simp only [← c22044, ← c22045, k2, ← c22046] at e_335_13
  have hr := e_335_13
  simp only [e_335_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2949 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 335 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 33)) := by
  have c22047 := (publicBatchWrapper86_copies688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22048 := (publicBatchWrapper86_copies689 a h).1
  have c22049 := (publicBatchWrapper86_copies689 a h).2.1
  have c22050 := (publicBatchWrapper86_copies689 a h).2.2.1
  have c22051 := (publicBatchWrapper86_copies689 a h).2.2.2.1
  have c22052 := (publicBatchWrapper86_copies689 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_335_14 := arithEq_of_rows h (row := 335) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_14
  simp only [← c22047, ← c22048, ← c22049] at e_335_14
  have e_335_15 := arithEq_of_rows h (row := 335) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_15
  simp only [← c22050, ← c22051, k2, ← c22052] at e_335_15
  have hr := e_335_15
  simp only [e_335_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2950 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 335 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 34)) := by
  have c22053 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.1
  have c22054 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.1
  have c22055 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.1
  have c22056 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.1
  have c22057 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.1
  have c22058 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_335_16 := arithEq_of_rows h (row := 335) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_16
  simp only [← c22053, ← c22054, ← c22055] at e_335_16
  have e_335_17 := arithEq_of_rows h (row := 335) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_17
  simp only [← c22056, ← c22057, k2, ← c22058] at e_335_17
  have hr := e_335_17
  simp only [e_335_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2951 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 335 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 35)) := by
  have c22059 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22060 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22061 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22062 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22063 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22064 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_335_18 := arithEq_of_rows h (row := 335) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_18
  simp only [← c22059, ← c22060, ← c22061] at e_335_18
  have e_335_19 := arithEq_of_rows h (row := 335) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_335_19
  simp only [← c22062, ← c22063, k2, ← c22064] at e_335_19
  have hr := e_335_19
  simp only [e_335_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2952 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 336 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 36)) := by
  have c22065 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22066 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22067 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22068 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22069 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22070 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_336_0 := arithEq_of_rows h (row := 336) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_0
  simp only [← c22065, ← c22066, ← c22067] at e_336_0
  have e_336_1 := arithEq_of_rows h (row := 336) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_1
  simp only [← c22068, ← c22069, k2, ← c22070] at e_336_1
  have hr := e_336_1
  simp only [e_336_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 336 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 37)) := by
  have c22071 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22072 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22073 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22074 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22075 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22076 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_336_2 := arithEq_of_rows h (row := 336) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_2
  simp only [← c22071, ← c22072, ← c22073] at e_336_2
  have e_336_3 := arithEq_of_rows h (row := 336) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_3
  simp only [← c22074, ← c22075, k2, ← c22076] at e_336_3
  have hr := e_336_3
  simp only [e_336_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2954 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 336 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 38)) := by
  have c22077 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22078 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22079 := (publicBatchWrapper86_copies689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22080 := (publicBatchWrapper86_copies690 a h).1
  have c22081 := (publicBatchWrapper86_copies690 a h).2.1
  have c22082 := (publicBatchWrapper86_copies690 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_336_4 := arithEq_of_rows h (row := 336) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_4
  simp only [← c22077, ← c22078, ← c22079] at e_336_4
  have e_336_5 := arithEq_of_rows h (row := 336) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_5
  simp only [← c22080, ← c22081, k2, ← c22082] at e_336_5
  have hr := e_336_5
  simp only [e_336_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2955 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 336 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 39)) := by
  have c22083 := (publicBatchWrapper86_copies690 a h).2.2.2.1
  have c22084 := (publicBatchWrapper86_copies690 a h).2.2.2.2.1
  have c22085 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.1
  have c22086 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.1
  have c22087 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.1
  have c22088 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_336_6 := arithEq_of_rows h (row := 336) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_6
  simp only [← c22083, ← c22084, ← c22085] at e_336_6
  have e_336_7 := arithEq_of_rows h (row := 336) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_7
  simp only [← c22086, ← c22087, k2, ← c22088] at e_336_7
  have hr := e_336_7
  simp only [e_336_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2956 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 336 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 40)) := by
  have c22089 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.1
  have c22090 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22091 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22092 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22093 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22094 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_336_8 := arithEq_of_rows h (row := 336) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_8
  simp only [← c22089, ← c22090, ← c22091] at e_336_8
  have e_336_9 := arithEq_of_rows h (row := 336) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_9
  simp only [← c22092, ← c22093, k2, ← c22094] at e_336_9
  have hr := e_336_9
  simp only [e_336_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2957 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 336 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 41)) := by
  have c22095 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22096 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22097 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22098 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22099 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22100 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_336_10 := arithEq_of_rows h (row := 336) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_10
  simp only [← c22095, ← c22096, ← c22097] at e_336_10
  have e_336_11 := arithEq_of_rows h (row := 336) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_11
  simp only [← c22098, ← c22099, k2, ← c22100] at e_336_11
  have hr := e_336_11
  simp only [e_336_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2958 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 336 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 42)) := by
  have c22101 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22102 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22103 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22104 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22105 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22106 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_336_12 := arithEq_of_rows h (row := 336) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_12
  simp only [← c22101, ← c22102, ← c22103] at e_336_12
  have e_336_13 := arithEq_of_rows h (row := 336) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_13
  simp only [← c22104, ← c22105, k2, ← c22106] at e_336_13
  have hr := e_336_13
  simp only [e_336_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2959 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 336 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 43)) := by
  have c22107 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22108 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22109 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22110 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22111 := (publicBatchWrapper86_copies690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22112 := (publicBatchWrapper86_copies691 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_336_14 := arithEq_of_rows h (row := 336) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_14
  simp only [← c22107, ← c22108, ← c22109] at e_336_14
  have e_336_15 := arithEq_of_rows h (row := 336) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_15
  simp only [← c22110, ← c22111, k2, ← c22112] at e_336_15
  have hr := e_336_15
  simp only [e_336_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2960 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 336 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 44)) := by
  have c22113 := (publicBatchWrapper86_copies691 a h).2.1
  have c22114 := (publicBatchWrapper86_copies691 a h).2.2.1
  have c22115 := (publicBatchWrapper86_copies691 a h).2.2.2.1
  have c22116 := (publicBatchWrapper86_copies691 a h).2.2.2.2.1
  have c22117 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.1
  have c22118 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_336_16 := arithEq_of_rows h (row := 336) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_16
  simp only [← c22113, ← c22114, ← c22115] at e_336_16
  have e_336_17 := arithEq_of_rows h (row := 336) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_17
  simp only [← c22116, ← c22117, k2, ← c22118] at e_336_17
  have hr := e_336_17
  simp only [e_336_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2961 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 336 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 45)) := by
  have c22119 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.1
  have c22120 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.1
  have c22121 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.1
  have c22122 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22123 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22124 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_336_18 := arithEq_of_rows h (row := 336) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_18
  simp only [← c22119, ← c22120, ← c22121] at e_336_18
  have e_336_19 := arithEq_of_rows h (row := 336) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_336_19
  simp only [← c22122, ← c22123, k2, ← c22124] at e_336_19
  have hr := e_336_19
  simp only [e_336_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2962 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 337 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 46)) := by
  have c22125 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22126 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22127 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22128 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22129 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22130 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_337_0 := arithEq_of_rows h (row := 337) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_0
  simp only [← c22125, ← c22126, ← c22127] at e_337_0
  have e_337_1 := arithEq_of_rows h (row := 337) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_1
  simp only [← c22128, ← c22129, k2, ← c22130] at e_337_1
  have hr := e_337_1
  simp only [e_337_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2963 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 337 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 47)) := by
  have c22131 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22132 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22133 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22134 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22135 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22136 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_337_2 := arithEq_of_rows h (row := 337) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_2
  simp only [← c22131, ← c22132, ← c22133] at e_337_2
  have e_337_3 := arithEq_of_rows h (row := 337) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_3
  simp only [← c22134, ← c22135, k2, ← c22136] at e_337_3
  have hr := e_337_3
  simp only [e_337_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2964 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 337 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 48)) := by
  have c22137 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22138 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22139 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22140 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22141 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22142 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_337_4 := arithEq_of_rows h (row := 337) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_4
  simp only [← c22137, ← c22138, ← c22139] at e_337_4
  have e_337_5 := arithEq_of_rows h (row := 337) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_5
  simp only [← c22140, ← c22141, k2, ← c22142] at e_337_5
  have hr := e_337_5
  simp only [e_337_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2965 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 337 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 49)) := by
  have c22143 := (publicBatchWrapper86_copies691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22144 := (publicBatchWrapper86_copies692 a h).1
  have c22145 := (publicBatchWrapper86_copies692 a h).2.1
  have c22146 := (publicBatchWrapper86_copies692 a h).2.2.1
  have c22147 := (publicBatchWrapper86_copies692 a h).2.2.2.1
  have c22148 := (publicBatchWrapper86_copies692 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_337_6 := arithEq_of_rows h (row := 337) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_6
  simp only [← c22143, ← c22144, ← c22145] at e_337_6
  have e_337_7 := arithEq_of_rows h (row := 337) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_7
  simp only [← c22146, ← c22147, k2, ← c22148] at e_337_7
  have hr := e_337_7
  simp only [e_337_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2966 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 337 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 50)) := by
  have c22149 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.1
  have c22150 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.1
  have c22151 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.1
  have c22152 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.1
  have c22153 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.1
  have c22154 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_337_8 := arithEq_of_rows h (row := 337) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_8
  simp only [← c22149, ← c22150, ← c22151] at e_337_8
  have e_337_9 := arithEq_of_rows h (row := 337) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_9
  simp only [← c22152, ← c22153, k2, ← c22154] at e_337_9
  have hr := e_337_9
  simp only [e_337_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2967 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 337 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 51)) := by
  have c22155 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22156 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22157 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22158 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22159 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22160 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_337_10 := arithEq_of_rows h (row := 337) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_10
  simp only [← c22155, ← c22156, ← c22157] at e_337_10
  have e_337_11 := arithEq_of_rows h (row := 337) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_11
  simp only [← c22158, ← c22159, k2, ← c22160] at e_337_11
  have hr := e_337_11
  simp only [e_337_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2968 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 337 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 52)) := by
  have c22161 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22162 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22163 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22164 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22165 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22166 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_337_12 := arithEq_of_rows h (row := 337) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_12
  simp only [← c22161, ← c22162, ← c22163] at e_337_12
  have e_337_13 := arithEq_of_rows h (row := 337) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_13
  simp only [← c22164, ← c22165, k2, ← c22166] at e_337_13
  have hr := e_337_13
  simp only [e_337_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2969 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 337 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 53)) := by
  have c22167 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22168 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22169 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22170 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22171 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22172 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_337_14 := arithEq_of_rows h (row := 337) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_14
  simp only [← c22167, ← c22168, ← c22169] at e_337_14
  have e_337_15 := arithEq_of_rows h (row := 337) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_15
  simp only [← c22170, ← c22171, k2, ← c22172] at e_337_15
  have hr := e_337_15
  simp only [e_337_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2970 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 337 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 54)) := by
  have c22173 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22174 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22175 := (publicBatchWrapper86_copies692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22176 := (publicBatchWrapper86_copies693 a h).1
  have c22177 := (publicBatchWrapper86_copies693 a h).2.1
  have c22178 := (publicBatchWrapper86_copies693 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_337_16 := arithEq_of_rows h (row := 337) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_16
  simp only [← c22173, ← c22174, ← c22175] at e_337_16
  have e_337_17 := arithEq_of_rows h (row := 337) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_17
  simp only [← c22176, ← c22177, k2, ← c22178] at e_337_17
  have hr := e_337_17
  simp only [e_337_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2971 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 337 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 55)) := by
  have c22179 := (publicBatchWrapper86_copies693 a h).2.2.2.1
  have c22180 := (publicBatchWrapper86_copies693 a h).2.2.2.2.1
  have c22181 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.1
  have c22182 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.1
  have c22183 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.1
  have c22184 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_337_18 := arithEq_of_rows h (row := 337) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_18
  simp only [← c22179, ← c22180, ← c22181] at e_337_18
  have e_337_19 := arithEq_of_rows h (row := 337) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_337_19
  simp only [← c22182, ← c22183, k2, ← c22184] at e_337_19
  have hr := e_337_19
  simp only [e_337_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2972 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 338 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 56)) := by
  have c22185 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.1
  have c22186 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22187 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22188 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22189 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22190 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_338_0 := arithEq_of_rows h (row := 338) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_0
  simp only [← c22185, ← c22186, ← c22187] at e_338_0
  have e_338_1 := arithEq_of_rows h (row := 338) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_1
  simp only [← c22188, ← c22189, k2, ← c22190] at e_338_1
  have hr := e_338_1
  simp only [e_338_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2973 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 338 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 57)) := by
  have c22191 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22192 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22193 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22194 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22195 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22196 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_338_2 := arithEq_of_rows h (row := 338) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_2
  simp only [← c22191, ← c22192, ← c22193] at e_338_2
  have e_338_3 := arithEq_of_rows h (row := 338) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_3
  simp only [← c22194, ← c22195, k2, ← c22196] at e_338_3
  have hr := e_338_3
  simp only [e_338_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 338 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 58)) := by
  have c22197 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22198 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22199 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22200 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22201 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22202 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_338_4 := arithEq_of_rows h (row := 338) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_4
  simp only [← c22197, ← c22198, ← c22199] at e_338_4
  have e_338_5 := arithEq_of_rows h (row := 338) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_5
  simp only [← c22200, ← c22201, k2, ← c22202] at e_338_5
  have hr := e_338_5
  simp only [e_338_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 338 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 59)) := by
  have c22203 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22204 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22205 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22206 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22207 := (publicBatchWrapper86_copies693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22208 := (publicBatchWrapper86_copies694 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_338_6 := arithEq_of_rows h (row := 338) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_6
  simp only [← c22203, ← c22204, ← c22205] at e_338_6
  have e_338_7 := arithEq_of_rows h (row := 338) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_7
  simp only [← c22206, ← c22207, k2, ← c22208] at e_338_7
  have hr := e_338_7
  simp only [e_338_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 338 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 60)) := by
  have c22209 := (publicBatchWrapper86_copies694 a h).2.1
  have c22210 := (publicBatchWrapper86_copies694 a h).2.2.1
  have c22211 := (publicBatchWrapper86_copies694 a h).2.2.2.1
  have c22212 := (publicBatchWrapper86_copies694 a h).2.2.2.2.1
  have c22213 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.1
  have c22214 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_338_8 := arithEq_of_rows h (row := 338) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_8
  simp only [← c22209, ← c22210, ← c22211] at e_338_8
  have e_338_9 := arithEq_of_rows h (row := 338) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_9
  simp only [← c22212, ← c22213, k2, ← c22214] at e_338_9
  have hr := e_338_9
  simp only [e_338_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 338 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 61)) := by
  have c22215 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.1
  have c22216 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.1
  have c22217 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.1
  have c22218 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22219 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22220 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_338_10 := arithEq_of_rows h (row := 338) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_10
  simp only [← c22215, ← c22216, ← c22217] at e_338_10
  have e_338_11 := arithEq_of_rows h (row := 338) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_11
  simp only [← c22218, ← c22219, k2, ← c22220] at e_338_11
  have hr := e_338_11
  simp only [e_338_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 338 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 62)) := by
  have c22221 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22222 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22223 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22224 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22225 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22226 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_338_12 := arithEq_of_rows h (row := 338) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_12
  simp only [← c22221, ← c22222, ← c22223] at e_338_12
  have e_338_13 := arithEq_of_rows h (row := 338) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_13
  simp only [← c22224, ← c22225, k2, ← c22226] at e_338_13
  have hr := e_338_13
  simp only [e_338_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 338 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 63)) := by
  have c22227 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22228 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22229 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22230 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22231 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22232 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_338_14 := arithEq_of_rows h (row := 338) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_14
  simp only [← c22227, ← c22228, ← c22229] at e_338_14
  have e_338_15 := arithEq_of_rows h (row := 338) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_15
  simp only [← c22230, ← c22231, k2, ← c22232] at e_338_15
  have hr := e_338_15
  simp only [e_338_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 338 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 64)) := by
  have c22233 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22234 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22235 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22236 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22237 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22238 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_338_16 := arithEq_of_rows h (row := 338) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_16
  simp only [← c22233, ← c22234, ← c22235] at e_338_16
  have e_338_17 := arithEq_of_rows h (row := 338) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_17
  simp only [← c22236, ← c22237, k2, ← c22238] at e_338_17
  have hr := e_338_17
  simp only [e_338_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 338 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 65)) := by
  have c22239 := (publicBatchWrapper86_copies694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22240 := (publicBatchWrapper86_copies695 a h).1
  have c22241 := (publicBatchWrapper86_copies695 a h).2.1
  have c22242 := (publicBatchWrapper86_copies695 a h).2.2.1
  have c22243 := (publicBatchWrapper86_copies695 a h).2.2.2.1
  have c22244 := (publicBatchWrapper86_copies695 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_338_18 := arithEq_of_rows h (row := 338) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_18
  simp only [← c22239, ← c22240, ← c22241] at e_338_18
  have e_338_19 := arithEq_of_rows h (row := 338) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_338_19
  simp only [← c22242, ← c22243, k2, ← c22244] at e_338_19
  have hr := e_338_19
  simp only [e_338_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 339 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 66)) := by
  have c22245 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.1
  have c22246 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.1
  have c22247 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.1
  have c22248 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.1
  have c22249 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.1
  have c22250 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_339_0 := arithEq_of_rows h (row := 339) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_0
  simp only [← c22245, ← c22246, ← c22247] at e_339_0
  have e_339_1 := arithEq_of_rows h (row := 339) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_1
  simp only [← c22248, ← c22249, k2, ← c22250] at e_339_1
  have hr := e_339_1
  simp only [e_339_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 339 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 67)) := by
  have c22251 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22252 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22253 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22254 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22255 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22256 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_339_2 := arithEq_of_rows h (row := 339) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_2
  simp only [← c22251, ← c22252, ← c22253] at e_339_2
  have e_339_3 := arithEq_of_rows h (row := 339) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_3
  simp only [← c22254, ← c22255, k2, ← c22256] at e_339_3
  have hr := e_339_3
  simp only [e_339_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 339 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 68)) := by
  have c22257 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22258 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22259 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22260 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22261 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22262 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_339_4 := arithEq_of_rows h (row := 339) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_4
  simp only [← c22257, ← c22258, ← c22259] at e_339_4
  have e_339_5 := arithEq_of_rows h (row := 339) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_5
  simp only [← c22260, ← c22261, k2, ← c22262] at e_339_5
  have hr := e_339_5
  simp only [e_339_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 339 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 69)) := by
  have c22263 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22264 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22265 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22266 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22267 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22268 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_339_6 := arithEq_of_rows h (row := 339) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_6
  simp only [← c22263, ← c22264, ← c22265] at e_339_6
  have e_339_7 := arithEq_of_rows h (row := 339) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_7
  simp only [← c22266, ← c22267, k2, ← c22268] at e_339_7
  have hr := e_339_7
  simp only [e_339_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 339 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 70)) := by
  have c22269 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22270 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22271 := (publicBatchWrapper86_copies695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22272 := (publicBatchWrapper86_copies696 a h).1
  have c22273 := (publicBatchWrapper86_copies696 a h).2.1
  have c22274 := (publicBatchWrapper86_copies696 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_339_8 := arithEq_of_rows h (row := 339) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_8
  simp only [← c22269, ← c22270, ← c22271] at e_339_8
  have e_339_9 := arithEq_of_rows h (row := 339) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_9
  simp only [← c22272, ← c22273, k2, ← c22274] at e_339_9
  have hr := e_339_9
  simp only [e_339_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 339 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 71)) := by
  have c22275 := (publicBatchWrapper86_copies696 a h).2.2.2.1
  have c22276 := (publicBatchWrapper86_copies696 a h).2.2.2.2.1
  have c22277 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.1
  have c22278 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.1
  have c22279 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.1
  have c22280 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_339_10 := arithEq_of_rows h (row := 339) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_10
  simp only [← c22275, ← c22276, ← c22277] at e_339_10
  have e_339_11 := arithEq_of_rows h (row := 339) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_11
  simp only [← c22278, ← c22279, k2, ← c22280] at e_339_11
  have hr := e_339_11
  simp only [e_339_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 339 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 72)) := by
  have c22281 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.1
  have c22282 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22283 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22284 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22285 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22286 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_339_12 := arithEq_of_rows h (row := 339) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_12
  simp only [← c22281, ← c22282, ← c22283] at e_339_12
  have e_339_13 := arithEq_of_rows h (row := 339) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_13
  simp only [← c22284, ← c22285, k2, ← c22286] at e_339_13
  have hr := e_339_13
  simp only [e_339_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 339 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 73)) := by
  have c22287 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22288 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22289 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22290 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22291 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22292 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_339_14 := arithEq_of_rows h (row := 339) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_14
  simp only [← c22287, ← c22288, ← c22289] at e_339_14
  have e_339_15 := arithEq_of_rows h (row := 339) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_15
  simp only [← c22290, ← c22291, k2, ← c22292] at e_339_15
  have hr := e_339_15
  simp only [e_339_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 339 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 74)) := by
  have c22293 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22294 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22295 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22296 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22297 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22298 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_339_16 := arithEq_of_rows h (row := 339) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_16
  simp only [← c22293, ← c22294, ← c22295] at e_339_16
  have e_339_17 := arithEq_of_rows h (row := 339) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_17
  simp only [← c22296, ← c22297, k2, ← c22298] at e_339_17
  have hr := e_339_17
  simp only [e_339_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 339 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 75)) := by
  have c22299 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22300 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22301 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22302 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22303 := (publicBatchWrapper86_copies696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22304 := (publicBatchWrapper86_copies697 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_339_18 := arithEq_of_rows h (row := 339) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_18
  simp only [← c22299, ← c22300, ← c22301] at e_339_18
  have e_339_19 := arithEq_of_rows h (row := 339) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_339_19
  simp only [← c22302, ← c22303, k2, ← c22304] at e_339_19
  have hr := e_339_19
  simp only [e_339_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 340 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 76)) := by
  have c22305 := (publicBatchWrapper86_copies697 a h).2.1
  have c22306 := (publicBatchWrapper86_copies697 a h).2.2.1
  have c22307 := (publicBatchWrapper86_copies697 a h).2.2.2.1
  have c22308 := (publicBatchWrapper86_copies697 a h).2.2.2.2.1
  have c22309 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.1
  have c22310 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_340_0 := arithEq_of_rows h (row := 340) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_0
  simp only [← c22305, ← c22306, ← c22307] at e_340_0
  have e_340_1 := arithEq_of_rows h (row := 340) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_1
  simp only [← c22308, ← c22309, k2, ← c22310] at e_340_1
  have hr := e_340_1
  simp only [e_340_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 340 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 77)) := by
  have c22311 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.1
  have c22312 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.1
  have c22313 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.1
  have c22314 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22315 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22316 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_340_2 := arithEq_of_rows h (row := 340) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_2
  simp only [← c22311, ← c22312, ← c22313] at e_340_2
  have e_340_3 := arithEq_of_rows h (row := 340) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_3
  simp only [← c22314, ← c22315, k2, ← c22316] at e_340_3
  have hr := e_340_3
  simp only [e_340_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 340 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 78)) := by
  have c22317 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22318 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22319 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22320 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22321 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22322 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_340_4 := arithEq_of_rows h (row := 340) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_4
  simp only [← c22317, ← c22318, ← c22319] at e_340_4
  have e_340_5 := arithEq_of_rows h (row := 340) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_5
  simp only [← c22320, ← c22321, k2, ← c22322] at e_340_5
  have hr := e_340_5
  simp only [e_340_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 340 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 79)) := by
  have c22323 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22324 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22325 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22326 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22327 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22328 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_340_6 := arithEq_of_rows h (row := 340) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_6
  simp only [← c22323, ← c22324, ← c22325] at e_340_6
  have e_340_7 := arithEq_of_rows h (row := 340) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_7
  simp only [← c22326, ← c22327, k2, ← c22328] at e_340_7
  have hr := e_340_7
  simp only [e_340_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 340 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 80)) := by
  have c22329 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22330 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22331 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22332 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22333 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22334 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_340_8 := arithEq_of_rows h (row := 340) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_8
  simp only [← c22329, ← c22330, ← c22331] at e_340_8
  have e_340_9 := arithEq_of_rows h (row := 340) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_9
  simp only [← c22332, ← c22333, k2, ← c22334] at e_340_9
  have hr := e_340_9
  simp only [e_340_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 340 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 81)) := by
  have c22335 := (publicBatchWrapper86_copies697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22336 := (publicBatchWrapper86_copies698 a h).1
  have c22337 := (publicBatchWrapper86_copies698 a h).2.1
  have c22338 := (publicBatchWrapper86_copies698 a h).2.2.1
  have c22339 := (publicBatchWrapper86_copies698 a h).2.2.2.1
  have c22340 := (publicBatchWrapper86_copies698 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_340_10 := arithEq_of_rows h (row := 340) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_10
  simp only [← c22335, ← c22336, ← c22337] at e_340_10
  have e_340_11 := arithEq_of_rows h (row := 340) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_11
  simp only [← c22338, ← c22339, k2, ← c22340] at e_340_11
  have hr := e_340_11
  simp only [e_340_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 340 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 82)) := by
  have c22341 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.1
  have c22342 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.1
  have c22343 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.1
  have c22344 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.1
  have c22345 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.1
  have c22346 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_340_12 := arithEq_of_rows h (row := 340) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_12
  simp only [← c22341, ← c22342, ← c22343] at e_340_12
  have e_340_13 := arithEq_of_rows h (row := 340) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_13
  simp only [← c22344, ← c22345, k2, ← c22346] at e_340_13
  have hr := e_340_13
  simp only [e_340_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f2999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 340 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 83)) := by
  have c22347 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22348 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22349 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22350 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22351 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22352 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_340_14 := arithEq_of_rows h (row := 340) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_14
  simp only [← c22347, ← c22348, ← c22349] at e_340_14
  have e_340_15 := arithEq_of_rows h (row := 340) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_15
  simp only [← c22350, ← c22351, k2, ← c22352] at e_340_15
  have hr := e_340_15
  simp only [e_340_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 340 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 84)) := by
  have c22353 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22354 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22355 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22356 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22357 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22358 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_340_16 := arithEq_of_rows h (row := 340) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_16
  simp only [← c22353, ← c22354, ← c22355] at e_340_16
  have e_340_17 := arithEq_of_rows h (row := 340) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_17
  simp only [← c22356, ← c22357, k2, ← c22358] at e_340_17
  have hr := e_340_17
  simp only [e_340_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 340 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 85)) := by
  have c22359 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22360 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22361 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22362 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22363 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22364 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_340_18 := arithEq_of_rows h (row := 340) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_18
  simp only [← c22359, ← c22360, ← c22361] at e_340_18
  have e_340_19 := arithEq_of_rows h (row := 340) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_340_19
  simp only [← c22362, ← c22363, k2, ← c22364] at e_340_19
  have hr := e_340_19
  simp only [e_340_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 341 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 86)) := by
  have c22365 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22366 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22367 := (publicBatchWrapper86_copies698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22368 := (publicBatchWrapper86_copies699 a h).1
  have c22369 := (publicBatchWrapper86_copies699 a h).2.1
  have c22370 := (publicBatchWrapper86_copies699 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_341_0 := arithEq_of_rows h (row := 341) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_0
  simp only [← c22365, ← c22366, ← c22367] at e_341_0
  have e_341_1 := arithEq_of_rows h (row := 341) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_1
  simp only [← c22368, ← c22369, k2, ← c22370] at e_341_1
  have hr := e_341_1
  simp only [e_341_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 341 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 87)) := by
  have c22371 := (publicBatchWrapper86_copies699 a h).2.2.2.1
  have c22372 := (publicBatchWrapper86_copies699 a h).2.2.2.2.1
  have c22373 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.1
  have c22374 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.1
  have c22375 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.1
  have c22376 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_341_2 := arithEq_of_rows h (row := 341) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_2
  simp only [← c22371, ← c22372, ← c22373] at e_341_2
  have e_341_3 := arithEq_of_rows h (row := 341) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_3
  simp only [← c22374, ← c22375, k2, ← c22376] at e_341_3
  have hr := e_341_3
  simp only [e_341_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 341 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 88)) := by
  have c22377 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.1
  have c22378 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22379 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22380 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22381 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22382 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_341_4 := arithEq_of_rows h (row := 341) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_4
  simp only [← c22377, ← c22378, ← c22379] at e_341_4
  have e_341_5 := arithEq_of_rows h (row := 341) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_5
  simp only [← c22380, ← c22381, k2, ← c22382] at e_341_5
  have hr := e_341_5
  simp only [e_341_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 341 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 89)) := by
  have c22383 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22384 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22385 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22386 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22387 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22388 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_341_6 := arithEq_of_rows h (row := 341) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_6
  simp only [← c22383, ← c22384, ← c22385] at e_341_6
  have e_341_7 := arithEq_of_rows h (row := 341) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_7
  simp only [← c22386, ← c22387, k2, ← c22388] at e_341_7
  have hr := e_341_7
  simp only [e_341_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 341 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 90)) := by
  have c22389 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22390 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22391 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22392 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22393 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22394 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_341_8 := arithEq_of_rows h (row := 341) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_8
  simp only [← c22389, ← c22390, ← c22391] at e_341_8
  have e_341_9 := arithEq_of_rows h (row := 341) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_9
  simp only [← c22392, ← c22393, k2, ← c22394] at e_341_9
  have hr := e_341_9
  simp only [e_341_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 341 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 91)) := by
  have c22395 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22396 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22397 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22398 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22399 := (publicBatchWrapper86_copies699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22400 := (publicBatchWrapper86_copies700 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_341_10 := arithEq_of_rows h (row := 341) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_10
  simp only [← c22395, ← c22396, ← c22397] at e_341_10
  have e_341_11 := arithEq_of_rows h (row := 341) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_11
  simp only [← c22398, ← c22399, k2, ← c22400] at e_341_11
  have hr := e_341_11
  simp only [e_341_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 341 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 92)) := by
  have c22401 := (publicBatchWrapper86_copies700 a h).2.1
  have c22402 := (publicBatchWrapper86_copies700 a h).2.2.1
  have c22403 := (publicBatchWrapper86_copies700 a h).2.2.2.1
  have c22404 := (publicBatchWrapper86_copies700 a h).2.2.2.2.1
  have c22405 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.1
  have c22406 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_341_12 := arithEq_of_rows h (row := 341) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_12
  simp only [← c22401, ← c22402, ← c22403] at e_341_12
  have e_341_13 := arithEq_of_rows h (row := 341) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_13
  simp only [← c22404, ← c22405, k2, ← c22406] at e_341_13
  have hr := e_341_13
  simp only [e_341_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 341 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 93)) := by
  have c22407 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.1
  have c22408 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.1
  have c22409 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.1
  have c22410 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22411 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22412 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_341_14 := arithEq_of_rows h (row := 341) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_14
  simp only [← c22407, ← c22408, ← c22409] at e_341_14
  have e_341_15 := arithEq_of_rows h (row := 341) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_15
  simp only [← c22410, ← c22411, k2, ← c22412] at e_341_15
  have hr := e_341_15
  simp only [e_341_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 341 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 94)) := by
  have c22413 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22414 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22415 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22416 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22417 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22418 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_341_16 := arithEq_of_rows h (row := 341) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_16
  simp only [← c22413, ← c22414, ← c22415] at e_341_16
  have e_341_17 := arithEq_of_rows h (row := 341) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_17
  simp only [← c22416, ← c22417, k2, ← c22418] at e_341_17
  have hr := e_341_17
  simp only [e_341_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 341 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 95)) := by
  have c22419 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22420 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22421 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22422 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22423 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22424 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_341_18 := arithEq_of_rows h (row := 341) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_18
  simp only [← c22419, ← c22420, ← c22421] at e_341_18
  have e_341_19 := arithEq_of_rows h (row := 341) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_341_19
  simp only [← c22422, ← c22423, k2, ← c22424] at e_341_19
  have hr := e_341_19
  simp only [e_341_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 342 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 96)) := by
  have c22425 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22426 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22427 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22428 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22429 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22430 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_342_0 := arithEq_of_rows h (row := 342) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_0
  simp only [← c22425, ← c22426, ← c22427] at e_342_0
  have e_342_1 := arithEq_of_rows h (row := 342) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_1
  simp only [← c22428, ← c22429, k2, ← c22430] at e_342_1
  have hr := e_342_1
  simp only [e_342_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 342 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 97)) := by
  have c22431 := (publicBatchWrapper86_copies700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22432 := (publicBatchWrapper86_copies701 a h).1
  have c22433 := (publicBatchWrapper86_copies701 a h).2.1
  have c22434 := (publicBatchWrapper86_copies701 a h).2.2.1
  have c22435 := (publicBatchWrapper86_copies701 a h).2.2.2.1
  have c22436 := (publicBatchWrapper86_copies701 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_342_2 := arithEq_of_rows h (row := 342) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_2
  simp only [← c22431, ← c22432, ← c22433] at e_342_2
  have e_342_3 := arithEq_of_rows h (row := 342) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_3
  simp only [← c22434, ← c22435, k2, ← c22436] at e_342_3
  have hr := e_342_3
  simp only [e_342_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 342 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 98)) := by
  have c22437 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.1
  have c22438 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.1
  have c22439 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.1
  have c22440 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.1
  have c22441 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.1
  have c22442 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_342_4 := arithEq_of_rows h (row := 342) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_4
  simp only [← c22437, ← c22438, ← c22439] at e_342_4
  have e_342_5 := arithEq_of_rows h (row := 342) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_5
  simp only [← c22440, ← c22441, k2, ← c22442] at e_342_5
  have hr := e_342_5
  simp only [e_342_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 342 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 99)) := by
  have c22443 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22444 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22445 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22446 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22447 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22448 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_342_6 := arithEq_of_rows h (row := 342) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_6
  simp only [← c22443, ← c22444, ← c22445] at e_342_6
  have e_342_7 := arithEq_of_rows h (row := 342) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_7
  simp only [← c22446, ← c22447, k2, ← c22448] at e_342_7
  have hr := e_342_7
  simp only [e_342_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 342 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 100)) := by
  have c22449 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22450 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22451 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22452 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22453 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22454 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_342_8 := arithEq_of_rows h (row := 342) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_8
  simp only [← c22449, ← c22450, ← c22451] at e_342_8
  have e_342_9 := arithEq_of_rows h (row := 342) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_9
  simp only [← c22452, ← c22453, k2, ← c22454] at e_342_9
  have hr := e_342_9
  simp only [e_342_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 342 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 101)) := by
  have c22455 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22456 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22457 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22458 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22459 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22460 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_342_10 := arithEq_of_rows h (row := 342) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_10
  simp only [← c22455, ← c22456, ← c22457] at e_342_10
  have e_342_11 := arithEq_of_rows h (row := 342) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_11
  simp only [← c22458, ← c22459, k2, ← c22460] at e_342_11
  have hr := e_342_11
  simp only [e_342_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 342 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 102)) := by
  have c22461 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22462 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22463 := (publicBatchWrapper86_copies701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22464 := (publicBatchWrapper86_copies702 a h).1
  have c22465 := (publicBatchWrapper86_copies702 a h).2.1
  have c22466 := (publicBatchWrapper86_copies702 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_342_12 := arithEq_of_rows h (row := 342) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_12
  simp only [← c22461, ← c22462, ← c22463] at e_342_12
  have e_342_13 := arithEq_of_rows h (row := 342) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_13
  simp only [← c22464, ← c22465, k2, ← c22466] at e_342_13
  have hr := e_342_13
  simp only [e_342_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 342 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 103)) := by
  have c22467 := (publicBatchWrapper86_copies702 a h).2.2.2.1
  have c22468 := (publicBatchWrapper86_copies702 a h).2.2.2.2.1
  have c22469 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.1
  have c22470 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.1
  have c22471 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.1
  have c22472 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_342_14 := arithEq_of_rows h (row := 342) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_14
  simp only [← c22467, ← c22468, ← c22469] at e_342_14
  have e_342_15 := arithEq_of_rows h (row := 342) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_15
  simp only [← c22470, ← c22471, k2, ← c22472] at e_342_15
  have hr := e_342_15
  simp only [e_342_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 342 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 104)) := by
  have c22473 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.1
  have c22474 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22475 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22476 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22477 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22478 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_342_16 := arithEq_of_rows h (row := 342) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_16
  simp only [← c22473, ← c22474, ← c22475] at e_342_16
  have e_342_17 := arithEq_of_rows h (row := 342) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_17
  simp only [← c22476, ← c22477, k2, ← c22478] at e_342_17
  have hr := e_342_17
  simp only [e_342_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 342 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 105)) := by
  have c22479 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22480 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22481 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22482 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22483 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22484 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_342_18 := arithEq_of_rows h (row := 342) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_18
  simp only [← c22479, ← c22480, ← c22481] at e_342_18
  have e_342_19 := arithEq_of_rows h (row := 342) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_342_19
  simp only [← c22482, ← c22483, k2, ← c22484] at e_342_19
  have hr := e_342_19
  simp only [e_342_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 343 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 106)) := by
  have c22485 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22486 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22487 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22488 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22489 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22490 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_343_0 := arithEq_of_rows h (row := 343) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_0
  simp only [← c22485, ← c22486, ← c22487] at e_343_0
  have e_343_1 := arithEq_of_rows h (row := 343) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_1
  simp only [← c22488, ← c22489, k2, ← c22490] at e_343_1
  have hr := e_343_1
  simp only [e_343_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 343 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 107)) := by
  have c22491 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22492 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22493 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22494 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22495 := (publicBatchWrapper86_copies702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22496 := (publicBatchWrapper86_copies703 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_343_2 := arithEq_of_rows h (row := 343) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_2
  simp only [← c22491, ← c22492, ← c22493] at e_343_2
  have e_343_3 := arithEq_of_rows h (row := 343) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_3
  simp only [← c22494, ← c22495, k2, ← c22496] at e_343_3
  have hr := e_343_3
  simp only [e_343_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 343 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 108)) := by
  have c22497 := (publicBatchWrapper86_copies703 a h).2.1
  have c22498 := (publicBatchWrapper86_copies703 a h).2.2.1
  have c22499 := (publicBatchWrapper86_copies703 a h).2.2.2.1
  have c22500 := (publicBatchWrapper86_copies703 a h).2.2.2.2.1
  have c22501 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.1
  have c22502 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_343_4 := arithEq_of_rows h (row := 343) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_4
  simp only [← c22497, ← c22498, ← c22499] at e_343_4
  have e_343_5 := arithEq_of_rows h (row := 343) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_5
  simp only [← c22500, ← c22501, k2, ← c22502] at e_343_5
  have hr := e_343_5
  simp only [e_343_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 343 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 109)) := by
  have c22503 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.1
  have c22504 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.1
  have c22505 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.1
  have c22506 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22507 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22508 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_343_6 := arithEq_of_rows h (row := 343) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_6
  simp only [← c22503, ← c22504, ← c22505] at e_343_6
  have e_343_7 := arithEq_of_rows h (row := 343) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_7
  simp only [← c22506, ← c22507, k2, ← c22508] at e_343_7
  have hr := e_343_7
  simp only [e_343_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 343 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 110)) := by
  have c22509 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22510 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22511 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22512 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22513 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22514 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_343_8 := arithEq_of_rows h (row := 343) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_8
  simp only [← c22509, ← c22510, ← c22511] at e_343_8
  have e_343_9 := arithEq_of_rows h (row := 343) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_9
  simp only [← c22512, ← c22513, k2, ← c22514] at e_343_9
  have hr := e_343_9
  simp only [e_343_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 343 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 111)) := by
  have c22515 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22516 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22517 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22518 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22519 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22520 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_343_10 := arithEq_of_rows h (row := 343) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_10
  simp only [← c22515, ← c22516, ← c22517] at e_343_10
  have e_343_11 := arithEq_of_rows h (row := 343) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_11
  simp only [← c22518, ← c22519, k2, ← c22520] at e_343_11
  have hr := e_343_11
  simp only [e_343_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 343 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 112)) := by
  have c22521 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22522 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22523 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22524 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22525 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22526 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_343_12 := arithEq_of_rows h (row := 343) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_12
  simp only [← c22521, ← c22522, ← c22523] at e_343_12
  have e_343_13 := arithEq_of_rows h (row := 343) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_13
  simp only [← c22524, ← c22525, k2, ← c22526] at e_343_13
  have hr := e_343_13
  simp only [e_343_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 343 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 113)) := by
  have c22527 := (publicBatchWrapper86_copies703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22528 := (publicBatchWrapper86_copies704 a h).1
  have c22529 := (publicBatchWrapper86_copies704 a h).2.1
  have c22530 := (publicBatchWrapper86_copies704 a h).2.2.1
  have c22531 := (publicBatchWrapper86_copies704 a h).2.2.2.1
  have c22532 := (publicBatchWrapper86_copies704 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_343_14 := arithEq_of_rows h (row := 343) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_14
  simp only [← c22527, ← c22528, ← c22529] at e_343_14
  have e_343_15 := arithEq_of_rows h (row := 343) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_15
  simp only [← c22530, ← c22531, k2, ← c22532] at e_343_15
  have hr := e_343_15
  simp only [e_343_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 343 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 114)) := by
  have c22533 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.1
  have c22534 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.1
  have c22535 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.1
  have c22536 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.1
  have c22537 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.1
  have c22538 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_343_16 := arithEq_of_rows h (row := 343) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_16
  simp only [← c22533, ← c22534, ← c22535] at e_343_16
  have e_343_17 := arithEq_of_rows h (row := 343) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_17
  simp only [← c22536, ← c22537, k2, ← c22538] at e_343_17
  have hr := e_343_17
  simp only [e_343_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 343 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 115)) := by
  have c22539 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22540 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22541 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22542 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22543 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22544 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_343_18 := arithEq_of_rows h (row := 343) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_18
  simp only [← c22539, ← c22540, ← c22541] at e_343_18
  have e_343_19 := arithEq_of_rows h (row := 343) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_343_19
  simp only [← c22542, ← c22543, k2, ← c22544] at e_343_19
  have hr := e_343_19
  simp only [e_343_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 344 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 116)) := by
  have c22545 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22546 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22547 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22548 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22549 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22550 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_344_0 := arithEq_of_rows h (row := 344) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_0
  simp only [← c22545, ← c22546, ← c22547] at e_344_0
  have e_344_1 := arithEq_of_rows h (row := 344) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_1
  simp only [← c22548, ← c22549, k2, ← c22550] at e_344_1
  have hr := e_344_1
  simp only [e_344_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 344 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 117)) := by
  have c22551 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22552 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22553 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22554 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22555 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22556 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_344_2 := arithEq_of_rows h (row := 344) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_2
  simp only [← c22551, ← c22552, ← c22553] at e_344_2
  have e_344_3 := arithEq_of_rows h (row := 344) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_3
  simp only [← c22554, ← c22555, k2, ← c22556] at e_344_3
  have hr := e_344_3
  simp only [e_344_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 344 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 118)) := by
  have c22557 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22558 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22559 := (publicBatchWrapper86_copies704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22560 := (publicBatchWrapper86_copies705 a h).1
  have c22561 := (publicBatchWrapper86_copies705 a h).2.1
  have c22562 := (publicBatchWrapper86_copies705 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_344_4 := arithEq_of_rows h (row := 344) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_4
  simp only [← c22557, ← c22558, ← c22559] at e_344_4
  have e_344_5 := arithEq_of_rows h (row := 344) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_5
  simp only [← c22560, ← c22561, k2, ← c22562] at e_344_5
  have hr := e_344_5
  simp only [e_344_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 344 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 119)) := by
  have c22563 := (publicBatchWrapper86_copies705 a h).2.2.2.1
  have c22564 := (publicBatchWrapper86_copies705 a h).2.2.2.2.1
  have c22565 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.1
  have c22566 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.1
  have c22567 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.1
  have c22568 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_344_6 := arithEq_of_rows h (row := 344) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_6
  simp only [← c22563, ← c22564, ← c22565] at e_344_6
  have e_344_7 := arithEq_of_rows h (row := 344) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_7
  simp only [← c22566, ← c22567, k2, ← c22568] at e_344_7
  have hr := e_344_7
  simp only [e_344_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 344 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 120)) := by
  have c22569 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.1
  have c22570 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22571 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22572 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22573 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22574 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_344_8 := arithEq_of_rows h (row := 344) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_8
  simp only [← c22569, ← c22570, ← c22571] at e_344_8
  have e_344_9 := arithEq_of_rows h (row := 344) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_9
  simp only [← c22572, ← c22573, k2, ← c22574] at e_344_9
  have hr := e_344_9
  simp only [e_344_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 344 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 121)) := by
  have c22575 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22576 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22577 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22578 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22579 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22580 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_344_10 := arithEq_of_rows h (row := 344) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_10
  simp only [← c22575, ← c22576, ← c22577] at e_344_10
  have e_344_11 := arithEq_of_rows h (row := 344) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_11
  simp only [← c22578, ← c22579, k2, ← c22580] at e_344_11
  have hr := e_344_11
  simp only [e_344_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 344 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 122)) := by
  have c22581 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22582 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22583 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22584 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22585 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22586 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_344_12 := arithEq_of_rows h (row := 344) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_12
  simp only [← c22581, ← c22582, ← c22583] at e_344_12
  have e_344_13 := arithEq_of_rows h (row := 344) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_13
  simp only [← c22584, ← c22585, k2, ← c22586] at e_344_13
  have hr := e_344_13
  simp only [e_344_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 344 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 123)) := by
  have c22587 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22588 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22589 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22590 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22591 := (publicBatchWrapper86_copies705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22592 := (publicBatchWrapper86_copies706 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_344_14 := arithEq_of_rows h (row := 344) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_14
  simp only [← c22587, ← c22588, ← c22589] at e_344_14
  have e_344_15 := arithEq_of_rows h (row := 344) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_15
  simp only [← c22590, ← c22591, k2, ← c22592] at e_344_15
  have hr := e_344_15
  simp only [e_344_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 344 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 124)) := by
  have c22593 := (publicBatchWrapper86_copies706 a h).2.1
  have c22594 := (publicBatchWrapper86_copies706 a h).2.2.1
  have c22595 := (publicBatchWrapper86_copies706 a h).2.2.2.1
  have c22596 := (publicBatchWrapper86_copies706 a h).2.2.2.2.1
  have c22597 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.1
  have c22598 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_344_16 := arithEq_of_rows h (row := 344) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_16
  simp only [← c22593, ← c22594, ← c22595] at e_344_16
  have e_344_17 := arithEq_of_rows h (row := 344) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_17
  simp only [← c22596, ← c22597, k2, ← c22598] at e_344_17
  have hr := e_344_17
  simp only [e_344_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 344 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 125)) := by
  have c22599 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.1
  have c22600 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.1
  have c22601 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.1
  have c22602 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22603 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22604 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_344_18 := arithEq_of_rows h (row := 344) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_18
  simp only [← c22599, ← c22600, ← c22601] at e_344_18
  have e_344_19 := arithEq_of_rows h (row := 344) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_344_19
  simp only [← c22602, ← c22603, k2, ← c22604] at e_344_19
  have hr := e_344_19
  simp only [e_344_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 345 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 126)) := by
  have c22605 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22606 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22607 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22608 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22609 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22610 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_345_0 := arithEq_of_rows h (row := 345) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_0
  simp only [← c22605, ← c22606, ← c22607] at e_345_0
  have e_345_1 := arithEq_of_rows h (row := 345) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_1
  simp only [← c22608, ← c22609, k2, ← c22610] at e_345_1
  have hr := e_345_1
  simp only [e_345_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 345 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 127)) := by
  have c22611 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22612 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22613 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22614 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22615 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22616 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_345_2 := arithEq_of_rows h (row := 345) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_2
  simp only [← c22611, ← c22612, ← c22613] at e_345_2
  have e_345_3 := arithEq_of_rows h (row := 345) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_3
  simp only [← c22614, ← c22615, k2, ← c22616] at e_345_3
  have hr := e_345_3
  simp only [e_345_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 345 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 128)) := by
  have c22617 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22618 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22619 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22620 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22621 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22622 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_345_4 := arithEq_of_rows h (row := 345) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_4
  simp only [← c22617, ← c22618, ← c22619] at e_345_4
  have e_345_5 := arithEq_of_rows h (row := 345) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_5
  simp only [← c22620, ← c22621, k2, ← c22622] at e_345_5
  have hr := e_345_5
  simp only [e_345_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 345 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 129)) := by
  have c22623 := (publicBatchWrapper86_copies706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22624 := (publicBatchWrapper86_copies707 a h).1
  have c22625 := (publicBatchWrapper86_copies707 a h).2.1
  have c22626 := (publicBatchWrapper86_copies707 a h).2.2.1
  have c22627 := (publicBatchWrapper86_copies707 a h).2.2.2.1
  have c22628 := (publicBatchWrapper86_copies707 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_345_6 := arithEq_of_rows h (row := 345) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_6
  simp only [← c22623, ← c22624, ← c22625] at e_345_6
  have e_345_7 := arithEq_of_rows h (row := 345) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_7
  simp only [← c22626, ← c22627, k2, ← c22628] at e_345_7
  have hr := e_345_7
  simp only [e_345_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 345 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 130)) := by
  have c22629 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.1
  have c22630 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.1
  have c22631 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.1
  have c22632 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.1
  have c22633 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.1
  have c22634 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_345_8 := arithEq_of_rows h (row := 345) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_8
  simp only [← c22629, ← c22630, ← c22631] at e_345_8
  have e_345_9 := arithEq_of_rows h (row := 345) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_9
  simp only [← c22632, ← c22633, k2, ← c22634] at e_345_9
  have hr := e_345_9
  simp only [e_345_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 345 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 131)) := by
  have c22635 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22636 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22637 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22638 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22639 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22640 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_345_10 := arithEq_of_rows h (row := 345) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_10
  simp only [← c22635, ← c22636, ← c22637] at e_345_10
  have e_345_11 := arithEq_of_rows h (row := 345) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_11
  simp only [← c22638, ← c22639, k2, ← c22640] at e_345_11
  have hr := e_345_11
  simp only [e_345_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 345 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 132)) := by
  have c22641 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22642 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22643 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22644 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22645 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22646 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_345_12 := arithEq_of_rows h (row := 345) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_12
  simp only [← c22641, ← c22642, ← c22643] at e_345_12
  have e_345_13 := arithEq_of_rows h (row := 345) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_13
  simp only [← c22644, ← c22645, k2, ← c22646] at e_345_13
  have hr := e_345_13
  simp only [e_345_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 345 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 133)) := by
  have c22647 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22648 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22649 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22650 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22651 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22652 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_345_14 := arithEq_of_rows h (row := 345) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_14
  simp only [← c22647, ← c22648, ← c22649] at e_345_14
  have e_345_15 := arithEq_of_rows h (row := 345) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_15
  simp only [← c22650, ← c22651, k2, ← c22652] at e_345_15
  have hr := e_345_15
  simp only [e_345_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 345 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 134)) := by
  have c22653 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22654 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22655 := (publicBatchWrapper86_copies707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22656 := (publicBatchWrapper86_copies708 a h).1
  have c22657 := (publicBatchWrapper86_copies708 a h).2.1
  have c22658 := (publicBatchWrapper86_copies708 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_345_16 := arithEq_of_rows h (row := 345) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_16
  simp only [← c22653, ← c22654, ← c22655] at e_345_16
  have e_345_17 := arithEq_of_rows h (row := 345) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_17
  simp only [← c22656, ← c22657, k2, ← c22658] at e_345_17
  have hr := e_345_17
  simp only [e_345_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 345 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 135)) := by
  have c22659 := (publicBatchWrapper86_copies708 a h).2.2.2.1
  have c22660 := (publicBatchWrapper86_copies708 a h).2.2.2.2.1
  have c22661 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.1
  have c22662 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.1
  have c22663 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.1
  have c22664 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_345_18 := arithEq_of_rows h (row := 345) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_18
  simp only [← c22659, ← c22660, ← c22661] at e_345_18
  have e_345_19 := arithEq_of_rows h (row := 345) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_345_19
  simp only [← c22662, ← c22663, k2, ← c22664] at e_345_19
  have hr := e_345_19
  simp only [e_345_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 346 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 136)) := by
  have c22665 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.1
  have c22666 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22667 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22668 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22669 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22670 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_346_0 := arithEq_of_rows h (row := 346) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_0
  simp only [← c22665, ← c22666, ← c22667] at e_346_0
  have e_346_1 := arithEq_of_rows h (row := 346) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_1
  simp only [← c22668, ← c22669, k2, ← c22670] at e_346_1
  have hr := e_346_1
  simp only [e_346_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 346 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 137)) := by
  have c22671 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22672 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22673 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22674 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22675 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22676 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_346_2 := arithEq_of_rows h (row := 346) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_2
  simp only [← c22671, ← c22672, ← c22673] at e_346_2
  have e_346_3 := arithEq_of_rows h (row := 346) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_3
  simp only [← c22674, ← c22675, k2, ← c22676] at e_346_3
  have hr := e_346_3
  simp only [e_346_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 346 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 138)) := by
  have c22677 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22678 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22679 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22680 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22681 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22682 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_346_4 := arithEq_of_rows h (row := 346) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_4
  simp only [← c22677, ← c22678, ← c22679] at e_346_4
  have e_346_5 := arithEq_of_rows h (row := 346) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_5
  simp only [← c22680, ← c22681, k2, ← c22682] at e_346_5
  have hr := e_346_5
  simp only [e_346_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 346 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 139)) := by
  have c22683 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22684 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22685 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22686 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22687 := (publicBatchWrapper86_copies708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22688 := (publicBatchWrapper86_copies709 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_346_6 := arithEq_of_rows h (row := 346) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_6
  simp only [← c22683, ← c22684, ← c22685] at e_346_6
  have e_346_7 := arithEq_of_rows h (row := 346) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_7
  simp only [← c22686, ← c22687, k2, ← c22688] at e_346_7
  have hr := e_346_7
  simp only [e_346_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 346 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 140)) := by
  have c22689 := (publicBatchWrapper86_copies709 a h).2.1
  have c22690 := (publicBatchWrapper86_copies709 a h).2.2.1
  have c22691 := (publicBatchWrapper86_copies709 a h).2.2.2.1
  have c22692 := (publicBatchWrapper86_copies709 a h).2.2.2.2.1
  have c22693 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.1
  have c22694 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_346_8 := arithEq_of_rows h (row := 346) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_8
  simp only [← c22689, ← c22690, ← c22691] at e_346_8
  have e_346_9 := arithEq_of_rows h (row := 346) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_9
  simp only [← c22692, ← c22693, k2, ← c22694] at e_346_9
  have hr := e_346_9
  simp only [e_346_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 346 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 141)) := by
  have c22695 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.1
  have c22696 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.1
  have c22697 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.1
  have c22698 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22699 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22700 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_346_10 := arithEq_of_rows h (row := 346) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_10
  simp only [← c22695, ← c22696, ← c22697] at e_346_10
  have e_346_11 := arithEq_of_rows h (row := 346) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_11
  simp only [← c22698, ← c22699, k2, ← c22700] at e_346_11
  have hr := e_346_11
  simp only [e_346_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 346 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 142)) := by
  have c22701 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22702 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22703 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22704 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22705 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22706 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_346_12 := arithEq_of_rows h (row := 346) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_12
  simp only [← c22701, ← c22702, ← c22703] at e_346_12
  have e_346_13 := arithEq_of_rows h (row := 346) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_13
  simp only [← c22704, ← c22705, k2, ← c22706] at e_346_13
  have hr := e_346_13
  simp only [e_346_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 346 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 143)) := by
  have c22707 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22708 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22709 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22710 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22711 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22712 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_346_14 := arithEq_of_rows h (row := 346) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_14
  simp only [← c22707, ← c22708, ← c22709] at e_346_14
  have e_346_15 := arithEq_of_rows h (row := 346) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_15
  simp only [← c22710, ← c22711, k2, ← c22712] at e_346_15
  have hr := e_346_15
  simp only [e_346_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 346 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 144)) := by
  have c22713 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22714 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22715 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22716 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22717 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22718 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_346_16 := arithEq_of_rows h (row := 346) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_16
  simp only [← c22713, ← c22714, ← c22715] at e_346_16
  have e_346_17 := arithEq_of_rows h (row := 346) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_17
  simp only [← c22716, ← c22717, k2, ← c22718] at e_346_17
  have hr := e_346_17
  simp only [e_346_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 346 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 145)) := by
  have c22719 := (publicBatchWrapper86_copies709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22720 := (publicBatchWrapper86_copies710 a h).1
  have c22721 := (publicBatchWrapper86_copies710 a h).2.1
  have c22722 := (publicBatchWrapper86_copies710 a h).2.2.1
  have c22723 := (publicBatchWrapper86_copies710 a h).2.2.2.1
  have c22724 := (publicBatchWrapper86_copies710 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_346_18 := arithEq_of_rows h (row := 346) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_18
  simp only [← c22719, ← c22720, ← c22721] at e_346_18
  have e_346_19 := arithEq_of_rows h (row := 346) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_346_19
  simp only [← c22722, ← c22723, k2, ← c22724] at e_346_19
  have hr := e_346_19
  simp only [e_346_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 347 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 146)) := by
  have c22725 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.1
  have c22726 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.1
  have c22727 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.1
  have c22728 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.1
  have c22729 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.1
  have c22730 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_347_0 := arithEq_of_rows h (row := 347) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_0
  simp only [← c22725, ← c22726, ← c22727] at e_347_0
  have e_347_1 := arithEq_of_rows h (row := 347) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_1
  simp only [← c22728, ← c22729, k2, ← c22730] at e_347_1
  have hr := e_347_1
  simp only [e_347_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 347 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 147)) := by
  have c22731 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22732 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22733 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22734 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22735 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22736 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_347_2 := arithEq_of_rows h (row := 347) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_2
  simp only [← c22731, ← c22732, ← c22733] at e_347_2
  have e_347_3 := arithEq_of_rows h (row := 347) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_3
  simp only [← c22734, ← c22735, k2, ← c22736] at e_347_3
  have hr := e_347_3
  simp only [e_347_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 347 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 148)) := by
  have c22737 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22738 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22739 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22740 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22741 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22742 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_347_4 := arithEq_of_rows h (row := 347) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_4
  simp only [← c22737, ← c22738, ← c22739] at e_347_4
  have e_347_5 := arithEq_of_rows h (row := 347) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_5
  simp only [← c22740, ← c22741, k2, ← c22742] at e_347_5
  have hr := e_347_5
  simp only [e_347_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 347 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 149)) := by
  have c22743 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22744 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22745 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22746 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22747 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22748 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_347_6 := arithEq_of_rows h (row := 347) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_6
  simp only [← c22743, ← c22744, ← c22745] at e_347_6
  have e_347_7 := arithEq_of_rows h (row := 347) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_7
  simp only [← c22746, ← c22747, k2, ← c22748] at e_347_7
  have hr := e_347_7
  simp only [e_347_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 347 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 150)) := by
  have c22749 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22750 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22751 := (publicBatchWrapper86_copies710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22752 := (publicBatchWrapper86_copies711 a h).1
  have c22753 := (publicBatchWrapper86_copies711 a h).2.1
  have c22754 := (publicBatchWrapper86_copies711 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_347_8 := arithEq_of_rows h (row := 347) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_8
  simp only [← c22749, ← c22750, ← c22751] at e_347_8
  have e_347_9 := arithEq_of_rows h (row := 347) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_9
  simp only [← c22752, ← c22753, k2, ← c22754] at e_347_9
  have hr := e_347_9
  simp only [e_347_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 347 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 151)) := by
  have c22755 := (publicBatchWrapper86_copies711 a h).2.2.2.1
  have c22756 := (publicBatchWrapper86_copies711 a h).2.2.2.2.1
  have c22757 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.1
  have c22758 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.1
  have c22759 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.1
  have c22760 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_347_10 := arithEq_of_rows h (row := 347) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_10
  simp only [← c22755, ← c22756, ← c22757] at e_347_10
  have e_347_11 := arithEq_of_rows h (row := 347) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_11
  simp only [← c22758, ← c22759, k2, ← c22760] at e_347_11
  have hr := e_347_11
  simp only [e_347_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 347 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 152)) := by
  have c22761 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.1
  have c22762 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22763 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22764 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22765 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22766 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_347_12 := arithEq_of_rows h (row := 347) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_12
  simp only [← c22761, ← c22762, ← c22763] at e_347_12
  have e_347_13 := arithEq_of_rows h (row := 347) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_13
  simp only [← c22764, ← c22765, k2, ← c22766] at e_347_13
  have hr := e_347_13
  simp only [e_347_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 347 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 153)) := by
  have c22767 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22768 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22769 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22770 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22771 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22772 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_347_14 := arithEq_of_rows h (row := 347) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_14
  simp only [← c22767, ← c22768, ← c22769] at e_347_14
  have e_347_15 := arithEq_of_rows h (row := 347) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_15
  simp only [← c22770, ← c22771, k2, ← c22772] at e_347_15
  have hr := e_347_15
  simp only [e_347_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 347 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 154)) := by
  have c22773 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22774 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22775 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22776 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22777 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22778 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_347_16 := arithEq_of_rows h (row := 347) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_16
  simp only [← c22773, ← c22774, ← c22775] at e_347_16
  have e_347_17 := arithEq_of_rows h (row := 347) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_17
  simp only [← c22776, ← c22777, k2, ← c22778] at e_347_17
  have hr := e_347_17
  simp only [e_347_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 347 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 155)) := by
  have c22779 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22780 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22781 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22782 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22783 := (publicBatchWrapper86_copies711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22784 := (publicBatchWrapper86_copies712 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_347_18 := arithEq_of_rows h (row := 347) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_18
  simp only [← c22779, ← c22780, ← c22781] at e_347_18
  have e_347_19 := arithEq_of_rows h (row := 347) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_347_19
  simp only [← c22782, ← c22783, k2, ← c22784] at e_347_19
  have hr := e_347_19
  simp only [e_347_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 348 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 156)) := by
  have c22785 := (publicBatchWrapper86_copies712 a h).2.1
  have c22786 := (publicBatchWrapper86_copies712 a h).2.2.1
  have c22787 := (publicBatchWrapper86_copies712 a h).2.2.2.1
  have c22788 := (publicBatchWrapper86_copies712 a h).2.2.2.2.1
  have c22789 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.1
  have c22790 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_348_0 := arithEq_of_rows h (row := 348) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_0
  simp only [← c22785, ← c22786, ← c22787] at e_348_0
  have e_348_1 := arithEq_of_rows h (row := 348) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_1
  simp only [← c22788, ← c22789, k2, ← c22790] at e_348_1
  have hr := e_348_1
  simp only [e_348_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 348 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 157)) := by
  have c22791 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.1
  have c22792 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.1
  have c22793 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.1
  have c22794 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22795 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22796 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_348_2 := arithEq_of_rows h (row := 348) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_2
  simp only [← c22791, ← c22792, ← c22793] at e_348_2
  have e_348_3 := arithEq_of_rows h (row := 348) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_3
  simp only [← c22794, ← c22795, k2, ← c22796] at e_348_3
  have hr := e_348_3
  simp only [e_348_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 348 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 158)) := by
  have c22797 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22798 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22799 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22800 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22801 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22802 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_348_4 := arithEq_of_rows h (row := 348) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_4
  simp only [← c22797, ← c22798, ← c22799] at e_348_4
  have e_348_5 := arithEq_of_rows h (row := 348) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_5
  simp only [← c22800, ← c22801, k2, ← c22802] at e_348_5
  have hr := e_348_5
  simp only [e_348_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 348 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 159)) := by
  have c22803 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22804 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22805 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22806 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22807 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22808 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_348_6 := arithEq_of_rows h (row := 348) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_6
  simp only [← c22803, ← c22804, ← c22805] at e_348_6
  have e_348_7 := arithEq_of_rows h (row := 348) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_7
  simp only [← c22806, ← c22807, k2, ← c22808] at e_348_7
  have hr := e_348_7
  simp only [e_348_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 348 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 160)) := by
  have c22809 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22810 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22811 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22812 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22813 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22814 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_348_8 := arithEq_of_rows h (row := 348) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_8
  simp only [← c22809, ← c22810, ← c22811] at e_348_8
  have e_348_9 := arithEq_of_rows h (row := 348) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_9
  simp only [← c22812, ← c22813, k2, ← c22814] at e_348_9
  have hr := e_348_9
  simp only [e_348_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 348 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 161)) := by
  have c22815 := (publicBatchWrapper86_copies712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22816 := (publicBatchWrapper86_copies713 a h).1
  have c22817 := (publicBatchWrapper86_copies713 a h).2.1
  have c22818 := (publicBatchWrapper86_copies713 a h).2.2.1
  have c22819 := (publicBatchWrapper86_copies713 a h).2.2.2.1
  have c22820 := (publicBatchWrapper86_copies713 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_348_10 := arithEq_of_rows h (row := 348) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_10
  simp only [← c22815, ← c22816, ← c22817] at e_348_10
  have e_348_11 := arithEq_of_rows h (row := 348) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_11
  simp only [← c22818, ← c22819, k2, ← c22820] at e_348_11
  have hr := e_348_11
  simp only [e_348_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 348 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 162)) := by
  have c22821 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.1
  have c22822 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.1
  have c22823 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.1
  have c22824 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.1
  have c22825 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.1
  have c22826 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_348_12 := arithEq_of_rows h (row := 348) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_12
  simp only [← c22821, ← c22822, ← c22823] at e_348_12
  have e_348_13 := arithEq_of_rows h (row := 348) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_13
  simp only [← c22824, ← c22825, k2, ← c22826] at e_348_13
  have hr := e_348_13
  simp only [e_348_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 348 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 163)) := by
  have c22827 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22828 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22829 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22830 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22831 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22832 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_348_14 := arithEq_of_rows h (row := 348) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_14
  simp only [← c22827, ← c22828, ← c22829] at e_348_14
  have e_348_15 := arithEq_of_rows h (row := 348) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_15
  simp only [← c22830, ← c22831, k2, ← c22832] at e_348_15
  have hr := e_348_15
  simp only [e_348_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 348 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 164)) := by
  have c22833 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22834 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22835 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22836 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22837 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22838 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_348_16 := arithEq_of_rows h (row := 348) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_16
  simp only [← c22833, ← c22834, ← c22835] at e_348_16
  have e_348_17 := arithEq_of_rows h (row := 348) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_17
  simp only [← c22836, ← c22837, k2, ← c22838] at e_348_17
  have hr := e_348_17
  simp only [e_348_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 348 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 165)) := by
  have c22839 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22840 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22841 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22842 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22843 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22844 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_348_18 := arithEq_of_rows h (row := 348) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_18
  simp only [← c22839, ← c22840, ← c22841] at e_348_18
  have e_348_19 := arithEq_of_rows h (row := 348) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_348_19
  simp only [← c22842, ← c22843, k2, ← c22844] at e_348_19
  have hr := e_348_19
  simp only [e_348_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 349 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 166)) := by
  have c22845 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22846 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22847 := (publicBatchWrapper86_copies713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22848 := (publicBatchWrapper86_copies714 a h).1
  have c22849 := (publicBatchWrapper86_copies714 a h).2.1
  have c22850 := (publicBatchWrapper86_copies714 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_349_0 := arithEq_of_rows h (row := 349) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_0
  simp only [← c22845, ← c22846, ← c22847] at e_349_0
  have e_349_1 := arithEq_of_rows h (row := 349) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_1
  simp only [← c22848, ← c22849, k2, ← c22850] at e_349_1
  have hr := e_349_1
  simp only [e_349_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 349 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 167)) := by
  have c22851 := (publicBatchWrapper86_copies714 a h).2.2.2.1
  have c22852 := (publicBatchWrapper86_copies714 a h).2.2.2.2.1
  have c22853 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.1
  have c22854 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.1
  have c22855 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.1
  have c22856 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_349_2 := arithEq_of_rows h (row := 349) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_2
  simp only [← c22851, ← c22852, ← c22853] at e_349_2
  have e_349_3 := arithEq_of_rows h (row := 349) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_3
  simp only [← c22854, ← c22855, k2, ← c22856] at e_349_3
  have hr := e_349_3
  simp only [e_349_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 349 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 210)) := by
  have c22857 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.1
  have c22858 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22859 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22860 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22861 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22862 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_349_4 := arithEq_of_rows h (row := 349) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_4
  simp only [← c22857, ← c22858, ← c22859] at e_349_4
  have e_349_5 := arithEq_of_rows h (row := 349) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_5
  simp only [← c22860, ← c22861, k2, ← c22862] at e_349_5
  have hr := e_349_5
  simp only [e_349_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 349 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 211)) := by
  have c22863 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22864 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22865 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22866 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22867 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22868 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_349_6 := arithEq_of_rows h (row := 349) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_6
  simp only [← c22863, ← c22864, ← c22865] at e_349_6
  have e_349_7 := arithEq_of_rows h (row := 349) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_7
  simp only [← c22866, ← c22867, k2, ← c22868] at e_349_7
  have hr := e_349_7
  simp only [e_349_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 349 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 212)) := by
  have c22869 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22870 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22871 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22872 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22873 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22874 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_349_8 := arithEq_of_rows h (row := 349) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_8
  simp only [← c22869, ← c22870, ← c22871] at e_349_8
  have e_349_9 := arithEq_of_rows h (row := 349) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_9
  simp only [← c22872, ← c22873, k2, ← c22874] at e_349_9
  have hr := e_349_9
  simp only [e_349_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 349 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 213)) := by
  have c22875 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22876 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22877 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22878 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22879 := (publicBatchWrapper86_copies714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22880 := (publicBatchWrapper86_copies715 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_349_10 := arithEq_of_rows h (row := 349) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_10
  simp only [← c22875, ← c22876, ← c22877] at e_349_10
  have e_349_11 := arithEq_of_rows h (row := 349) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_11
  simp only [← c22878, ← c22879, k2, ← c22880] at e_349_11
  have hr := e_349_11
  simp only [e_349_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 349 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 214)) := by
  have c22881 := (publicBatchWrapper86_copies715 a h).2.1
  have c22882 := (publicBatchWrapper86_copies715 a h).2.2.1
  have c22883 := (publicBatchWrapper86_copies715 a h).2.2.2.1
  have c22884 := (publicBatchWrapper86_copies715 a h).2.2.2.2.1
  have c22885 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.1
  have c22886 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_349_12 := arithEq_of_rows h (row := 349) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_12
  simp only [← c22881, ← c22882, ← c22883] at e_349_12
  have e_349_13 := arithEq_of_rows h (row := 349) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_13
  simp only [← c22884, ← c22885, k2, ← c22886] at e_349_13
  have hr := e_349_13
  simp only [e_349_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 349 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 215)) := by
  have c22887 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.1
  have c22888 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.1
  have c22889 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.1
  have c22890 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22891 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22892 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_349_14 := arithEq_of_rows h (row := 349) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_14
  simp only [← c22887, ← c22888, ← c22889] at e_349_14
  have e_349_15 := arithEq_of_rows h (row := 349) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_15
  simp only [← c22890, ← c22891, k2, ← c22892] at e_349_15
  have hr := e_349_15
  simp only [e_349_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 349 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 216)) := by
  have c22893 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22894 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22895 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22896 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22897 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22898 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_349_16 := arithEq_of_rows h (row := 349) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_16
  simp only [← c22893, ← c22894, ← c22895] at e_349_16
  have e_349_17 := arithEq_of_rows h (row := 349) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_17
  simp only [← c22896, ← c22897, k2, ← c22898] at e_349_17
  have hr := e_349_17
  simp only [e_349_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 349 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 217)) := by
  have c22899 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22900 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22901 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22902 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22903 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22904 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_349_18 := arithEq_of_rows h (row := 349) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_18
  simp only [← c22899, ← c22900, ← c22901] at e_349_18
  have e_349_19 := arithEq_of_rows h (row := 349) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_349_19
  simp only [← c22902, ← c22903, k2, ← c22904] at e_349_19
  have hr := e_349_19
  simp only [e_349_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 350 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 218)) := by
  have c22905 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22906 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22907 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22908 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22909 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22910 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_350_0 := arithEq_of_rows h (row := 350) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_0
  simp only [← c22905, ← c22906, ← c22907] at e_350_0
  have e_350_1 := arithEq_of_rows h (row := 350) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_1
  simp only [← c22908, ← c22909, k2, ← c22910] at e_350_1
  have hr := e_350_1
  simp only [e_350_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 350 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 219)) := by
  have c22911 := (publicBatchWrapper86_copies715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22912 := (publicBatchWrapper86_copies716 a h).1
  have c22913 := (publicBatchWrapper86_copies716 a h).2.1
  have c22914 := (publicBatchWrapper86_copies716 a h).2.2.1
  have c22915 := (publicBatchWrapper86_copies716 a h).2.2.2.1
  have c22916 := (publicBatchWrapper86_copies716 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_350_2 := arithEq_of_rows h (row := 350) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_2
  simp only [← c22911, ← c22912, ← c22913] at e_350_2
  have e_350_3 := arithEq_of_rows h (row := 350) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_3
  simp only [← c22914, ← c22915, k2, ← c22916] at e_350_3
  have hr := e_350_3
  simp only [e_350_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 350 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 220)) := by
  have c22917 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.1
  have c22918 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.1
  have c22919 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.1
  have c22920 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.1
  have c22921 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.1
  have c22922 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_350_4 := arithEq_of_rows h (row := 350) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_4
  simp only [← c22917, ← c22918, ← c22919] at e_350_4
  have e_350_5 := arithEq_of_rows h (row := 350) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_5
  simp only [← c22920, ← c22921, k2, ← c22922] at e_350_5
  have hr := e_350_5
  simp only [e_350_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 350 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 221)) := by
  have c22923 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22924 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22925 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22926 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22927 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22928 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_350_6 := arithEq_of_rows h (row := 350) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_6
  simp only [← c22923, ← c22924, ← c22925] at e_350_6
  have e_350_7 := arithEq_of_rows h (row := 350) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_7
  simp only [← c22926, ← c22927, k2, ← c22928] at e_350_7
  have hr := e_350_7
  simp only [e_350_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 350 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 222)) := by
  have c22929 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22930 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22931 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22932 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22933 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22934 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_350_8 := arithEq_of_rows h (row := 350) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_8
  simp only [← c22929, ← c22930, ← c22931] at e_350_8
  have e_350_9 := arithEq_of_rows h (row := 350) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_9
  simp only [← c22932, ← c22933, k2, ← c22934] at e_350_9
  have hr := e_350_9
  simp only [e_350_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 350 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 223)) := by
  have c22935 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22936 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22937 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22938 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22939 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22940 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_350_10 := arithEq_of_rows h (row := 350) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_10
  simp only [← c22935, ← c22936, ← c22937] at e_350_10
  have e_350_11 := arithEq_of_rows h (row := 350) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_11
  simp only [← c22938, ← c22939, k2, ← c22940] at e_350_11
  have hr := e_350_11
  simp only [e_350_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 350 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 224)) := by
  have c22941 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22942 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22943 := (publicBatchWrapper86_copies716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22944 := (publicBatchWrapper86_copies717 a h).1
  have c22945 := (publicBatchWrapper86_copies717 a h).2.1
  have c22946 := (publicBatchWrapper86_copies717 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_350_12 := arithEq_of_rows h (row := 350) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_12
  simp only [← c22941, ← c22942, ← c22943] at e_350_12
  have e_350_13 := arithEq_of_rows h (row := 350) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_13
  simp only [← c22944, ← c22945, k2, ← c22946] at e_350_13
  have hr := e_350_13
  simp only [e_350_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 350 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 225)) := by
  have c22947 := (publicBatchWrapper86_copies717 a h).2.2.2.1
  have c22948 := (publicBatchWrapper86_copies717 a h).2.2.2.2.1
  have c22949 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.1
  have c22950 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.1
  have c22951 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.1
  have c22952 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_350_14 := arithEq_of_rows h (row := 350) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_14
  simp only [← c22947, ← c22948, ← c22949] at e_350_14
  have e_350_15 := arithEq_of_rows h (row := 350) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_15
  simp only [← c22950, ← c22951, k2, ← c22952] at e_350_15
  have hr := e_350_15
  simp only [e_350_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 350 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 226)) := by
  have c22953 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.1
  have c22954 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22955 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22956 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22957 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22958 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_350_16 := arithEq_of_rows h (row := 350) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_16
  simp only [← c22953, ← c22954, ← c22955] at e_350_16
  have e_350_17 := arithEq_of_rows h (row := 350) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_17
  simp only [← c22956, ← c22957, k2, ← c22958] at e_350_17
  have hr := e_350_17
  simp only [e_350_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 350 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 227)) := by
  have c22959 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22960 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22961 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22962 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22963 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22964 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_350_18 := arithEq_of_rows h (row := 350) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_18
  simp only [← c22959, ← c22960, ← c22961] at e_350_18
  have e_350_19 := arithEq_of_rows h (row := 350) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_350_19
  simp only [← c22962, ← c22963, k2, ← c22964] at e_350_19
  have hr := e_350_19
  simp only [e_350_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 351 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 228)) := by
  have c22965 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22966 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22967 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22968 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22969 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22970 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_351_0 := arithEq_of_rows h (row := 351) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_0
  simp only [← c22965, ← c22966, ← c22967] at e_351_0
  have e_351_1 := arithEq_of_rows h (row := 351) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_1
  simp only [← c22968, ← c22969, k2, ← c22970] at e_351_1
  have hr := e_351_1
  simp only [e_351_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 351 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 229)) := by
  have c22971 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22972 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22973 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22974 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22975 := (publicBatchWrapper86_copies717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c22976 := (publicBatchWrapper86_copies718 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_351_2 := arithEq_of_rows h (row := 351) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_2
  simp only [← c22971, ← c22972, ← c22973] at e_351_2
  have e_351_3 := arithEq_of_rows h (row := 351) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_3
  simp only [← c22974, ← c22975, k2, ← c22976] at e_351_3
  have hr := e_351_3
  simp only [e_351_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 351 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 230)) := by
  have c22977 := (publicBatchWrapper86_copies718 a h).2.1
  have c22978 := (publicBatchWrapper86_copies718 a h).2.2.1
  have c22979 := (publicBatchWrapper86_copies718 a h).2.2.2.1
  have c22980 := (publicBatchWrapper86_copies718 a h).2.2.2.2.1
  have c22981 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.1
  have c22982 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_351_4 := arithEq_of_rows h (row := 351) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_4
  simp only [← c22977, ← c22978, ← c22979] at e_351_4
  have e_351_5 := arithEq_of_rows h (row := 351) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_5
  simp only [← c22980, ← c22981, k2, ← c22982] at e_351_5
  have hr := e_351_5
  simp only [e_351_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 351 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 231)) := by
  have c22983 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.1
  have c22984 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.1
  have c22985 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.1
  have c22986 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.1
  have c22987 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c22988 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_351_6 := arithEq_of_rows h (row := 351) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_6
  simp only [← c22983, ← c22984, ← c22985] at e_351_6
  have e_351_7 := arithEq_of_rows h (row := 351) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_7
  simp only [← c22986, ← c22987, k2, ← c22988] at e_351_7
  have hr := e_351_7
  simp only [e_351_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 351 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 232)) := by
  have c22989 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22990 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22991 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22992 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22993 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22994 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_351_8 := arithEq_of_rows h (row := 351) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_8
  simp only [← c22989, ← c22990, ← c22991] at e_351_8
  have e_351_9 := arithEq_of_rows h (row := 351) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_9
  simp only [← c22992, ← c22993, k2, ← c22994] at e_351_9
  have hr := e_351_9
  simp only [e_351_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 351 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 233)) := by
  have c22995 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22996 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22997 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22998 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c22999 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23000 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_351_10 := arithEq_of_rows h (row := 351) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_10
  simp only [← c22995, ← c22996, ← c22997] at e_351_10
  have e_351_11 := arithEq_of_rows h (row := 351) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_11
  simp only [← c22998, ← c22999, k2, ← c23000] at e_351_11
  have hr := e_351_11
  simp only [e_351_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 351 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 234)) := by
  have c23001 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23002 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23003 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23004 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23005 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23006 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_351_12 := arithEq_of_rows h (row := 351) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_12
  simp only [← c23001, ← c23002, ← c23003] at e_351_12
  have e_351_13 := arithEq_of_rows h (row := 351) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_13
  simp only [← c23004, ← c23005, k2, ← c23006] at e_351_13
  have hr := e_351_13
  simp only [e_351_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 351 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 235)) := by
  have c23007 := (publicBatchWrapper86_copies718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23008 := (publicBatchWrapper86_copies719 a h).1
  have c23009 := (publicBatchWrapper86_copies719 a h).2.1
  have c23010 := (publicBatchWrapper86_copies719 a h).2.2.1
  have c23011 := (publicBatchWrapper86_copies719 a h).2.2.2.1
  have c23012 := (publicBatchWrapper86_copies719 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_351_14 := arithEq_of_rows h (row := 351) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_14
  simp only [← c23007, ← c23008, ← c23009] at e_351_14
  have e_351_15 := arithEq_of_rows h (row := 351) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_15
  simp only [← c23010, ← c23011, k2, ← c23012] at e_351_15
  have hr := e_351_15
  simp only [e_351_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 351 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 236)) := by
  have c23013 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.1
  have c23014 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.1
  have c23015 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.1
  have c23016 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.1
  have c23017 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.1
  have c23018 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_351_16 := arithEq_of_rows h (row := 351) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_16
  simp only [← c23013, ← c23014, ← c23015] at e_351_16
  have e_351_17 := arithEq_of_rows h (row := 351) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_17
  simp only [← c23016, ← c23017, k2, ← c23018] at e_351_17
  have hr := e_351_17
  simp only [e_351_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 351 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 237)) := by
  have c23019 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23020 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23021 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23022 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23023 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23024 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_351_18 := arithEq_of_rows h (row := 351) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_18
  simp only [← c23019, ← c23020, ← c23021] at e_351_18
  have e_351_19 := arithEq_of_rows h (row := 351) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_351_19
  simp only [← c23022, ← c23023, k2, ← c23024] at e_351_19
  have hr := e_351_19
  simp only [e_351_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 352 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 238)) := by
  have c23025 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23026 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23027 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23028 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23029 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23030 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_352_0 := arithEq_of_rows h (row := 352) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_0
  simp only [← c23025, ← c23026, ← c23027] at e_352_0
  have e_352_1 := arithEq_of_rows h (row := 352) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_1
  simp only [← c23028, ← c23029, k2, ← c23030] at e_352_1
  have hr := e_352_1
  simp only [e_352_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 352 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 239)) := by
  have c23031 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23032 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23033 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23034 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23035 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23036 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_352_2 := arithEq_of_rows h (row := 352) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_2
  simp only [← c23031, ← c23032, ← c23033] at e_352_2
  have e_352_3 := arithEq_of_rows h (row := 352) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_3
  simp only [← c23034, ← c23035, k2, ← c23036] at e_352_3
  have hr := e_352_3
  simp only [e_352_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 352 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 240)) := by
  have c23037 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23038 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23039 := (publicBatchWrapper86_copies719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23040 := (publicBatchWrapper86_copies720 a h).1
  have c23041 := (publicBatchWrapper86_copies720 a h).2.1
  have c23042 := (publicBatchWrapper86_copies720 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_352_4 := arithEq_of_rows h (row := 352) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_4
  simp only [← c23037, ← c23038, ← c23039] at e_352_4
  have e_352_5 := arithEq_of_rows h (row := 352) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_5
  simp only [← c23040, ← c23041, k2, ← c23042] at e_352_5
  have hr := e_352_5
  simp only [e_352_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 352 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 241)) := by
  have c23043 := (publicBatchWrapper86_copies720 a h).2.2.2.1
  have c23044 := (publicBatchWrapper86_copies720 a h).2.2.2.2.1
  have c23045 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.1
  have c23046 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.1
  have c23047 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.1
  have c23048 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_352_6 := arithEq_of_rows h (row := 352) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_6
  simp only [← c23043, ← c23044, ← c23045] at e_352_6
  have e_352_7 := arithEq_of_rows h (row := 352) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_7
  simp only [← c23046, ← c23047, k2, ← c23048] at e_352_7
  have hr := e_352_7
  simp only [e_352_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 352 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 242)) := by
  have c23049 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.1
  have c23050 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23051 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23052 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23053 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23054 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_352_8 := arithEq_of_rows h (row := 352) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_8
  simp only [← c23049, ← c23050, ← c23051] at e_352_8
  have e_352_9 := arithEq_of_rows h (row := 352) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_9
  simp only [← c23052, ← c23053, k2, ← c23054] at e_352_9
  have hr := e_352_9
  simp only [e_352_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 352 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 243)) := by
  have c23055 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23056 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23057 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23058 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23059 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23060 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_352_10 := arithEq_of_rows h (row := 352) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_10
  simp only [← c23055, ← c23056, ← c23057] at e_352_10
  have e_352_11 := arithEq_of_rows h (row := 352) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_11
  simp only [← c23058, ← c23059, k2, ← c23060] at e_352_11
  have hr := e_352_11
  simp only [e_352_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 352 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 244)) := by
  have c23061 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23062 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23063 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23064 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23065 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23066 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_352_12 := arithEq_of_rows h (row := 352) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_12
  simp only [← c23061, ← c23062, ← c23063] at e_352_12
  have e_352_13 := arithEq_of_rows h (row := 352) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_13
  simp only [← c23064, ← c23065, k2, ← c23066] at e_352_13
  have hr := e_352_13
  simp only [e_352_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 352 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 245)) := by
  have c23067 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23068 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23069 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23070 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23071 := (publicBatchWrapper86_copies720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23072 := (publicBatchWrapper86_copies721 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_352_14 := arithEq_of_rows h (row := 352) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_14
  simp only [← c23067, ← c23068, ← c23069] at e_352_14
  have e_352_15 := arithEq_of_rows h (row := 352) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_15
  simp only [← c23070, ← c23071, k2, ← c23072] at e_352_15
  have hr := e_352_15
  simp only [e_352_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 352 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 246)) := by
  have c23073 := (publicBatchWrapper86_copies721 a h).2.1
  have c23074 := (publicBatchWrapper86_copies721 a h).2.2.1
  have c23075 := (publicBatchWrapper86_copies721 a h).2.2.2.1
  have c23076 := (publicBatchWrapper86_copies721 a h).2.2.2.2.1
  have c23077 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.1
  have c23078 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_352_16 := arithEq_of_rows h (row := 352) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_16
  simp only [← c23073, ← c23074, ← c23075] at e_352_16
  have e_352_17 := arithEq_of_rows h (row := 352) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_17
  simp only [← c23076, ← c23077, k2, ← c23078] at e_352_17
  have hr := e_352_17
  simp only [e_352_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 352 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 247)) := by
  have c23079 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.1
  have c23080 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.1
  have c23081 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.1
  have c23082 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23083 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23084 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_352_18 := arithEq_of_rows h (row := 352) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_18
  simp only [← c23079, ← c23080, ← c23081] at e_352_18
  have e_352_19 := arithEq_of_rows h (row := 352) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_352_19
  simp only [← c23082, ← c23083, k2, ← c23084] at e_352_19
  have hr := e_352_19
  simp only [e_352_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 353 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 248)) := by
  have c23085 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23086 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23087 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23088 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23089 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23090 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_353_0 := arithEq_of_rows h (row := 353) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_0
  simp only [← c23085, ← c23086, ← c23087] at e_353_0
  have e_353_1 := arithEq_of_rows h (row := 353) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_1
  simp only [← c23088, ← c23089, k2, ← c23090] at e_353_1
  have hr := e_353_1
  simp only [e_353_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 353 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 249)) := by
  have c23091 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23092 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23093 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23094 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23095 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23096 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_353_2 := arithEq_of_rows h (row := 353) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_2
  simp only [← c23091, ← c23092, ← c23093] at e_353_2
  have e_353_3 := arithEq_of_rows h (row := 353) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_3
  simp only [← c23094, ← c23095, k2, ← c23096] at e_353_3
  have hr := e_353_3
  simp only [e_353_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 353 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 250)) := by
  have c23097 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23098 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23099 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23100 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23101 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23102 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_353_4 := arithEq_of_rows h (row := 353) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_4
  simp only [← c23097, ← c23098, ← c23099] at e_353_4
  have e_353_5 := arithEq_of_rows h (row := 353) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_5
  simp only [← c23100, ← c23101, k2, ← c23102] at e_353_5
  have hr := e_353_5
  simp only [e_353_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 353 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 251)) := by
  have c23103 := (publicBatchWrapper86_copies721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23104 := (publicBatchWrapper86_copies722 a h).1
  have c23105 := (publicBatchWrapper86_copies722 a h).2.1
  have c23106 := (publicBatchWrapper86_copies722 a h).2.2.1
  have c23107 := (publicBatchWrapper86_copies722 a h).2.2.2.1
  have c23108 := (publicBatchWrapper86_copies722 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_353_6 := arithEq_of_rows h (row := 353) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_6
  simp only [← c23103, ← c23104, ← c23105] at e_353_6
  have e_353_7 := arithEq_of_rows h (row := 353) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_7
  simp only [← c23106, ← c23107, k2, ← c23108] at e_353_7
  have hr := e_353_7
  simp only [e_353_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 353 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 252)) := by
  have c23109 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.1
  have c23110 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.1
  have c23111 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.1
  have c23112 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.1
  have c23113 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.1
  have c23114 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_353_8 := arithEq_of_rows h (row := 353) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_8
  simp only [← c23109, ← c23110, ← c23111] at e_353_8
  have e_353_9 := arithEq_of_rows h (row := 353) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_9
  simp only [← c23112, ← c23113, k2, ← c23114] at e_353_9
  have hr := e_353_9
  simp only [e_353_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 353 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 253)) := by
  have c23115 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23116 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23117 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23118 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23119 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23120 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_353_10 := arithEq_of_rows h (row := 353) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_10
  simp only [← c23115, ← c23116, ← c23117] at e_353_10
  have e_353_11 := arithEq_of_rows h (row := 353) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_11
  simp only [← c23118, ← c23119, k2, ← c23120] at e_353_11
  have hr := e_353_11
  simp only [e_353_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 353 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 254)) := by
  have c23121 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23122 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23123 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23124 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23125 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23126 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_353_12 := arithEq_of_rows h (row := 353) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_12
  simp only [← c23121, ← c23122, ← c23123] at e_353_12
  have e_353_13 := arithEq_of_rows h (row := 353) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_13
  simp only [← c23124, ← c23125, k2, ← c23126] at e_353_13
  have hr := e_353_13
  simp only [e_353_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 353 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 255)) := by
  have c23127 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23128 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23129 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23130 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23131 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23132 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_353_14 := arithEq_of_rows h (row := 353) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_14
  simp only [← c23127, ← c23128, ← c23129] at e_353_14
  have e_353_15 := arithEq_of_rows h (row := 353) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_15
  simp only [← c23130, ← c23131, k2, ← c23132] at e_353_15
  have hr := e_353_15
  simp only [e_353_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 353 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 256)) := by
  have c23133 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23134 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23135 := (publicBatchWrapper86_copies722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23136 := (publicBatchWrapper86_copies723 a h).1
  have c23137 := (publicBatchWrapper86_copies723 a h).2.1
  have c23138 := (publicBatchWrapper86_copies723 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_353_16 := arithEq_of_rows h (row := 353) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_16
  simp only [← c23133, ← c23134, ← c23135] at e_353_16
  have e_353_17 := arithEq_of_rows h (row := 353) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_17
  simp only [← c23136, ← c23137, k2, ← c23138] at e_353_17
  have hr := e_353_17
  simp only [e_353_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 353 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 257)) := by
  have c23139 := (publicBatchWrapper86_copies723 a h).2.2.2.1
  have c23140 := (publicBatchWrapper86_copies723 a h).2.2.2.2.1
  have c23141 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.1
  have c23142 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.1
  have c23143 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.1
  have c23144 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_353_18 := arithEq_of_rows h (row := 353) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_18
  simp only [← c23139, ← c23140, ← c23141] at e_353_18
  have e_353_19 := arithEq_of_rows h (row := 353) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_353_19
  simp only [← c23142, ← c23143, k2, ← c23144] at e_353_19
  have hr := e_353_19
  simp only [e_353_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 354 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 258)) := by
  have c23145 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.1
  have c23146 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23147 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23148 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23149 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23150 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_354_0 := arithEq_of_rows h (row := 354) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_0
  simp only [← c23145, ← c23146, ← c23147] at e_354_0
  have e_354_1 := arithEq_of_rows h (row := 354) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_1
  simp only [← c23148, ← c23149, k2, ← c23150] at e_354_1
  have hr := e_354_1
  simp only [e_354_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 354 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 259)) := by
  have c23151 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23152 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23153 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23154 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23155 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23156 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_354_2 := arithEq_of_rows h (row := 354) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_2
  simp only [← c23151, ← c23152, ← c23153] at e_354_2
  have e_354_3 := arithEq_of_rows h (row := 354) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_3
  simp only [← c23154, ← c23155, k2, ← c23156] at e_354_3
  have hr := e_354_3
  simp only [e_354_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 354 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 260)) := by
  have c23157 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23158 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23159 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23160 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23161 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23162 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_354_4 := arithEq_of_rows h (row := 354) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_4
  simp only [← c23157, ← c23158, ← c23159] at e_354_4
  have e_354_5 := arithEq_of_rows h (row := 354) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_5
  simp only [← c23160, ← c23161, k2, ← c23162] at e_354_5
  have hr := e_354_5
  simp only [e_354_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 354 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 261)) := by
  have c23163 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23164 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23165 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23166 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23167 := (publicBatchWrapper86_copies723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23168 := (publicBatchWrapper86_copies724 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_354_6 := arithEq_of_rows h (row := 354) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_6
  simp only [← c23163, ← c23164, ← c23165] at e_354_6
  have e_354_7 := arithEq_of_rows h (row := 354) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_7
  simp only [← c23166, ← c23167, k2, ← c23168] at e_354_7
  have hr := e_354_7
  simp only [e_354_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 354 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 262)) := by
  have c23169 := (publicBatchWrapper86_copies724 a h).2.1
  have c23170 := (publicBatchWrapper86_copies724 a h).2.2.1
  have c23171 := (publicBatchWrapper86_copies724 a h).2.2.2.1
  have c23172 := (publicBatchWrapper86_copies724 a h).2.2.2.2.1
  have c23173 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.1
  have c23174 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_354_8 := arithEq_of_rows h (row := 354) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_8
  simp only [← c23169, ← c23170, ← c23171] at e_354_8
  have e_354_9 := arithEq_of_rows h (row := 354) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_9
  simp only [← c23172, ← c23173, k2, ← c23174] at e_354_9
  have hr := e_354_9
  simp only [e_354_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 354 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 263)) := by
  have c23175 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.1
  have c23176 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.1
  have c23177 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.1
  have c23178 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23179 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23180 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_354_10 := arithEq_of_rows h (row := 354) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_10
  simp only [← c23175, ← c23176, ← c23177] at e_354_10
  have e_354_11 := arithEq_of_rows h (row := 354) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_11
  simp only [← c23178, ← c23179, k2, ← c23180] at e_354_11
  have hr := e_354_11
  simp only [e_354_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 354 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 264)) := by
  have c23181 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23182 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23183 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23184 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23185 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23186 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_354_12 := arithEq_of_rows h (row := 354) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_12
  simp only [← c23181, ← c23182, ← c23183] at e_354_12
  have e_354_13 := arithEq_of_rows h (row := 354) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_13
  simp only [← c23184, ← c23185, k2, ← c23186] at e_354_13
  have hr := e_354_13
  simp only [e_354_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 354 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 265)) := by
  have c23187 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23188 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23189 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23190 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23191 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23192 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_354_14 := arithEq_of_rows h (row := 354) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_14
  simp only [← c23187, ← c23188, ← c23189] at e_354_14
  have e_354_15 := arithEq_of_rows h (row := 354) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_15
  simp only [← c23190, ← c23191, k2, ← c23192] at e_354_15
  have hr := e_354_15
  simp only [e_354_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 354 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 266)) := by
  have c23193 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23194 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23195 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23196 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23197 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23198 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_354_16 := arithEq_of_rows h (row := 354) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_16
  simp only [← c23193, ← c23194, ← c23195] at e_354_16
  have e_354_17 := arithEq_of_rows h (row := 354) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_17
  simp only [← c23196, ← c23197, k2, ← c23198] at e_354_17
  have hr := e_354_17
  simp only [e_354_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 354 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 267)) := by
  have c23199 := (publicBatchWrapper86_copies724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23200 := (publicBatchWrapper86_copies725 a h).1
  have c23201 := (publicBatchWrapper86_copies725 a h).2.1
  have c23202 := (publicBatchWrapper86_copies725 a h).2.2.1
  have c23203 := (publicBatchWrapper86_copies725 a h).2.2.2.1
  have c23204 := (publicBatchWrapper86_copies725 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_354_18 := arithEq_of_rows h (row := 354) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_18
  simp only [← c23199, ← c23200, ← c23201] at e_354_18
  have e_354_19 := arithEq_of_rows h (row := 354) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_354_19
  simp only [← c23202, ← c23203, k2, ← c23204] at e_354_19
  have hr := e_354_19
  simp only [e_354_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 355 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 268)) := by
  have c23205 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.1
  have c23206 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.1
  have c23207 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.1
  have c23208 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.1
  have c23209 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.1
  have c23210 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_355_0 := arithEq_of_rows h (row := 355) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_0
  simp only [← c23205, ← c23206, ← c23207] at e_355_0
  have e_355_1 := arithEq_of_rows h (row := 355) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_1
  simp only [← c23208, ← c23209, k2, ← c23210] at e_355_1
  have hr := e_355_1
  simp only [e_355_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 355 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 269)) := by
  have c23211 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23212 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23213 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23214 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23215 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23216 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_355_2 := arithEq_of_rows h (row := 355) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_2
  simp only [← c23211, ← c23212, ← c23213] at e_355_2
  have e_355_3 := arithEq_of_rows h (row := 355) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_3
  simp only [← c23214, ← c23215, k2, ← c23216] at e_355_3
  have hr := e_355_3
  simp only [e_355_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 355 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 270)) := by
  have c23217 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23218 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23219 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23220 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23221 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23222 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_355_4 := arithEq_of_rows h (row := 355) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_4
  simp only [← c23217, ← c23218, ← c23219] at e_355_4
  have e_355_5 := arithEq_of_rows h (row := 355) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_5
  simp only [← c23220, ← c23221, k2, ← c23222] at e_355_5
  have hr := e_355_5
  simp only [e_355_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 355 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 271)) := by
  have c23223 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23224 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23225 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23226 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23227 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23228 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_355_6 := arithEq_of_rows h (row := 355) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_6
  simp only [← c23223, ← c23224, ← c23225] at e_355_6
  have e_355_7 := arithEq_of_rows h (row := 355) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_7
  simp only [← c23226, ← c23227, k2, ← c23228] at e_355_7
  have hr := e_355_7
  simp only [e_355_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 355 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 272)) := by
  have c23229 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23230 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23231 := (publicBatchWrapper86_copies725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23232 := (publicBatchWrapper86_copies726 a h).1
  have c23233 := (publicBatchWrapper86_copies726 a h).2.1
  have c23234 := (publicBatchWrapper86_copies726 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_355_8 := arithEq_of_rows h (row := 355) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_8
  simp only [← c23229, ← c23230, ← c23231] at e_355_8
  have e_355_9 := arithEq_of_rows h (row := 355) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_9
  simp only [← c23232, ← c23233, k2, ← c23234] at e_355_9
  have hr := e_355_9
  simp only [e_355_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 355 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 273)) := by
  have c23235 := (publicBatchWrapper86_copies726 a h).2.2.2.1
  have c23236 := (publicBatchWrapper86_copies726 a h).2.2.2.2.1
  have c23237 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.1
  have c23238 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.1
  have c23239 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.1
  have c23240 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_355_10 := arithEq_of_rows h (row := 355) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_10
  simp only [← c23235, ← c23236, ← c23237] at e_355_10
  have e_355_11 := arithEq_of_rows h (row := 355) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_11
  simp only [← c23238, ← c23239, k2, ← c23240] at e_355_11
  have hr := e_355_11
  simp only [e_355_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 355 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 274)) := by
  have c23241 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.1
  have c23242 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23243 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23244 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23245 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23246 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_355_12 := arithEq_of_rows h (row := 355) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_12
  simp only [← c23241, ← c23242, ← c23243] at e_355_12
  have e_355_13 := arithEq_of_rows h (row := 355) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_13
  simp only [← c23244, ← c23245, k2, ← c23246] at e_355_13
  have hr := e_355_13
  simp only [e_355_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 355 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 275)) := by
  have c23247 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23248 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23249 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23250 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23251 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23252 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_355_14 := arithEq_of_rows h (row := 355) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_14
  simp only [← c23247, ← c23248, ← c23249] at e_355_14
  have e_355_15 := arithEq_of_rows h (row := 355) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_15
  simp only [← c23250, ← c23251, k2, ← c23252] at e_355_15
  have hr := e_355_15
  simp only [e_355_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 355 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 276)) := by
  have c23253 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23254 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23255 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23256 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23257 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23258 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_355_16 := arithEq_of_rows h (row := 355) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_16
  simp only [← c23253, ← c23254, ← c23255] at e_355_16
  have e_355_17 := arithEq_of_rows h (row := 355) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_17
  simp only [← c23256, ← c23257, k2, ← c23258] at e_355_17
  have hr := e_355_17
  simp only [e_355_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 355 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 277)) := by
  have c23259 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23260 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23261 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23262 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23263 := (publicBatchWrapper86_copies726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23264 := (publicBatchWrapper86_copies727 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_355_18 := arithEq_of_rows h (row := 355) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_18
  simp only [← c23259, ← c23260, ← c23261] at e_355_18
  have e_355_19 := arithEq_of_rows h (row := 355) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_355_19
  simp only [← c23262, ← c23263, k2, ← c23264] at e_355_19
  have hr := e_355_19
  simp only [e_355_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 356 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 278)) := by
  have c23265 := (publicBatchWrapper86_copies727 a h).2.1
  have c23266 := (publicBatchWrapper86_copies727 a h).2.2.1
  have c23267 := (publicBatchWrapper86_copies727 a h).2.2.2.1
  have c23268 := (publicBatchWrapper86_copies727 a h).2.2.2.2.1
  have c23269 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.1
  have c23270 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_356_0 := arithEq_of_rows h (row := 356) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_0
  simp only [← c23265, ← c23266, ← c23267] at e_356_0
  have e_356_1 := arithEq_of_rows h (row := 356) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_1
  simp only [← c23268, ← c23269, k2, ← c23270] at e_356_1
  have hr := e_356_1
  simp only [e_356_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 356 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 279)) := by
  have c23271 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.1
  have c23272 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.1
  have c23273 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.1
  have c23274 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23275 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23276 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_356_2 := arithEq_of_rows h (row := 356) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_2
  simp only [← c23271, ← c23272, ← c23273] at e_356_2
  have e_356_3 := arithEq_of_rows h (row := 356) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_3
  simp only [← c23274, ← c23275, k2, ← c23276] at e_356_3
  have hr := e_356_3
  simp only [e_356_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 356 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 280)) := by
  have c23277 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23278 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23279 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23280 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23281 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23282 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_356_4 := arithEq_of_rows h (row := 356) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_4
  simp only [← c23277, ← c23278, ← c23279] at e_356_4
  have e_356_5 := arithEq_of_rows h (row := 356) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_5
  simp only [← c23280, ← c23281, k2, ← c23282] at e_356_5
  have hr := e_356_5
  simp only [e_356_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 356 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 281)) := by
  have c23283 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23284 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23285 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23286 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23287 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23288 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_356_6 := arithEq_of_rows h (row := 356) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_6
  simp only [← c23283, ← c23284, ← c23285] at e_356_6
  have e_356_7 := arithEq_of_rows h (row := 356) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_7
  simp only [← c23286, ← c23287, k2, ← c23288] at e_356_7
  have hr := e_356_7
  simp only [e_356_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 356 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 282)) := by
  have c23289 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23290 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23291 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23292 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23293 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23294 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_356_8 := arithEq_of_rows h (row := 356) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_8
  simp only [← c23289, ← c23290, ← c23291] at e_356_8
  have e_356_9 := arithEq_of_rows h (row := 356) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_9
  simp only [← c23292, ← c23293, k2, ← c23294] at e_356_9
  have hr := e_356_9
  simp only [e_356_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 356 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 283)) := by
  have c23295 := (publicBatchWrapper86_copies727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23296 := (publicBatchWrapper86_copies728 a h).1
  have c23297 := (publicBatchWrapper86_copies728 a h).2.1
  have c23298 := (publicBatchWrapper86_copies728 a h).2.2.1
  have c23299 := (publicBatchWrapper86_copies728 a h).2.2.2.1
  have c23300 := (publicBatchWrapper86_copies728 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_356_10 := arithEq_of_rows h (row := 356) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_10
  simp only [← c23295, ← c23296, ← c23297] at e_356_10
  have e_356_11 := arithEq_of_rows h (row := 356) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_11
  simp only [← c23298, ← c23299, k2, ← c23300] at e_356_11
  have hr := e_356_11
  simp only [e_356_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 356 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 284)) := by
  have c23301 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.1
  have c23302 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.1
  have c23303 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.1
  have c23304 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.1
  have c23305 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.1
  have c23306 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_356_12 := arithEq_of_rows h (row := 356) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_12
  simp only [← c23301, ← c23302, ← c23303] at e_356_12
  have e_356_13 := arithEq_of_rows h (row := 356) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_13
  simp only [← c23304, ← c23305, k2, ← c23306] at e_356_13
  have hr := e_356_13
  simp only [e_356_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 356 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 285)) := by
  have c23307 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23308 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23309 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23310 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23311 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23312 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_356_14 := arithEq_of_rows h (row := 356) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_14
  simp only [← c23307, ← c23308, ← c23309] at e_356_14
  have e_356_15 := arithEq_of_rows h (row := 356) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_15
  simp only [← c23310, ← c23311, k2, ← c23312] at e_356_15
  have hr := e_356_15
  simp only [e_356_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 356 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 286)) := by
  have c23313 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23314 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23315 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23316 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23317 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23318 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_356_16 := arithEq_of_rows h (row := 356) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_16
  simp only [← c23313, ← c23314, ← c23315] at e_356_16
  have e_356_17 := arithEq_of_rows h (row := 356) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_17
  simp only [← c23316, ← c23317, k2, ← c23318] at e_356_17
  have hr := e_356_17
  simp only [e_356_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 356 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 287)) := by
  have c23319 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23320 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23321 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23322 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23323 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23324 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_356_18 := arithEq_of_rows h (row := 356) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_18
  simp only [← c23319, ← c23320, ← c23321] at e_356_18
  have e_356_19 := arithEq_of_rows h (row := 356) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_356_19
  simp only [← c23322, ← c23323, k2, ← c23324] at e_356_19
  have hr := e_356_19
  simp only [e_356_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 357 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 288)) := by
  have c23325 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23326 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23327 := (publicBatchWrapper86_copies728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23328 := (publicBatchWrapper86_copies729 a h).1
  have c23329 := (publicBatchWrapper86_copies729 a h).2.1
  have c23330 := (publicBatchWrapper86_copies729 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_357_0 := arithEq_of_rows h (row := 357) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_0
  simp only [← c23325, ← c23326, ← c23327] at e_357_0
  have e_357_1 := arithEq_of_rows h (row := 357) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_1
  simp only [← c23328, ← c23329, k2, ← c23330] at e_357_1
  have hr := e_357_1
  simp only [e_357_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 357 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 289)) := by
  have c23331 := (publicBatchWrapper86_copies729 a h).2.2.2.1
  have c23332 := (publicBatchWrapper86_copies729 a h).2.2.2.2.1
  have c23333 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.1
  have c23334 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.1
  have c23335 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.1
  have c23336 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_357_2 := arithEq_of_rows h (row := 357) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_2
  simp only [← c23331, ← c23332, ← c23333] at e_357_2
  have e_357_3 := arithEq_of_rows h (row := 357) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_3
  simp only [← c23334, ← c23335, k2, ← c23336] at e_357_3
  have hr := e_357_3
  simp only [e_357_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 357 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 290)) := by
  have c23337 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.1
  have c23338 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23339 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23340 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23341 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23342 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_357_4 := arithEq_of_rows h (row := 357) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_4
  simp only [← c23337, ← c23338, ← c23339] at e_357_4
  have e_357_5 := arithEq_of_rows h (row := 357) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_5
  simp only [← c23340, ← c23341, k2, ← c23342] at e_357_5
  have hr := e_357_5
  simp only [e_357_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 357 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 291)) := by
  have c23343 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23344 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23345 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23346 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23347 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23348 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_357_6 := arithEq_of_rows h (row := 357) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_6
  simp only [← c23343, ← c23344, ← c23345] at e_357_6
  have e_357_7 := arithEq_of_rows h (row := 357) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_7
  simp only [← c23346, ← c23347, k2, ← c23348] at e_357_7
  have hr := e_357_7
  simp only [e_357_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 357 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 292)) := by
  have c23349 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23350 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23351 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23352 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23353 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23354 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_357_8 := arithEq_of_rows h (row := 357) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_8
  simp only [← c23349, ← c23350, ← c23351] at e_357_8
  have e_357_9 := arithEq_of_rows h (row := 357) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_9
  simp only [← c23352, ← c23353, k2, ← c23354] at e_357_9
  have hr := e_357_9
  simp only [e_357_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 357 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 293)) := by
  have c23355 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23356 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23357 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23358 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23359 := (publicBatchWrapper86_copies729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23360 := (publicBatchWrapper86_copies730 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_357_10 := arithEq_of_rows h (row := 357) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_10
  simp only [← c23355, ← c23356, ← c23357] at e_357_10
  have e_357_11 := arithEq_of_rows h (row := 357) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_11
  simp only [← c23358, ← c23359, k2, ← c23360] at e_357_11
  have hr := e_357_11
  simp only [e_357_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 357 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 294)) := by
  have c23361 := (publicBatchWrapper86_copies730 a h).2.1
  have c23362 := (publicBatchWrapper86_copies730 a h).2.2.1
  have c23363 := (publicBatchWrapper86_copies730 a h).2.2.2.1
  have c23364 := (publicBatchWrapper86_copies730 a h).2.2.2.2.1
  have c23365 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.1
  have c23366 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_357_12 := arithEq_of_rows h (row := 357) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_12
  simp only [← c23361, ← c23362, ← c23363] at e_357_12
  have e_357_13 := arithEq_of_rows h (row := 357) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_13
  simp only [← c23364, ← c23365, k2, ← c23366] at e_357_13
  have hr := e_357_13
  simp only [e_357_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 357 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 295)) := by
  have c23367 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.1
  have c23368 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.1
  have c23369 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.1
  have c23370 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23371 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23372 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_357_14 := arithEq_of_rows h (row := 357) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_14
  simp only [← c23367, ← c23368, ← c23369] at e_357_14
  have e_357_15 := arithEq_of_rows h (row := 357) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_15
  simp only [← c23370, ← c23371, k2, ← c23372] at e_357_15
  have hr := e_357_15
  simp only [e_357_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 357 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 296)) := by
  have c23373 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23374 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23375 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23376 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23377 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23378 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_357_16 := arithEq_of_rows h (row := 357) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_16
  simp only [← c23373, ← c23374, ← c23375] at e_357_16
  have e_357_17 := arithEq_of_rows h (row := 357) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_17
  simp only [← c23376, ← c23377, k2, ← c23378] at e_357_17
  have hr := e_357_17
  simp only [e_357_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 357 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 297)) := by
  have c23379 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23380 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23381 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23382 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23383 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23384 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_357_18 := arithEq_of_rows h (row := 357) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_18
  simp only [← c23379, ← c23380, ← c23381] at e_357_18
  have e_357_19 := arithEq_of_rows h (row := 357) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_357_19
  simp only [← c23382, ← c23383, k2, ← c23384] at e_357_19
  have hr := e_357_19
  simp only [e_357_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 358 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 298)) := by
  have c23385 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23386 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23387 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23388 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23389 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23390 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_358_0 := arithEq_of_rows h (row := 358) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_0
  simp only [← c23385, ← c23386, ← c23387] at e_358_0
  have e_358_1 := arithEq_of_rows h (row := 358) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_1
  simp only [← c23388, ← c23389, k2, ← c23390] at e_358_1
  have hr := e_358_1
  simp only [e_358_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 358 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 299)) := by
  have c23391 := (publicBatchWrapper86_copies730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23392 := (publicBatchWrapper86_copies731 a h).1
  have c23393 := (publicBatchWrapper86_copies731 a h).2.1
  have c23394 := (publicBatchWrapper86_copies731 a h).2.2.1
  have c23395 := (publicBatchWrapper86_copies731 a h).2.2.2.1
  have c23396 := (publicBatchWrapper86_copies731 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_358_2 := arithEq_of_rows h (row := 358) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_2
  simp only [← c23391, ← c23392, ← c23393] at e_358_2
  have e_358_3 := arithEq_of_rows h (row := 358) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_3
  simp only [← c23394, ← c23395, k2, ← c23396] at e_358_3
  have hr := e_358_3
  simp only [e_358_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 358 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 300)) := by
  have c23397 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.1
  have c23398 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.1
  have c23399 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.1
  have c23400 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.1
  have c23401 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.1
  have c23402 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_358_4 := arithEq_of_rows h (row := 358) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_4
  simp only [← c23397, ← c23398, ← c23399] at e_358_4
  have e_358_5 := arithEq_of_rows h (row := 358) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_5
  simp only [← c23400, ← c23401, k2, ← c23402] at e_358_5
  have hr := e_358_5
  simp only [e_358_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 358 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 301)) := by
  have c23403 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23404 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23405 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23406 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23407 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23408 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_358_6 := arithEq_of_rows h (row := 358) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_6
  simp only [← c23403, ← c23404, ← c23405] at e_358_6
  have e_358_7 := arithEq_of_rows h (row := 358) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_7
  simp only [← c23406, ← c23407, k2, ← c23408] at e_358_7
  have hr := e_358_7
  simp only [e_358_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 358 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 302)) := by
  have c23409 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23410 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23411 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23412 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23413 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23414 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_358_8 := arithEq_of_rows h (row := 358) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_8
  simp only [← c23409, ← c23410, ← c23411] at e_358_8
  have e_358_9 := arithEq_of_rows h (row := 358) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_9
  simp only [← c23412, ← c23413, k2, ← c23414] at e_358_9
  have hr := e_358_9
  simp only [e_358_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 358 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 303)) := by
  have c23415 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23416 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23417 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23418 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23419 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23420 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_358_10 := arithEq_of_rows h (row := 358) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_10
  simp only [← c23415, ← c23416, ← c23417] at e_358_10
  have e_358_11 := arithEq_of_rows h (row := 358) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_11
  simp only [← c23418, ← c23419, k2, ← c23420] at e_358_11
  have hr := e_358_11
  simp only [e_358_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 358 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 304)) := by
  have c23421 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23422 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23423 := (publicBatchWrapper86_copies731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23424 := (publicBatchWrapper86_copies732 a h).1
  have c23425 := (publicBatchWrapper86_copies732 a h).2.1
  have c23426 := (publicBatchWrapper86_copies732 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_358_12 := arithEq_of_rows h (row := 358) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_12
  simp only [← c23421, ← c23422, ← c23423] at e_358_12
  have e_358_13 := arithEq_of_rows h (row := 358) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_13
  simp only [← c23424, ← c23425, k2, ← c23426] at e_358_13
  have hr := e_358_13
  simp only [e_358_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 358 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 305)) := by
  have c23427 := (publicBatchWrapper86_copies732 a h).2.2.2.1
  have c23428 := (publicBatchWrapper86_copies732 a h).2.2.2.2.1
  have c23429 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.1
  have c23430 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.1
  have c23431 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.1
  have c23432 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_358_14 := arithEq_of_rows h (row := 358) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_14
  simp only [← c23427, ← c23428, ← c23429] at e_358_14
  have e_358_15 := arithEq_of_rows h (row := 358) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_15
  simp only [← c23430, ← c23431, k2, ← c23432] at e_358_15
  have hr := e_358_15
  simp only [e_358_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 358 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 306)) := by
  have c23433 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.1
  have c23434 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23435 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23436 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23437 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23438 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_358_16 := arithEq_of_rows h (row := 358) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_16
  simp only [← c23433, ← c23434, ← c23435] at e_358_16
  have e_358_17 := arithEq_of_rows h (row := 358) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_17
  simp only [← c23436, ← c23437, k2, ← c23438] at e_358_17
  have hr := e_358_17
  simp only [e_358_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 358 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 307)) := by
  have c23439 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23440 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23441 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23442 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23443 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23444 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_358_18 := arithEq_of_rows h (row := 358) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_18
  simp only [← c23439, ← c23440, ← c23441] at e_358_18
  have e_358_19 := arithEq_of_rows h (row := 358) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_358_19
  simp only [← c23442, ← c23443, k2, ← c23444] at e_358_19
  have hr := e_358_19
  simp only [e_358_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 359 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 308)) := by
  have c23445 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23446 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23447 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23448 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23449 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23450 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_359_0 := arithEq_of_rows h (row := 359) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_0
  simp only [← c23445, ← c23446, ← c23447] at e_359_0
  have e_359_1 := arithEq_of_rows h (row := 359) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_1
  simp only [← c23448, ← c23449, k2, ← c23450] at e_359_1
  have hr := e_359_1
  simp only [e_359_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 359 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 309)) := by
  have c23451 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23452 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23453 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23454 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23455 := (publicBatchWrapper86_copies732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23456 := (publicBatchWrapper86_copies733 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_359_2 := arithEq_of_rows h (row := 359) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_2
  simp only [← c23451, ← c23452, ← c23453] at e_359_2
  have e_359_3 := arithEq_of_rows h (row := 359) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_3
  simp only [← c23454, ← c23455, k2, ← c23456] at e_359_3
  have hr := e_359_3
  simp only [e_359_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 359 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 310)) := by
  have c23457 := (publicBatchWrapper86_copies733 a h).2.1
  have c23458 := (publicBatchWrapper86_copies733 a h).2.2.1
  have c23459 := (publicBatchWrapper86_copies733 a h).2.2.2.1
  have c23460 := (publicBatchWrapper86_copies733 a h).2.2.2.2.1
  have c23461 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.1
  have c23462 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_359_4 := arithEq_of_rows h (row := 359) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_4
  simp only [← c23457, ← c23458, ← c23459] at e_359_4
  have e_359_5 := arithEq_of_rows h (row := 359) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_5
  simp only [← c23460, ← c23461, k2, ← c23462] at e_359_5
  have hr := e_359_5
  simp only [e_359_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 359 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 311)) := by
  have c23463 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.1
  have c23464 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.1
  have c23465 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.1
  have c23466 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23467 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23468 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_359_6 := arithEq_of_rows h (row := 359) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_6
  simp only [← c23463, ← c23464, ← c23465] at e_359_6
  have e_359_7 := arithEq_of_rows h (row := 359) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_7
  simp only [← c23466, ← c23467, k2, ← c23468] at e_359_7
  have hr := e_359_7
  simp only [e_359_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 359 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 312)) := by
  have c23469 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23470 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23471 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23472 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23473 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23474 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_359_8 := arithEq_of_rows h (row := 359) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_8
  simp only [← c23469, ← c23470, ← c23471] at e_359_8
  have e_359_9 := arithEq_of_rows h (row := 359) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_9
  simp only [← c23472, ← c23473, k2, ← c23474] at e_359_9
  have hr := e_359_9
  simp only [e_359_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 359 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 313)) := by
  have c23475 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23476 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23477 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23478 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23479 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23480 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_359_10 := arithEq_of_rows h (row := 359) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_10
  simp only [← c23475, ← c23476, ← c23477] at e_359_10
  have e_359_11 := arithEq_of_rows h (row := 359) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_11
  simp only [← c23478, ← c23479, k2, ← c23480] at e_359_11
  have hr := e_359_11
  simp only [e_359_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 359 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 314)) := by
  have c23481 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23482 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23483 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23484 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23485 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23486 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_359_12 := arithEq_of_rows h (row := 359) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_12
  simp only [← c23481, ← c23482, ← c23483] at e_359_12
  have e_359_13 := arithEq_of_rows h (row := 359) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_13
  simp only [← c23484, ← c23485, k2, ← c23486] at e_359_13
  have hr := e_359_13
  simp only [e_359_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 359 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 315)) := by
  have c23487 := (publicBatchWrapper86_copies733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23488 := (publicBatchWrapper86_copies734 a h).1
  have c23489 := (publicBatchWrapper86_copies734 a h).2.1
  have c23490 := (publicBatchWrapper86_copies734 a h).2.2.1
  have c23491 := (publicBatchWrapper86_copies734 a h).2.2.2.1
  have c23492 := (publicBatchWrapper86_copies734 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_359_14 := arithEq_of_rows h (row := 359) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_14
  simp only [← c23487, ← c23488, ← c23489] at e_359_14
  have e_359_15 := arithEq_of_rows h (row := 359) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_15
  simp only [← c23490, ← c23491, k2, ← c23492] at e_359_15
  have hr := e_359_15
  simp only [e_359_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 359 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 316)) := by
  have c23493 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.1
  have c23494 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.1
  have c23495 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.1
  have c23496 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.1
  have c23497 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.1
  have c23498 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_359_16 := arithEq_of_rows h (row := 359) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_16
  simp only [← c23493, ← c23494, ← c23495] at e_359_16
  have e_359_17 := arithEq_of_rows h (row := 359) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_17
  simp only [← c23496, ← c23497, k2, ← c23498] at e_359_17
  have hr := e_359_17
  simp only [e_359_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 359 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 317)) := by
  have c23499 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23500 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23501 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23502 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23503 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23504 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_359_18 := arithEq_of_rows h (row := 359) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_18
  simp only [← c23499, ← c23500, ← c23501] at e_359_18
  have e_359_19 := arithEq_of_rows h (row := 359) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_359_19
  simp only [← c23502, ← c23503, k2, ← c23504] at e_359_19
  have hr := e_359_19
  simp only [e_359_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 360 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 318)) := by
  have c23505 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23506 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23507 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23508 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23509 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23510 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_360_0 := arithEq_of_rows h (row := 360) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_0
  simp only [← c23505, ← c23506, ← c23507] at e_360_0
  have e_360_1 := arithEq_of_rows h (row := 360) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_1
  simp only [← c23508, ← c23509, k2, ← c23510] at e_360_1
  have hr := e_360_1
  simp only [e_360_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 360 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 319)) := by
  have c23511 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23512 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23513 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23514 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23515 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23516 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_360_2 := arithEq_of_rows h (row := 360) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_2
  simp only [← c23511, ← c23512, ← c23513] at e_360_2
  have e_360_3 := arithEq_of_rows h (row := 360) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_3
  simp only [← c23514, ← c23515, k2, ← c23516] at e_360_3
  have hr := e_360_3
  simp only [e_360_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 360 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 320)) := by
  have c23517 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23518 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23519 := (publicBatchWrapper86_copies734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23520 := (publicBatchWrapper86_copies735 a h).1
  have c23521 := (publicBatchWrapper86_copies735 a h).2.1
  have c23522 := (publicBatchWrapper86_copies735 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_360_4 := arithEq_of_rows h (row := 360) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_4
  simp only [← c23517, ← c23518, ← c23519] at e_360_4
  have e_360_5 := arithEq_of_rows h (row := 360) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_5
  simp only [← c23520, ← c23521, k2, ← c23522] at e_360_5
  have hr := e_360_5
  simp only [e_360_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 360 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 321)) := by
  have c23523 := (publicBatchWrapper86_copies735 a h).2.2.2.1
  have c23524 := (publicBatchWrapper86_copies735 a h).2.2.2.2.1
  have c23525 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.1
  have c23526 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.1
  have c23527 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.1
  have c23528 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_360_6 := arithEq_of_rows h (row := 360) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_6
  simp only [← c23523, ← c23524, ← c23525] at e_360_6
  have e_360_7 := arithEq_of_rows h (row := 360) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_7
  simp only [← c23526, ← c23527, k2, ← c23528] at e_360_7
  have hr := e_360_7
  simp only [e_360_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 360 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 322)) := by
  have c23529 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.1
  have c23530 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23531 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23532 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23533 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23534 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_360_8 := arithEq_of_rows h (row := 360) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_8
  simp only [← c23529, ← c23530, ← c23531] at e_360_8
  have e_360_9 := arithEq_of_rows h (row := 360) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_9
  simp only [← c23532, ← c23533, k2, ← c23534] at e_360_9
  have hr := e_360_9
  simp only [e_360_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 360 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 323)) := by
  have c23535 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23536 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23537 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23538 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23539 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23540 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_360_10 := arithEq_of_rows h (row := 360) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_10
  simp only [← c23535, ← c23536, ← c23537] at e_360_10
  have e_360_11 := arithEq_of_rows h (row := 360) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_11
  simp only [← c23538, ← c23539, k2, ← c23540] at e_360_11
  have hr := e_360_11
  simp only [e_360_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 360 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 324)) := by
  have c23541 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23542 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23543 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23544 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23545 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23546 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_360_12 := arithEq_of_rows h (row := 360) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_12
  simp only [← c23541, ← c23542, ← c23543] at e_360_12
  have e_360_13 := arithEq_of_rows h (row := 360) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_13
  simp only [← c23544, ← c23545, k2, ← c23546] at e_360_13
  have hr := e_360_13
  simp only [e_360_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 360 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 325)) := by
  have c23547 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23548 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23549 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23550 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23551 := (publicBatchWrapper86_copies735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23552 := (publicBatchWrapper86_copies736 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_360_14 := arithEq_of_rows h (row := 360) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_14
  simp only [← c23547, ← c23548, ← c23549] at e_360_14
  have e_360_15 := arithEq_of_rows h (row := 360) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_15
  simp only [← c23550, ← c23551, k2, ← c23552] at e_360_15
  have hr := e_360_15
  simp only [e_360_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 360 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 326)) := by
  have c23553 := (publicBatchWrapper86_copies736 a h).2.1
  have c23554 := (publicBatchWrapper86_copies736 a h).2.2.1
  have c23555 := (publicBatchWrapper86_copies736 a h).2.2.2.1
  have c23556 := (publicBatchWrapper86_copies736 a h).2.2.2.2.1
  have c23557 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.1
  have c23558 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_360_16 := arithEq_of_rows h (row := 360) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_16
  simp only [← c23553, ← c23554, ← c23555] at e_360_16
  have e_360_17 := arithEq_of_rows h (row := 360) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_17
  simp only [← c23556, ← c23557, k2, ← c23558] at e_360_17
  have hr := e_360_17
  simp only [e_360_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 360 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 327)) := by
  have c23559 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.1
  have c23560 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.1
  have c23561 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.1
  have c23562 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23563 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23564 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_360_18 := arithEq_of_rows h (row := 360) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_18
  simp only [← c23559, ← c23560, ← c23561] at e_360_18
  have e_360_19 := arithEq_of_rows h (row := 360) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_360_19
  simp only [← c23562, ← c23563, k2, ← c23564] at e_360_19
  have hr := e_360_19
  simp only [e_360_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 361 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 328)) := by
  have c23565 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23566 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23567 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23568 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23569 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23570 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_361_0 := arithEq_of_rows h (row := 361) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_0
  simp only [← c23565, ← c23566, ← c23567] at e_361_0
  have e_361_1 := arithEq_of_rows h (row := 361) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_1
  simp only [← c23568, ← c23569, k2, ← c23570] at e_361_1
  have hr := e_361_1
  simp only [e_361_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 361 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 329)) := by
  have c23571 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23572 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23573 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23574 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23575 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23576 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_361_2 := arithEq_of_rows h (row := 361) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_2
  simp only [← c23571, ← c23572, ← c23573] at e_361_2
  have e_361_3 := arithEq_of_rows h (row := 361) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_3
  simp only [← c23574, ← c23575, k2, ← c23576] at e_361_3
  have hr := e_361_3
  simp only [e_361_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 361 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 330)) := by
  have c23577 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23578 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23579 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23580 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23581 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23582 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_361_4 := arithEq_of_rows h (row := 361) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_4
  simp only [← c23577, ← c23578, ← c23579] at e_361_4
  have e_361_5 := arithEq_of_rows h (row := 361) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_5
  simp only [← c23580, ← c23581, k2, ← c23582] at e_361_5
  have hr := e_361_5
  simp only [e_361_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 361 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 331)) := by
  have c23583 := (publicBatchWrapper86_copies736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23584 := (publicBatchWrapper86_copies737 a h).1
  have c23585 := (publicBatchWrapper86_copies737 a h).2.1
  have c23586 := (publicBatchWrapper86_copies737 a h).2.2.1
  have c23587 := (publicBatchWrapper86_copies737 a h).2.2.2.1
  have c23588 := (publicBatchWrapper86_copies737 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_361_6 := arithEq_of_rows h (row := 361) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_6
  simp only [← c23583, ← c23584, ← c23585] at e_361_6
  have e_361_7 := arithEq_of_rows h (row := 361) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_7
  simp only [← c23586, ← c23587, k2, ← c23588] at e_361_7
  have hr := e_361_7
  simp only [e_361_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 361 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 332)) := by
  have c23589 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.1
  have c23590 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.1
  have c23591 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.1
  have c23592 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.1
  have c23593 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.1
  have c23594 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_361_8 := arithEq_of_rows h (row := 361) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_8
  simp only [← c23589, ← c23590, ← c23591] at e_361_8
  have e_361_9 := arithEq_of_rows h (row := 361) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_9
  simp only [← c23592, ← c23593, k2, ← c23594] at e_361_9
  have hr := e_361_9
  simp only [e_361_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 361 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 333)) := by
  have c23595 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23596 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23597 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23598 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23599 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23600 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_361_10 := arithEq_of_rows h (row := 361) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_10
  simp only [← c23595, ← c23596, ← c23597] at e_361_10
  have e_361_11 := arithEq_of_rows h (row := 361) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_11
  simp only [← c23598, ← c23599, k2, ← c23600] at e_361_11
  have hr := e_361_11
  simp only [e_361_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 361 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 334)) := by
  have c23601 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23602 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23603 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23604 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23605 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23606 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_361_12 := arithEq_of_rows h (row := 361) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_12
  simp only [← c23601, ← c23602, ← c23603] at e_361_12
  have e_361_13 := arithEq_of_rows h (row := 361) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_13
  simp only [← c23604, ← c23605, k2, ← c23606] at e_361_13
  have hr := e_361_13
  simp only [e_361_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 361 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 335)) := by
  have c23607 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23608 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23609 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23610 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23611 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23612 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_361_14 := arithEq_of_rows h (row := 361) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_14
  simp only [← c23607, ← c23608, ← c23609] at e_361_14
  have e_361_15 := arithEq_of_rows h (row := 361) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_15
  simp only [← c23610, ← c23611, k2, ← c23612] at e_361_15
  have hr := e_361_15
  simp only [e_361_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 361 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 336)) := by
  have c23613 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23614 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23615 := (publicBatchWrapper86_copies737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23616 := (publicBatchWrapper86_copies738 a h).1
  have c23617 := (publicBatchWrapper86_copies738 a h).2.1
  have c23618 := (publicBatchWrapper86_copies738 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_361_16 := arithEq_of_rows h (row := 361) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_16
  simp only [← c23613, ← c23614, ← c23615] at e_361_16
  have e_361_17 := arithEq_of_rows h (row := 361) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_17
  simp only [← c23616, ← c23617, k2, ← c23618] at e_361_17
  have hr := e_361_17
  simp only [e_361_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 361 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 337)) := by
  have c23619 := (publicBatchWrapper86_copies738 a h).2.2.2.1
  have c23620 := (publicBatchWrapper86_copies738 a h).2.2.2.2.1
  have c23621 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.1
  have c23622 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.1
  have c23623 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.1
  have c23624 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_361_18 := arithEq_of_rows h (row := 361) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_18
  simp only [← c23619, ← c23620, ← c23621] at e_361_18
  have e_361_19 := arithEq_of_rows h (row := 361) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_361_19
  simp only [← c23622, ← c23623, k2, ← c23624] at e_361_19
  have hr := e_361_19
  simp only [e_361_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 362 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 338)) := by
  have c23625 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.1
  have c23626 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23627 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23628 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23629 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23630 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_362_0 := arithEq_of_rows h (row := 362) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_0
  simp only [← c23625, ← c23626, ← c23627] at e_362_0
  have e_362_1 := arithEq_of_rows h (row := 362) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_1
  simp only [← c23628, ← c23629, k2, ← c23630] at e_362_1
  have hr := e_362_1
  simp only [e_362_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 362 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 339)) := by
  have c23631 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23632 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23633 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23634 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23635 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23636 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_362_2 := arithEq_of_rows h (row := 362) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_2
  simp only [← c23631, ← c23632, ← c23633] at e_362_2
  have e_362_3 := arithEq_of_rows h (row := 362) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_3
  simp only [← c23634, ← c23635, k2, ← c23636] at e_362_3
  have hr := e_362_3
  simp only [e_362_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 362 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 340)) := by
  have c23637 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23638 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23639 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23640 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23641 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23642 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_362_4 := arithEq_of_rows h (row := 362) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_4
  simp only [← c23637, ← c23638, ← c23639] at e_362_4
  have e_362_5 := arithEq_of_rows h (row := 362) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_5
  simp only [← c23640, ← c23641, k2, ← c23642] at e_362_5
  have hr := e_362_5
  simp only [e_362_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 362 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 341)) := by
  have c23643 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23644 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23645 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23646 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23647 := (publicBatchWrapper86_copies738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23648 := (publicBatchWrapper86_copies739 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_362_6 := arithEq_of_rows h (row := 362) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_6
  simp only [← c23643, ← c23644, ← c23645] at e_362_6
  have e_362_7 := arithEq_of_rows h (row := 362) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_7
  simp only [← c23646, ← c23647, k2, ← c23648] at e_362_7
  have hr := e_362_7
  simp only [e_362_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 362 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 342)) := by
  have c23649 := (publicBatchWrapper86_copies739 a h).2.1
  have c23650 := (publicBatchWrapper86_copies739 a h).2.2.1
  have c23651 := (publicBatchWrapper86_copies739 a h).2.2.2.1
  have c23652 := (publicBatchWrapper86_copies739 a h).2.2.2.2.1
  have c23653 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.1
  have c23654 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_362_8 := arithEq_of_rows h (row := 362) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_8
  simp only [← c23649, ← c23650, ← c23651] at e_362_8
  have e_362_9 := arithEq_of_rows h (row := 362) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_9
  simp only [← c23652, ← c23653, k2, ← c23654] at e_362_9
  have hr := e_362_9
  simp only [e_362_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 362 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 343)) := by
  have c23655 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.1
  have c23656 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.1
  have c23657 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.1
  have c23658 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23659 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23660 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_362_10 := arithEq_of_rows h (row := 362) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_10
  simp only [← c23655, ← c23656, ← c23657] at e_362_10
  have e_362_11 := arithEq_of_rows h (row := 362) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_11
  simp only [← c23658, ← c23659, k2, ← c23660] at e_362_11
  have hr := e_362_11
  simp only [e_362_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 362 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 344)) := by
  have c23661 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23662 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23663 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23664 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23665 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23666 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_362_12 := arithEq_of_rows h (row := 362) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_12
  simp only [← c23661, ← c23662, ← c23663] at e_362_12
  have e_362_13 := arithEq_of_rows h (row := 362) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_13
  simp only [← c23664, ← c23665, k2, ← c23666] at e_362_13
  have hr := e_362_13
  simp only [e_362_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 362 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 345)) := by
  have c23667 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23668 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23669 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23670 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23671 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23672 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_362_14 := arithEq_of_rows h (row := 362) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_14
  simp only [← c23667, ← c23668, ← c23669] at e_362_14
  have e_362_15 := arithEq_of_rows h (row := 362) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_15
  simp only [← c23670, ← c23671, k2, ← c23672] at e_362_15
  have hr := e_362_15
  simp only [e_362_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 362 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 346)) := by
  have c23673 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23674 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23675 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23676 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23677 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23678 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_362_16 := arithEq_of_rows h (row := 362) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_16
  simp only [← c23673, ← c23674, ← c23675] at e_362_16
  have e_362_17 := arithEq_of_rows h (row := 362) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_17
  simp only [← c23676, ← c23677, k2, ← c23678] at e_362_17
  have hr := e_362_17
  simp only [e_362_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 362 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 347)) := by
  have c23679 := (publicBatchWrapper86_copies739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23680 := (publicBatchWrapper86_copies740 a h).1
  have c23681 := (publicBatchWrapper86_copies740 a h).2.1
  have c23682 := (publicBatchWrapper86_copies740 a h).2.2.1
  have c23683 := (publicBatchWrapper86_copies740 a h).2.2.2.1
  have c23684 := (publicBatchWrapper86_copies740 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_362_18 := arithEq_of_rows h (row := 362) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_18
  simp only [← c23679, ← c23680, ← c23681] at e_362_18
  have e_362_19 := arithEq_of_rows h (row := 362) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_362_19
  simp only [← c23682, ← c23683, k2, ← c23684] at e_362_19
  have hr := e_362_19
  simp only [e_362_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 363 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 348)) := by
  have c23685 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.1
  have c23686 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.1
  have c23687 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.1
  have c23688 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.1
  have c23689 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.1
  have c23690 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_363_0 := arithEq_of_rows h (row := 363) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_0
  simp only [← c23685, ← c23686, ← c23687] at e_363_0
  have e_363_1 := arithEq_of_rows h (row := 363) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_1
  simp only [← c23688, ← c23689, k2, ← c23690] at e_363_1
  have hr := e_363_1
  simp only [e_363_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 363 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 349)) := by
  have c23691 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23692 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23693 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23694 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23695 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23696 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_363_2 := arithEq_of_rows h (row := 363) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_2
  simp only [← c23691, ← c23692, ← c23693] at e_363_2
  have e_363_3 := arithEq_of_rows h (row := 363) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_3
  simp only [← c23694, ← c23695, k2, ← c23696] at e_363_3
  have hr := e_363_3
  simp only [e_363_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 363 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 350)) := by
  have c23697 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23698 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23699 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23700 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23701 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23702 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_363_4 := arithEq_of_rows h (row := 363) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_4
  simp only [← c23697, ← c23698, ← c23699] at e_363_4
  have e_363_5 := arithEq_of_rows h (row := 363) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_5
  simp only [← c23700, ← c23701, k2, ← c23702] at e_363_5
  have hr := e_363_5
  simp only [e_363_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 363 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 351)) := by
  have c23703 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23704 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23705 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23706 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23707 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23708 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_363_6 := arithEq_of_rows h (row := 363) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_6
  simp only [← c23703, ← c23704, ← c23705] at e_363_6
  have e_363_7 := arithEq_of_rows h (row := 363) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_7
  simp only [← c23706, ← c23707, k2, ← c23708] at e_363_7
  have hr := e_363_7
  simp only [e_363_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 363 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 352)) := by
  have c23709 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23710 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23711 := (publicBatchWrapper86_copies740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23712 := (publicBatchWrapper86_copies741 a h).1
  have c23713 := (publicBatchWrapper86_copies741 a h).2.1
  have c23714 := (publicBatchWrapper86_copies741 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_363_8 := arithEq_of_rows h (row := 363) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_8
  simp only [← c23709, ← c23710, ← c23711] at e_363_8
  have e_363_9 := arithEq_of_rows h (row := 363) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_9
  simp only [← c23712, ← c23713, k2, ← c23714] at e_363_9
  have hr := e_363_9
  simp only [e_363_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 363 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 353)) := by
  have c23715 := (publicBatchWrapper86_copies741 a h).2.2.2.1
  have c23716 := (publicBatchWrapper86_copies741 a h).2.2.2.2.1
  have c23717 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.1
  have c23718 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.1
  have c23719 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.1
  have c23720 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_363_10 := arithEq_of_rows h (row := 363) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_10
  simp only [← c23715, ← c23716, ← c23717] at e_363_10
  have e_363_11 := arithEq_of_rows h (row := 363) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_11
  simp only [← c23718, ← c23719, k2, ← c23720] at e_363_11
  have hr := e_363_11
  simp only [e_363_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 363 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 354)) := by
  have c23721 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.1
  have c23722 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23723 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23724 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23725 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23726 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_363_12 := arithEq_of_rows h (row := 363) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_12
  simp only [← c23721, ← c23722, ← c23723] at e_363_12
  have e_363_13 := arithEq_of_rows h (row := 363) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_13
  simp only [← c23724, ← c23725, k2, ← c23726] at e_363_13
  have hr := e_363_13
  simp only [e_363_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 363 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 355)) := by
  have c23727 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23728 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23729 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23730 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23731 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23732 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_363_14 := arithEq_of_rows h (row := 363) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_14
  simp only [← c23727, ← c23728, ← c23729] at e_363_14
  have e_363_15 := arithEq_of_rows h (row := 363) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_15
  simp only [← c23730, ← c23731, k2, ← c23732] at e_363_15
  have hr := e_363_15
  simp only [e_363_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 363 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 356)) := by
  have c23733 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23734 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23735 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23736 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23737 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23738 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_363_16 := arithEq_of_rows h (row := 363) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_16
  simp only [← c23733, ← c23734, ← c23735] at e_363_16
  have e_363_17 := arithEq_of_rows h (row := 363) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_17
  simp only [← c23736, ← c23737, k2, ← c23738] at e_363_17
  have hr := e_363_17
  simp only [e_363_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 363 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 357)) := by
  have c23739 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23740 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23741 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23742 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23743 := (publicBatchWrapper86_copies741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23744 := (publicBatchWrapper86_copies742 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_363_18 := arithEq_of_rows h (row := 363) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_18
  simp only [← c23739, ← c23740, ← c23741] at e_363_18
  have e_363_19 := arithEq_of_rows h (row := 363) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_363_19
  simp only [← c23742, ← c23743, k2, ← c23744] at e_363_19
  have hr := e_363_19
  simp only [e_363_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 364 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 358)) := by
  have c23745 := (publicBatchWrapper86_copies742 a h).2.1
  have c23746 := (publicBatchWrapper86_copies742 a h).2.2.1
  have c23747 := (publicBatchWrapper86_copies742 a h).2.2.2.1
  have c23748 := (publicBatchWrapper86_copies742 a h).2.2.2.2.1
  have c23749 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.1
  have c23750 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_364_0 := arithEq_of_rows h (row := 364) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_0
  simp only [← c23745, ← c23746, ← c23747] at e_364_0
  have e_364_1 := arithEq_of_rows h (row := 364) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_1
  simp only [← c23748, ← c23749, k2, ← c23750] at e_364_1
  have hr := e_364_1
  simp only [e_364_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 364 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 359)) := by
  have c23751 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.1
  have c23752 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.1
  have c23753 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.1
  have c23754 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23755 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23756 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_364_2 := arithEq_of_rows h (row := 364) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_2
  simp only [← c23751, ← c23752, ← c23753] at e_364_2
  have e_364_3 := arithEq_of_rows h (row := 364) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_3
  simp only [← c23754, ← c23755, k2, ← c23756] at e_364_3
  have hr := e_364_3
  simp only [e_364_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 364 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 360)) := by
  have c23757 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23758 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23759 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23760 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23761 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23762 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_364_4 := arithEq_of_rows h (row := 364) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_4
  simp only [← c23757, ← c23758, ← c23759] at e_364_4
  have e_364_5 := arithEq_of_rows h (row := 364) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_5
  simp only [← c23760, ← c23761, k2, ← c23762] at e_364_5
  have hr := e_364_5
  simp only [e_364_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 364 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 361)) := by
  have c23763 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23764 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23765 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23766 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23767 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23768 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_364_6 := arithEq_of_rows h (row := 364) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_6
  simp only [← c23763, ← c23764, ← c23765] at e_364_6
  have e_364_7 := arithEq_of_rows h (row := 364) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_7
  simp only [← c23766, ← c23767, k2, ← c23768] at e_364_7
  have hr := e_364_7
  simp only [e_364_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 364 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 362)) := by
  have c23769 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23770 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23771 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23772 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23773 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23774 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_364_8 := arithEq_of_rows h (row := 364) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_8
  simp only [← c23769, ← c23770, ← c23771] at e_364_8
  have e_364_9 := arithEq_of_rows h (row := 364) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_9
  simp only [← c23772, ← c23773, k2, ← c23774] at e_364_9
  have hr := e_364_9
  simp only [e_364_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 364 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 363)) := by
  have c23775 := (publicBatchWrapper86_copies742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23776 := (publicBatchWrapper86_copies743 a h).1
  have c23777 := (publicBatchWrapper86_copies743 a h).2.1
  have c23778 := (publicBatchWrapper86_copies743 a h).2.2.1
  have c23779 := (publicBatchWrapper86_copies743 a h).2.2.2.1
  have c23780 := (publicBatchWrapper86_copies743 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_364_10 := arithEq_of_rows h (row := 364) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_10
  simp only [← c23775, ← c23776, ← c23777] at e_364_10
  have e_364_11 := arithEq_of_rows h (row := 364) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_11
  simp only [← c23778, ← c23779, k2, ← c23780] at e_364_11
  have hr := e_364_11
  simp only [e_364_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 364 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 364)) := by
  have c23781 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.1
  have c23782 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.1
  have c23783 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.1
  have c23784 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.1
  have c23785 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.1
  have c23786 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_364_12 := arithEq_of_rows h (row := 364) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_12
  simp only [← c23781, ← c23782, ← c23783] at e_364_12
  have e_364_13 := arithEq_of_rows h (row := 364) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_13
  simp only [← c23784, ← c23785, k2, ← c23786] at e_364_13
  have hr := e_364_13
  simp only [e_364_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 364 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 365)) := by
  have c23787 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23788 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23789 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23790 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23791 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23792 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_364_14 := arithEq_of_rows h (row := 364) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_14
  simp only [← c23787, ← c23788, ← c23789] at e_364_14
  have e_364_15 := arithEq_of_rows h (row := 364) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_15
  simp only [← c23790, ← c23791, k2, ← c23792] at e_364_15
  have hr := e_364_15
  simp only [e_364_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 364 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 366)) := by
  have c23793 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23794 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23795 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23796 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23797 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23798 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_364_16 := arithEq_of_rows h (row := 364) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_16
  simp only [← c23793, ← c23794, ← c23795] at e_364_16
  have e_364_17 := arithEq_of_rows h (row := 364) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_17
  simp only [← c23796, ← c23797, k2, ← c23798] at e_364_17
  have hr := e_364_17
  simp only [e_364_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 364 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 367)) := by
  have c23799 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23800 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23801 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23802 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23803 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23804 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_364_18 := arithEq_of_rows h (row := 364) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_18
  simp only [← c23799, ← c23800, ← c23801] at e_364_18
  have e_364_19 := arithEq_of_rows h (row := 364) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_364_19
  simp only [← c23802, ← c23803, k2, ← c23804] at e_364_19
  have hr := e_364_19
  simp only [e_364_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 365 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 368)) := by
  have c23805 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23806 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23807 := (publicBatchWrapper86_copies743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23808 := (publicBatchWrapper86_copies744 a h).1
  have c23809 := (publicBatchWrapper86_copies744 a h).2.1
  have c23810 := (publicBatchWrapper86_copies744 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_365_0 := arithEq_of_rows h (row := 365) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_0
  simp only [← c23805, ← c23806, ← c23807] at e_365_0
  have e_365_1 := arithEq_of_rows h (row := 365) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_1
  simp only [← c23808, ← c23809, k2, ← c23810] at e_365_1
  have hr := e_365_1
  simp only [e_365_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 365 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 369)) := by
  have c23811 := (publicBatchWrapper86_copies744 a h).2.2.2.1
  have c23812 := (publicBatchWrapper86_copies744 a h).2.2.2.2.1
  have c23813 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.1
  have c23814 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.1
  have c23815 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.1
  have c23816 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_365_2 := arithEq_of_rows h (row := 365) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_2
  simp only [← c23811, ← c23812, ← c23813] at e_365_2
  have e_365_3 := arithEq_of_rows h (row := 365) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_3
  simp only [← c23814, ← c23815, k2, ← c23816] at e_365_3
  have hr := e_365_3
  simp only [e_365_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 365 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 410)) := by
  have c23817 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.1
  have c23818 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23819 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23820 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23821 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23822 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_365_4 := arithEq_of_rows h (row := 365) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_4
  simp only [← c23817, ← c23818, ← c23819] at e_365_4
  have e_365_5 := arithEq_of_rows h (row := 365) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_5
  simp only [← c23820, ← c23821, k2, ← c23822] at e_365_5
  have hr := e_365_5
  simp only [e_365_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 365 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 411)) := by
  have c23823 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23824 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23825 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23826 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23827 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23828 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_365_6 := arithEq_of_rows h (row := 365) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_6
  simp only [← c23823, ← c23824, ← c23825] at e_365_6
  have e_365_7 := arithEq_of_rows h (row := 365) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_7
  simp only [← c23826, ← c23827, k2, ← c23828] at e_365_7
  have hr := e_365_7
  simp only [e_365_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 365 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 412)) := by
  have c23829 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23830 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23831 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23832 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23833 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23834 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_365_8 := arithEq_of_rows h (row := 365) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_8
  simp only [← c23829, ← c23830, ← c23831] at e_365_8
  have e_365_9 := arithEq_of_rows h (row := 365) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_9
  simp only [← c23832, ← c23833, k2, ← c23834] at e_365_9
  have hr := e_365_9
  simp only [e_365_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 365 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 413)) := by
  have c23835 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23836 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23837 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23838 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23839 := (publicBatchWrapper86_copies744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23840 := (publicBatchWrapper86_copies745 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_365_10 := arithEq_of_rows h (row := 365) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_10
  simp only [← c23835, ← c23836, ← c23837] at e_365_10
  have e_365_11 := arithEq_of_rows h (row := 365) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_11
  simp only [← c23838, ← c23839, k2, ← c23840] at e_365_11
  have hr := e_365_11
  simp only [e_365_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 365 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 414)) := by
  have c23841 := (publicBatchWrapper86_copies745 a h).2.1
  have c23842 := (publicBatchWrapper86_copies745 a h).2.2.1
  have c23843 := (publicBatchWrapper86_copies745 a h).2.2.2.1
  have c23844 := (publicBatchWrapper86_copies745 a h).2.2.2.2.1
  have c23845 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.1
  have c23846 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_365_12 := arithEq_of_rows h (row := 365) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_12
  simp only [← c23841, ← c23842, ← c23843] at e_365_12
  have e_365_13 := arithEq_of_rows h (row := 365) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_13
  simp only [← c23844, ← c23845, k2, ← c23846] at e_365_13
  have hr := e_365_13
  simp only [e_365_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 365 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 415)) := by
  have c23847 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.1
  have c23848 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.1
  have c23849 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.1
  have c23850 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23851 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23852 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_365_14 := arithEq_of_rows h (row := 365) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_14
  simp only [← c23847, ← c23848, ← c23849] at e_365_14
  have e_365_15 := arithEq_of_rows h (row := 365) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_15
  simp only [← c23850, ← c23851, k2, ← c23852] at e_365_15
  have hr := e_365_15
  simp only [e_365_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 365 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 416)) := by
  have c23853 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23854 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23855 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23856 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23857 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23858 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_365_16 := arithEq_of_rows h (row := 365) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_16
  simp only [← c23853, ← c23854, ← c23855] at e_365_16
  have e_365_17 := arithEq_of_rows h (row := 365) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_17
  simp only [← c23856, ← c23857, k2, ← c23858] at e_365_17
  have hr := e_365_17
  simp only [e_365_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 365 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 417)) := by
  have c23859 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23860 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23861 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23862 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23863 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23864 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_365_18 := arithEq_of_rows h (row := 365) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_18
  simp only [← c23859, ← c23860, ← c23861] at e_365_18
  have e_365_19 := arithEq_of_rows h (row := 365) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_365_19
  simp only [← c23862, ← c23863, k2, ← c23864] at e_365_19
  have hr := e_365_19
  simp only [e_365_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 366 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 418)) := by
  have c23865 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23866 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23867 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23868 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23869 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23870 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_366_0 := arithEq_of_rows h (row := 366) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_0
  simp only [← c23865, ← c23866, ← c23867] at e_366_0
  have e_366_1 := arithEq_of_rows h (row := 366) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_1
  simp only [← c23868, ← c23869, k2, ← c23870] at e_366_1
  have hr := e_366_1
  simp only [e_366_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 366 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 419)) := by
  have c23871 := (publicBatchWrapper86_copies745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23872 := (publicBatchWrapper86_copies746 a h).1
  have c23873 := (publicBatchWrapper86_copies746 a h).2.1
  have c23874 := (publicBatchWrapper86_copies746 a h).2.2.1
  have c23875 := (publicBatchWrapper86_copies746 a h).2.2.2.1
  have c23876 := (publicBatchWrapper86_copies746 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_366_2 := arithEq_of_rows h (row := 366) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_2
  simp only [← c23871, ← c23872, ← c23873] at e_366_2
  have e_366_3 := arithEq_of_rows h (row := 366) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_3
  simp only [← c23874, ← c23875, k2, ← c23876] at e_366_3
  have hr := e_366_3
  simp only [e_366_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 366 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 420)) := by
  have c23877 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.1
  have c23878 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.1
  have c23879 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.1
  have c23880 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.1
  have c23881 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.1
  have c23882 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_366_4 := arithEq_of_rows h (row := 366) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_4
  simp only [← c23877, ← c23878, ← c23879] at e_366_4
  have e_366_5 := arithEq_of_rows h (row := 366) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_5
  simp only [← c23880, ← c23881, k2, ← c23882] at e_366_5
  have hr := e_366_5
  simp only [e_366_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 366 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 421)) := by
  have c23883 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23884 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23885 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23886 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23887 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23888 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_366_6 := arithEq_of_rows h (row := 366) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_6
  simp only [← c23883, ← c23884, ← c23885] at e_366_6
  have e_366_7 := arithEq_of_rows h (row := 366) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_7
  simp only [← c23886, ← c23887, k2, ← c23888] at e_366_7
  have hr := e_366_7
  simp only [e_366_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 366 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 422)) := by
  have c23889 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23890 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23891 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23892 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23893 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23894 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_366_8 := arithEq_of_rows h (row := 366) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_8
  simp only [← c23889, ← c23890, ← c23891] at e_366_8
  have e_366_9 := arithEq_of_rows h (row := 366) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_9
  simp only [← c23892, ← c23893, k2, ← c23894] at e_366_9
  have hr := e_366_9
  simp only [e_366_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 366 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 423)) := by
  have c23895 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23896 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23897 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23898 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23899 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23900 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_366_10 := arithEq_of_rows h (row := 366) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_10
  simp only [← c23895, ← c23896, ← c23897] at e_366_10
  have e_366_11 := arithEq_of_rows h (row := 366) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_11
  simp only [← c23898, ← c23899, k2, ← c23900] at e_366_11
  have hr := e_366_11
  simp only [e_366_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 366 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 424)) := by
  have c23901 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23902 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23903 := (publicBatchWrapper86_copies746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23904 := (publicBatchWrapper86_copies747 a h).1
  have c23905 := (publicBatchWrapper86_copies747 a h).2.1
  have c23906 := (publicBatchWrapper86_copies747 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_366_12 := arithEq_of_rows h (row := 366) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_12
  simp only [← c23901, ← c23902, ← c23903] at e_366_12
  have e_366_13 := arithEq_of_rows h (row := 366) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_13
  simp only [← c23904, ← c23905, k2, ← c23906] at e_366_13
  have hr := e_366_13
  simp only [e_366_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 366 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 425)) := by
  have c23907 := (publicBatchWrapper86_copies747 a h).2.2.2.1
  have c23908 := (publicBatchWrapper86_copies747 a h).2.2.2.2.1
  have c23909 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.1
  have c23910 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.1
  have c23911 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.1
  have c23912 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_366_14 := arithEq_of_rows h (row := 366) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_14
  simp only [← c23907, ← c23908, ← c23909] at e_366_14
  have e_366_15 := arithEq_of_rows h (row := 366) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_15
  simp only [← c23910, ← c23911, k2, ← c23912] at e_366_15
  have hr := e_366_15
  simp only [e_366_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 366 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 426)) := by
  have c23913 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.1
  have c23914 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23915 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23916 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23917 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23918 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_366_16 := arithEq_of_rows h (row := 366) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_16
  simp only [← c23913, ← c23914, ← c23915] at e_366_16
  have e_366_17 := arithEq_of_rows h (row := 366) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_17
  simp only [← c23916, ← c23917, k2, ← c23918] at e_366_17
  have hr := e_366_17
  simp only [e_366_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 366 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 427)) := by
  have c23919 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23920 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23921 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23922 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23923 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23924 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_366_18 := arithEq_of_rows h (row := 366) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_18
  simp only [← c23919, ← c23920, ← c23921] at e_366_18
  have e_366_19 := arithEq_of_rows h (row := 366) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_366_19
  simp only [← c23922, ← c23923, k2, ← c23924] at e_366_19
  have hr := e_366_19
  simp only [e_366_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 367 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 428)) := by
  have c23925 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23926 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23927 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23928 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23929 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23930 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_367_0 := arithEq_of_rows h (row := 367) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_0
  simp only [← c23925, ← c23926, ← c23927] at e_367_0
  have e_367_1 := arithEq_of_rows h (row := 367) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_1
  simp only [← c23928, ← c23929, k2, ← c23930] at e_367_1
  have hr := e_367_1
  simp only [e_367_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 367 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 429)) := by
  have c23931 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23932 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23933 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23934 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23935 := (publicBatchWrapper86_copies747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23936 := (publicBatchWrapper86_copies748 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_367_2 := arithEq_of_rows h (row := 367) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_2
  simp only [← c23931, ← c23932, ← c23933] at e_367_2
  have e_367_3 := arithEq_of_rows h (row := 367) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_3
  simp only [← c23934, ← c23935, k2, ← c23936] at e_367_3
  have hr := e_367_3
  simp only [e_367_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 367 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 430)) := by
  have c23937 := (publicBatchWrapper86_copies748 a h).2.1
  have c23938 := (publicBatchWrapper86_copies748 a h).2.2.1
  have c23939 := (publicBatchWrapper86_copies748 a h).2.2.2.1
  have c23940 := (publicBatchWrapper86_copies748 a h).2.2.2.2.1
  have c23941 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.1
  have c23942 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_367_4 := arithEq_of_rows h (row := 367) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_4
  simp only [← c23937, ← c23938, ← c23939] at e_367_4
  have e_367_5 := arithEq_of_rows h (row := 367) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_5
  simp only [← c23940, ← c23941, k2, ← c23942] at e_367_5
  have hr := e_367_5
  simp only [e_367_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 367 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 431)) := by
  have c23943 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.1
  have c23944 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.1
  have c23945 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.1
  have c23946 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.1
  have c23947 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23948 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_367_6 := arithEq_of_rows h (row := 367) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_6
  simp only [← c23943, ← c23944, ← c23945] at e_367_6
  have e_367_7 := arithEq_of_rows h (row := 367) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_7
  simp only [← c23946, ← c23947, k2, ← c23948] at e_367_7
  have hr := e_367_7
  simp only [e_367_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 367 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 432)) := by
  have c23949 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23950 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23951 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23952 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23953 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23954 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_367_8 := arithEq_of_rows h (row := 367) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_8
  simp only [← c23949, ← c23950, ← c23951] at e_367_8
  have e_367_9 := arithEq_of_rows h (row := 367) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_9
  simp only [← c23952, ← c23953, k2, ← c23954] at e_367_9
  have hr := e_367_9
  simp only [e_367_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 367 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 433)) := by
  have c23955 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23956 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23957 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23958 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23959 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23960 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_367_10 := arithEq_of_rows h (row := 367) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_10
  simp only [← c23955, ← c23956, ← c23957] at e_367_10
  have e_367_11 := arithEq_of_rows h (row := 367) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_11
  simp only [← c23958, ← c23959, k2, ← c23960] at e_367_11
  have hr := e_367_11
  simp only [e_367_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 367 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 434)) := by
  have c23961 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23962 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23963 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23964 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23965 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23966 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_367_12 := arithEq_of_rows h (row := 367) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_12
  simp only [← c23961, ← c23962, ← c23963] at e_367_12
  have e_367_13 := arithEq_of_rows h (row := 367) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_13
  simp only [← c23964, ← c23965, k2, ← c23966] at e_367_13
  have hr := e_367_13
  simp only [e_367_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 367 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 435)) := by
  have c23967 := (publicBatchWrapper86_copies748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c23968 := (publicBatchWrapper86_copies749 a h).1
  have c23969 := (publicBatchWrapper86_copies749 a h).2.1
  have c23970 := (publicBatchWrapper86_copies749 a h).2.2.1
  have c23971 := (publicBatchWrapper86_copies749 a h).2.2.2.1
  have c23972 := (publicBatchWrapper86_copies749 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_367_14 := arithEq_of_rows h (row := 367) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_14
  simp only [← c23967, ← c23968, ← c23969] at e_367_14
  have e_367_15 := arithEq_of_rows h (row := 367) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_15
  simp only [← c23970, ← c23971, k2, ← c23972] at e_367_15
  have hr := e_367_15
  simp only [e_367_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 367 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 436)) := by
  have c23973 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.1
  have c23974 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.1
  have c23975 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.1
  have c23976 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.1
  have c23977 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.1
  have c23978 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_367_16 := arithEq_of_rows h (row := 367) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_16
  simp only [← c23973, ← c23974, ← c23975] at e_367_16
  have e_367_17 := arithEq_of_rows h (row := 367) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_17
  simp only [← c23976, ← c23977, k2, ← c23978] at e_367_17
  have hr := e_367_17
  simp only [e_367_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 367 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 437)) := by
  have c23979 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c23980 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23981 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23982 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23983 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23984 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_367_18 := arithEq_of_rows h (row := 367) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_18
  simp only [← c23979, ← c23980, ← c23981] at e_367_18
  have e_367_19 := arithEq_of_rows h (row := 367) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_367_19
  simp only [← c23982, ← c23983, k2, ← c23984] at e_367_19
  have hr := e_367_19
  simp only [e_367_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 368 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 438)) := by
  have c23985 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23986 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23987 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23988 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23989 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23990 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_368_0 := arithEq_of_rows h (row := 368) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_0
  simp only [← c23985, ← c23986, ← c23987] at e_368_0
  have e_368_1 := arithEq_of_rows h (row := 368) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_1
  simp only [← c23988, ← c23989, k2, ← c23990] at e_368_1
  have hr := e_368_1
  simp only [e_368_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 368 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 439)) := by
  have c23991 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23992 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23993 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23994 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23995 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23996 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_368_2 := arithEq_of_rows h (row := 368) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_2
  simp only [← c23991, ← c23992, ← c23993] at e_368_2
  have e_368_3 := arithEq_of_rows h (row := 368) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_3
  simp only [← c23994, ← c23995, k2, ← c23996] at e_368_3
  have hr := e_368_3
  simp only [e_368_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 368 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 440)) := by
  have c23997 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23998 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c23999 := (publicBatchWrapper86_copies749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24000 := (publicBatchWrapper86_copies750 a h).1
  have c24001 := (publicBatchWrapper86_copies750 a h).2.1
  have c24002 := (publicBatchWrapper86_copies750 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_368_4 := arithEq_of_rows h (row := 368) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_4
  simp only [← c23997, ← c23998, ← c23999] at e_368_4
  have e_368_5 := arithEq_of_rows h (row := 368) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_5
  simp only [← c24000, ← c24001, k2, ← c24002] at e_368_5
  have hr := e_368_5
  simp only [e_368_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 368 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 441)) := by
  have c24003 := (publicBatchWrapper86_copies750 a h).2.2.2.1
  have c24004 := (publicBatchWrapper86_copies750 a h).2.2.2.2.1
  have c24005 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.1
  have c24006 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.1
  have c24007 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.1
  have c24008 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_368_6 := arithEq_of_rows h (row := 368) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_6
  simp only [← c24003, ← c24004, ← c24005] at e_368_6
  have e_368_7 := arithEq_of_rows h (row := 368) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_7
  simp only [← c24006, ← c24007, k2, ← c24008] at e_368_7
  have hr := e_368_7
  simp only [e_368_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 368 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 442)) := by
  have c24009 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.1
  have c24010 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24011 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24012 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24013 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24014 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_368_8 := arithEq_of_rows h (row := 368) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_8
  simp only [← c24009, ← c24010, ← c24011] at e_368_8
  have e_368_9 := arithEq_of_rows h (row := 368) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_9
  simp only [← c24012, ← c24013, k2, ← c24014] at e_368_9
  have hr := e_368_9
  simp only [e_368_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 368 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 443)) := by
  have c24015 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24016 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24017 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24018 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24019 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24020 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_368_10 := arithEq_of_rows h (row := 368) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_10
  simp only [← c24015, ← c24016, ← c24017] at e_368_10
  have e_368_11 := arithEq_of_rows h (row := 368) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_11
  simp only [← c24018, ← c24019, k2, ← c24020] at e_368_11
  have hr := e_368_11
  simp only [e_368_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 368 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 444)) := by
  have c24021 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24022 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24023 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24024 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24025 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24026 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_368_12 := arithEq_of_rows h (row := 368) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_12
  simp only [← c24021, ← c24022, ← c24023] at e_368_12
  have e_368_13 := arithEq_of_rows h (row := 368) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_13
  simp only [← c24024, ← c24025, k2, ← c24026] at e_368_13
  have hr := e_368_13
  simp only [e_368_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 368 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 445)) := by
  have c24027 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24028 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24029 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24030 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24031 := (publicBatchWrapper86_copies750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24032 := (publicBatchWrapper86_copies751 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_368_14 := arithEq_of_rows h (row := 368) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_14
  simp only [← c24027, ← c24028, ← c24029] at e_368_14
  have e_368_15 := arithEq_of_rows h (row := 368) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_15
  simp only [← c24030, ← c24031, k2, ← c24032] at e_368_15
  have hr := e_368_15
  simp only [e_368_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 368 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 446)) := by
  have c24033 := (publicBatchWrapper86_copies751 a h).2.1
  have c24034 := (publicBatchWrapper86_copies751 a h).2.2.1
  have c24035 := (publicBatchWrapper86_copies751 a h).2.2.2.1
  have c24036 := (publicBatchWrapper86_copies751 a h).2.2.2.2.1
  have c24037 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.1
  have c24038 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_368_16 := arithEq_of_rows h (row := 368) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_16
  simp only [← c24033, ← c24034, ← c24035] at e_368_16
  have e_368_17 := arithEq_of_rows h (row := 368) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_17
  simp only [← c24036, ← c24037, k2, ← c24038] at e_368_17
  have hr := e_368_17
  simp only [e_368_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 368 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 447)) := by
  have c24039 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.1
  have c24040 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.1
  have c24041 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.1
  have c24042 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24043 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24044 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_368_18 := arithEq_of_rows h (row := 368) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_18
  simp only [← c24039, ← c24040, ← c24041] at e_368_18
  have e_368_19 := arithEq_of_rows h (row := 368) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_368_19
  simp only [← c24042, ← c24043, k2, ← c24044] at e_368_19
  have hr := e_368_19
  simp only [e_368_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 369 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 448)) := by
  have c24045 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24046 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24047 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24048 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24049 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24050 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_369_0 := arithEq_of_rows h (row := 369) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_0
  simp only [← c24045, ← c24046, ← c24047] at e_369_0
  have e_369_1 := arithEq_of_rows h (row := 369) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_1
  simp only [← c24048, ← c24049, k2, ← c24050] at e_369_1
  have hr := e_369_1
  simp only [e_369_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 369 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 449)) := by
  have c24051 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24052 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24053 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24054 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24055 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24056 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_369_2 := arithEq_of_rows h (row := 369) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_2
  simp only [← c24051, ← c24052, ← c24053] at e_369_2
  have e_369_3 := arithEq_of_rows h (row := 369) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_3
  simp only [← c24054, ← c24055, k2, ← c24056] at e_369_3
  have hr := e_369_3
  simp only [e_369_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 369 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 450)) := by
  have c24057 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24058 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24059 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24060 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24061 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24062 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_369_4 := arithEq_of_rows h (row := 369) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_4
  simp only [← c24057, ← c24058, ← c24059] at e_369_4
  have e_369_5 := arithEq_of_rows h (row := 369) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_5
  simp only [← c24060, ← c24061, k2, ← c24062] at e_369_5
  have hr := e_369_5
  simp only [e_369_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 369 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 451)) := by
  have c24063 := (publicBatchWrapper86_copies751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24064 := (publicBatchWrapper86_copies752 a h).1
  have c24065 := (publicBatchWrapper86_copies752 a h).2.1
  have c24066 := (publicBatchWrapper86_copies752 a h).2.2.1
  have c24067 := (publicBatchWrapper86_copies752 a h).2.2.2.1
  have c24068 := (publicBatchWrapper86_copies752 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_369_6 := arithEq_of_rows h (row := 369) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_6
  simp only [← c24063, ← c24064, ← c24065] at e_369_6
  have e_369_7 := arithEq_of_rows h (row := 369) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_7
  simp only [← c24066, ← c24067, k2, ← c24068] at e_369_7
  have hr := e_369_7
  simp only [e_369_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 369 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 452)) := by
  have c24069 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.1
  have c24070 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.1
  have c24071 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.1
  have c24072 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.1
  have c24073 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.1
  have c24074 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_369_8 := arithEq_of_rows h (row := 369) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_8
  simp only [← c24069, ← c24070, ← c24071] at e_369_8
  have e_369_9 := arithEq_of_rows h (row := 369) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_9
  simp only [← c24072, ← c24073, k2, ← c24074] at e_369_9
  have hr := e_369_9
  simp only [e_369_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 369 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 453)) := by
  have c24075 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24076 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24077 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24078 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24079 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24080 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_369_10 := arithEq_of_rows h (row := 369) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_10
  simp only [← c24075, ← c24076, ← c24077] at e_369_10
  have e_369_11 := arithEq_of_rows h (row := 369) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_11
  simp only [← c24078, ← c24079, k2, ← c24080] at e_369_11
  have hr := e_369_11
  simp only [e_369_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 369 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 454)) := by
  have c24081 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24082 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24083 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24084 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24085 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24086 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_369_12 := arithEq_of_rows h (row := 369) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_12
  simp only [← c24081, ← c24082, ← c24083] at e_369_12
  have e_369_13 := arithEq_of_rows h (row := 369) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_13
  simp only [← c24084, ← c24085, k2, ← c24086] at e_369_13
  have hr := e_369_13
  simp only [e_369_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 369 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 455)) := by
  have c24087 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24088 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24089 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24090 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24091 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24092 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_369_14 := arithEq_of_rows h (row := 369) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_14
  simp only [← c24087, ← c24088, ← c24089] at e_369_14
  have e_369_15 := arithEq_of_rows h (row := 369) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_15
  simp only [← c24090, ← c24091, k2, ← c24092] at e_369_15
  have hr := e_369_15
  simp only [e_369_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 369 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 456)) := by
  have c24093 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24094 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24095 := (publicBatchWrapper86_copies752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24096 := (publicBatchWrapper86_copies753 a h).1
  have c24097 := (publicBatchWrapper86_copies753 a h).2.1
  have c24098 := (publicBatchWrapper86_copies753 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_369_16 := arithEq_of_rows h (row := 369) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_16
  simp only [← c24093, ← c24094, ← c24095] at e_369_16
  have e_369_17 := arithEq_of_rows h (row := 369) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_17
  simp only [← c24096, ← c24097, k2, ← c24098] at e_369_17
  have hr := e_369_17
  simp only [e_369_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 369 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 457)) := by
  have c24099 := (publicBatchWrapper86_copies753 a h).2.2.2.1
  have c24100 := (publicBatchWrapper86_copies753 a h).2.2.2.2.1
  have c24101 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.1
  have c24102 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.1
  have c24103 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.1
  have c24104 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_369_18 := arithEq_of_rows h (row := 369) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_18
  simp only [← c24099, ← c24100, ← c24101] at e_369_18
  have e_369_19 := arithEq_of_rows h (row := 369) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_369_19
  simp only [← c24102, ← c24103, k2, ← c24104] at e_369_19
  have hr := e_369_19
  simp only [e_369_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 370 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 458)) := by
  have c24105 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.1
  have c24106 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24107 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24108 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24109 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24110 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_370_0 := arithEq_of_rows h (row := 370) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_0
  simp only [← c24105, ← c24106, ← c24107] at e_370_0
  have e_370_1 := arithEq_of_rows h (row := 370) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_1
  simp only [← c24108, ← c24109, k2, ← c24110] at e_370_1
  have hr := e_370_1
  simp only [e_370_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 370 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 459)) := by
  have c24111 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24112 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24113 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24114 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24115 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24116 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_370_2 := arithEq_of_rows h (row := 370) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_2
  simp only [← c24111, ← c24112, ← c24113] at e_370_2
  have e_370_3 := arithEq_of_rows h (row := 370) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_3
  simp only [← c24114, ← c24115, k2, ← c24116] at e_370_3
  have hr := e_370_3
  simp only [e_370_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 370 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 460)) := by
  have c24117 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24118 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24119 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24120 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24121 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24122 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_370_4 := arithEq_of_rows h (row := 370) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_4
  simp only [← c24117, ← c24118, ← c24119] at e_370_4
  have e_370_5 := arithEq_of_rows h (row := 370) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_5
  simp only [← c24120, ← c24121, k2, ← c24122] at e_370_5
  have hr := e_370_5
  simp only [e_370_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 370 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 461)) := by
  have c24123 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24124 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24125 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24126 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24127 := (publicBatchWrapper86_copies753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24128 := (publicBatchWrapper86_copies754 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_370_6 := arithEq_of_rows h (row := 370) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_6
  simp only [← c24123, ← c24124, ← c24125] at e_370_6
  have e_370_7 := arithEq_of_rows h (row := 370) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_7
  simp only [← c24126, ← c24127, k2, ← c24128] at e_370_7
  have hr := e_370_7
  simp only [e_370_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 370 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 462)) := by
  have c24129 := (publicBatchWrapper86_copies754 a h).2.1
  have c24130 := (publicBatchWrapper86_copies754 a h).2.2.1
  have c24131 := (publicBatchWrapper86_copies754 a h).2.2.2.1
  have c24132 := (publicBatchWrapper86_copies754 a h).2.2.2.2.1
  have c24133 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.1
  have c24134 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_370_8 := arithEq_of_rows h (row := 370) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_8
  simp only [← c24129, ← c24130, ← c24131] at e_370_8
  have e_370_9 := arithEq_of_rows h (row := 370) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_9
  simp only [← c24132, ← c24133, k2, ← c24134] at e_370_9
  have hr := e_370_9
  simp only [e_370_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 370 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 463)) := by
  have c24135 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.1
  have c24136 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.1
  have c24137 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.1
  have c24138 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24139 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24140 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_370_10 := arithEq_of_rows h (row := 370) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_10
  simp only [← c24135, ← c24136, ← c24137] at e_370_10
  have e_370_11 := arithEq_of_rows h (row := 370) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_11
  simp only [← c24138, ← c24139, k2, ← c24140] at e_370_11
  have hr := e_370_11
  simp only [e_370_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 370 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 464)) := by
  have c24141 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24142 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24143 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24144 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24145 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24146 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_370_12 := arithEq_of_rows h (row := 370) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_12
  simp only [← c24141, ← c24142, ← c24143] at e_370_12
  have e_370_13 := arithEq_of_rows h (row := 370) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_13
  simp only [← c24144, ← c24145, k2, ← c24146] at e_370_13
  have hr := e_370_13
  simp only [e_370_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 370 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 465)) := by
  have c24147 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24148 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24149 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24150 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24151 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24152 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_370_14 := arithEq_of_rows h (row := 370) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_14
  simp only [← c24147, ← c24148, ← c24149] at e_370_14
  have e_370_15 := arithEq_of_rows h (row := 370) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_15
  simp only [← c24150, ← c24151, k2, ← c24152] at e_370_15
  have hr := e_370_15
  simp only [e_370_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 370 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 466)) := by
  have c24153 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24154 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24155 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24156 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24157 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24158 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_370_16 := arithEq_of_rows h (row := 370) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_16
  simp only [← c24153, ← c24154, ← c24155] at e_370_16
  have e_370_17 := arithEq_of_rows h (row := 370) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_17
  simp only [← c24156, ← c24157, k2, ← c24158] at e_370_17
  have hr := e_370_17
  simp only [e_370_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 370 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 467)) := by
  have c24159 := (publicBatchWrapper86_copies754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24160 := (publicBatchWrapper86_copies755 a h).1
  have c24161 := (publicBatchWrapper86_copies755 a h).2.1
  have c24162 := (publicBatchWrapper86_copies755 a h).2.2.1
  have c24163 := (publicBatchWrapper86_copies755 a h).2.2.2.1
  have c24164 := (publicBatchWrapper86_copies755 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_370_18 := arithEq_of_rows h (row := 370) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_18
  simp only [← c24159, ← c24160, ← c24161] at e_370_18
  have e_370_19 := arithEq_of_rows h (row := 370) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_370_19
  simp only [← c24162, ← c24163, k2, ← c24164] at e_370_19
  have hr := e_370_19
  simp only [e_370_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 371 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 468)) := by
  have c24165 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.1
  have c24166 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.1
  have c24167 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.1
  have c24168 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.1
  have c24169 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.1
  have c24170 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_371_0 := arithEq_of_rows h (row := 371) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_0
  simp only [← c24165, ← c24166, ← c24167] at e_371_0
  have e_371_1 := arithEq_of_rows h (row := 371) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_1
  simp only [← c24168, ← c24169, k2, ← c24170] at e_371_1
  have hr := e_371_1
  simp only [e_371_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 371 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 469)) := by
  have c24171 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24172 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24173 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24174 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24175 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24176 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_371_2 := arithEq_of_rows h (row := 371) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_2
  simp only [← c24171, ← c24172, ← c24173] at e_371_2
  have e_371_3 := arithEq_of_rows h (row := 371) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_3
  simp only [← c24174, ← c24175, k2, ← c24176] at e_371_3
  have hr := e_371_3
  simp only [e_371_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 371 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 470)) := by
  have c24177 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24178 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24179 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24180 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24181 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24182 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_371_4 := arithEq_of_rows h (row := 371) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_4
  simp only [← c24177, ← c24178, ← c24179] at e_371_4
  have e_371_5 := arithEq_of_rows h (row := 371) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_5
  simp only [← c24180, ← c24181, k2, ← c24182] at e_371_5
  have hr := e_371_5
  simp only [e_371_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 371 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 471)) := by
  have c24183 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24184 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24185 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24186 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24187 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24188 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_371_6 := arithEq_of_rows h (row := 371) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_6
  simp only [← c24183, ← c24184, ← c24185] at e_371_6
  have e_371_7 := arithEq_of_rows h (row := 371) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_7
  simp only [← c24186, ← c24187, k2, ← c24188] at e_371_7
  have hr := e_371_7
  simp only [e_371_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 371 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 472)) := by
  have c24189 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24190 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24191 := (publicBatchWrapper86_copies755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24192 := (publicBatchWrapper86_copies756 a h).1
  have c24193 := (publicBatchWrapper86_copies756 a h).2.1
  have c24194 := (publicBatchWrapper86_copies756 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_371_8 := arithEq_of_rows h (row := 371) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_8
  simp only [← c24189, ← c24190, ← c24191] at e_371_8
  have e_371_9 := arithEq_of_rows h (row := 371) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_9
  simp only [← c24192, ← c24193, k2, ← c24194] at e_371_9
  have hr := e_371_9
  simp only [e_371_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 371 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 473)) := by
  have c24195 := (publicBatchWrapper86_copies756 a h).2.2.2.1
  have c24196 := (publicBatchWrapper86_copies756 a h).2.2.2.2.1
  have c24197 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.1
  have c24198 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.1
  have c24199 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.1
  have c24200 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_371_10 := arithEq_of_rows h (row := 371) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_10
  simp only [← c24195, ← c24196, ← c24197] at e_371_10
  have e_371_11 := arithEq_of_rows h (row := 371) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_11
  simp only [← c24198, ← c24199, k2, ← c24200] at e_371_11
  have hr := e_371_11
  simp only [e_371_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 371 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 474)) := by
  have c24201 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.1
  have c24202 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24203 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24204 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24205 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24206 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_371_12 := arithEq_of_rows h (row := 371) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_12
  simp only [← c24201, ← c24202, ← c24203] at e_371_12
  have e_371_13 := arithEq_of_rows h (row := 371) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_13
  simp only [← c24204, ← c24205, k2, ← c24206] at e_371_13
  have hr := e_371_13
  simp only [e_371_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 371 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 475)) := by
  have c24207 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24208 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24209 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24210 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24211 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24212 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_371_14 := arithEq_of_rows h (row := 371) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_14
  simp only [← c24207, ← c24208, ← c24209] at e_371_14
  have e_371_15 := arithEq_of_rows h (row := 371) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_15
  simp only [← c24210, ← c24211, k2, ← c24212] at e_371_15
  have hr := e_371_15
  simp only [e_371_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 371 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 476)) := by
  have c24213 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24214 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24215 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24216 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24217 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24218 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_371_16 := arithEq_of_rows h (row := 371) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_16
  simp only [← c24213, ← c24214, ← c24215] at e_371_16
  have e_371_17 := arithEq_of_rows h (row := 371) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_17
  simp only [← c24216, ← c24217, k2, ← c24218] at e_371_17
  have hr := e_371_17
  simp only [e_371_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 371 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 477)) := by
  have c24219 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24220 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24221 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24222 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24223 := (publicBatchWrapper86_copies756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24224 := (publicBatchWrapper86_copies757 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_371_18 := arithEq_of_rows h (row := 371) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_18
  simp only [← c24219, ← c24220, ← c24221] at e_371_18
  have e_371_19 := arithEq_of_rows h (row := 371) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_371_19
  simp only [← c24222, ← c24223, k2, ← c24224] at e_371_19
  have hr := e_371_19
  simp only [e_371_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 372 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 478)) := by
  have c24225 := (publicBatchWrapper86_copies757 a h).2.1
  have c24226 := (publicBatchWrapper86_copies757 a h).2.2.1
  have c24227 := (publicBatchWrapper86_copies757 a h).2.2.2.1
  have c24228 := (publicBatchWrapper86_copies757 a h).2.2.2.2.1
  have c24229 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.1
  have c24230 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_372_0 := arithEq_of_rows h (row := 372) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_0
  simp only [← c24225, ← c24226, ← c24227] at e_372_0
  have e_372_1 := arithEq_of_rows h (row := 372) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_1
  simp only [← c24228, ← c24229, k2, ← c24230] at e_372_1
  have hr := e_372_1
  simp only [e_372_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 372 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 479)) := by
  have c24231 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.1
  have c24232 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.1
  have c24233 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.1
  have c24234 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24235 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24236 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_372_2 := arithEq_of_rows h (row := 372) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_2
  simp only [← c24231, ← c24232, ← c24233] at e_372_2
  have e_372_3 := arithEq_of_rows h (row := 372) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_3
  simp only [← c24234, ← c24235, k2, ← c24236] at e_372_3
  have hr := e_372_3
  simp only [e_372_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 372 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 480)) := by
  have c24237 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24238 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24239 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24240 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24241 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24242 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_372_4 := arithEq_of_rows h (row := 372) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_4
  simp only [← c24237, ← c24238, ← c24239] at e_372_4
  have e_372_5 := arithEq_of_rows h (row := 372) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_5
  simp only [← c24240, ← c24241, k2, ← c24242] at e_372_5
  have hr := e_372_5
  simp only [e_372_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 372 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 481)) := by
  have c24243 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24244 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24245 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24246 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24247 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24248 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_372_6 := arithEq_of_rows h (row := 372) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_6
  simp only [← c24243, ← c24244, ← c24245] at e_372_6
  have e_372_7 := arithEq_of_rows h (row := 372) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_7
  simp only [← c24246, ← c24247, k2, ← c24248] at e_372_7
  have hr := e_372_7
  simp only [e_372_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 372 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 482)) := by
  have c24249 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24250 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24251 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24252 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24253 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24254 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_372_8 := arithEq_of_rows h (row := 372) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_8
  simp only [← c24249, ← c24250, ← c24251] at e_372_8
  have e_372_9 := arithEq_of_rows h (row := 372) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_9
  simp only [← c24252, ← c24253, k2, ← c24254] at e_372_9
  have hr := e_372_9
  simp only [e_372_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 372 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 483)) := by
  have c24255 := (publicBatchWrapper86_copies757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24256 := (publicBatchWrapper86_copies758 a h).1
  have c24257 := (publicBatchWrapper86_copies758 a h).2.1
  have c24258 := (publicBatchWrapper86_copies758 a h).2.2.1
  have c24259 := (publicBatchWrapper86_copies758 a h).2.2.2.1
  have c24260 := (publicBatchWrapper86_copies758 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_372_10 := arithEq_of_rows h (row := 372) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_10
  simp only [← c24255, ← c24256, ← c24257] at e_372_10
  have e_372_11 := arithEq_of_rows h (row := 372) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_11
  simp only [← c24258, ← c24259, k2, ← c24260] at e_372_11
  have hr := e_372_11
  simp only [e_372_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 372 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 484)) := by
  have c24261 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.1
  have c24262 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.1
  have c24263 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.1
  have c24264 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.1
  have c24265 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.1
  have c24266 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_372_12 := arithEq_of_rows h (row := 372) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_12
  simp only [← c24261, ← c24262, ← c24263] at e_372_12
  have e_372_13 := arithEq_of_rows h (row := 372) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_13
  simp only [← c24264, ← c24265, k2, ← c24266] at e_372_13
  have hr := e_372_13
  simp only [e_372_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 372 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 485)) := by
  have c24267 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24268 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24269 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24270 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24271 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24272 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_372_14 := arithEq_of_rows h (row := 372) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_14
  simp only [← c24267, ← c24268, ← c24269] at e_372_14
  have e_372_15 := arithEq_of_rows h (row := 372) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_15
  simp only [← c24270, ← c24271, k2, ← c24272] at e_372_15
  have hr := e_372_15
  simp only [e_372_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 372 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 486)) := by
  have c24273 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24274 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24275 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24276 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24277 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24278 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_372_16 := arithEq_of_rows h (row := 372) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_16
  simp only [← c24273, ← c24274, ← c24275] at e_372_16
  have e_372_17 := arithEq_of_rows h (row := 372) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_17
  simp only [← c24276, ← c24277, k2, ← c24278] at e_372_17
  have hr := e_372_17
  simp only [e_372_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 372 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 487)) := by
  have c24279 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24280 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24281 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24282 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24283 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24284 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_372_18 := arithEq_of_rows h (row := 372) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_18
  simp only [← c24279, ← c24280, ← c24281] at e_372_18
  have e_372_19 := arithEq_of_rows h (row := 372) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_372_19
  simp only [← c24282, ← c24283, k2, ← c24284] at e_372_19
  have hr := e_372_19
  simp only [e_372_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 373 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 488)) := by
  have c24285 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24286 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24287 := (publicBatchWrapper86_copies758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24288 := (publicBatchWrapper86_copies759 a h).1
  have c24289 := (publicBatchWrapper86_copies759 a h).2.1
  have c24290 := (publicBatchWrapper86_copies759 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_373_0 := arithEq_of_rows h (row := 373) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_0
  simp only [← c24285, ← c24286, ← c24287] at e_373_0
  have e_373_1 := arithEq_of_rows h (row := 373) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_1
  simp only [← c24288, ← c24289, k2, ← c24290] at e_373_1
  have hr := e_373_1
  simp only [e_373_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 373 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 489)) := by
  have c24291 := (publicBatchWrapper86_copies759 a h).2.2.2.1
  have c24292 := (publicBatchWrapper86_copies759 a h).2.2.2.2.1
  have c24293 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.1
  have c24294 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.1
  have c24295 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.1
  have c24296 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_373_2 := arithEq_of_rows h (row := 373) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_2
  simp only [← c24291, ← c24292, ← c24293] at e_373_2
  have e_373_3 := arithEq_of_rows h (row := 373) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_3
  simp only [← c24294, ← c24295, k2, ← c24296] at e_373_3
  have hr := e_373_3
  simp only [e_373_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 373 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 490)) := by
  have c24297 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.1
  have c24298 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24299 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24300 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24301 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24302 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_373_4 := arithEq_of_rows h (row := 373) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_4
  simp only [← c24297, ← c24298, ← c24299] at e_373_4
  have e_373_5 := arithEq_of_rows h (row := 373) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_5
  simp only [← c24300, ← c24301, k2, ← c24302] at e_373_5
  have hr := e_373_5
  simp only [e_373_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 373 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 491)) := by
  have c24303 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24304 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24305 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24306 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24307 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24308 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_373_6 := arithEq_of_rows h (row := 373) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_6
  simp only [← c24303, ← c24304, ← c24305] at e_373_6
  have e_373_7 := arithEq_of_rows h (row := 373) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_7
  simp only [← c24306, ← c24307, k2, ← c24308] at e_373_7
  have hr := e_373_7
  simp only [e_373_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 373 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 492)) := by
  have c24309 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24310 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24311 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24312 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24313 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24314 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_373_8 := arithEq_of_rows h (row := 373) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_8
  simp only [← c24309, ← c24310, ← c24311] at e_373_8
  have e_373_9 := arithEq_of_rows h (row := 373) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_9
  simp only [← c24312, ← c24313, k2, ← c24314] at e_373_9
  have hr := e_373_9
  simp only [e_373_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 373 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 493)) := by
  have c24315 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24316 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24317 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24318 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24319 := (publicBatchWrapper86_copies759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24320 := (publicBatchWrapper86_copies760 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_373_10 := arithEq_of_rows h (row := 373) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_10
  simp only [← c24315, ← c24316, ← c24317] at e_373_10
  have e_373_11 := arithEq_of_rows h (row := 373) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_11
  simp only [← c24318, ← c24319, k2, ← c24320] at e_373_11
  have hr := e_373_11
  simp only [e_373_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 373 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 494)) := by
  have c24321 := (publicBatchWrapper86_copies760 a h).2.1
  have c24322 := (publicBatchWrapper86_copies760 a h).2.2.1
  have c24323 := (publicBatchWrapper86_copies760 a h).2.2.2.1
  have c24324 := (publicBatchWrapper86_copies760 a h).2.2.2.2.1
  have c24325 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.1
  have c24326 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_373_12 := arithEq_of_rows h (row := 373) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_12
  simp only [← c24321, ← c24322, ← c24323] at e_373_12
  have e_373_13 := arithEq_of_rows h (row := 373) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_13
  simp only [← c24324, ← c24325, k2, ← c24326] at e_373_13
  have hr := e_373_13
  simp only [e_373_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 373 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 495)) := by
  have c24327 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.1
  have c24328 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.1
  have c24329 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.1
  have c24330 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24331 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24332 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_373_14 := arithEq_of_rows h (row := 373) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_14
  simp only [← c24327, ← c24328, ← c24329] at e_373_14
  have e_373_15 := arithEq_of_rows h (row := 373) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_15
  simp only [← c24330, ← c24331, k2, ← c24332] at e_373_15
  have hr := e_373_15
  simp only [e_373_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 373 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 496)) := by
  have c24333 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24334 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24335 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24336 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24337 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24338 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_373_16 := arithEq_of_rows h (row := 373) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_16
  simp only [← c24333, ← c24334, ← c24335] at e_373_16
  have e_373_17 := arithEq_of_rows h (row := 373) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_17
  simp only [← c24336, ← c24337, k2, ← c24338] at e_373_17
  have hr := e_373_17
  simp only [e_373_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 373 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 497)) := by
  have c24339 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24340 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24341 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24342 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24343 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24344 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_373_18 := arithEq_of_rows h (row := 373) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_18
  simp only [← c24339, ← c24340, ← c24341] at e_373_18
  have e_373_19 := arithEq_of_rows h (row := 373) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_373_19
  simp only [← c24342, ← c24343, k2, ← c24344] at e_373_19
  have hr := e_373_19
  simp only [e_373_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 374 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 498)) := by
  have c24345 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24346 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24347 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24348 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24349 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24350 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_374_0 := arithEq_of_rows h (row := 374) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_0
  simp only [← c24345, ← c24346, ← c24347] at e_374_0
  have e_374_1 := arithEq_of_rows h (row := 374) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_1
  simp only [← c24348, ← c24349, k2, ← c24350] at e_374_1
  have hr := e_374_1
  simp only [e_374_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 374 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 499)) := by
  have c24351 := (publicBatchWrapper86_copies760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24352 := (publicBatchWrapper86_copies761 a h).1
  have c24353 := (publicBatchWrapper86_copies761 a h).2.1
  have c24354 := (publicBatchWrapper86_copies761 a h).2.2.1
  have c24355 := (publicBatchWrapper86_copies761 a h).2.2.2.1
  have c24356 := (publicBatchWrapper86_copies761 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_374_2 := arithEq_of_rows h (row := 374) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_2
  simp only [← c24351, ← c24352, ← c24353] at e_374_2
  have e_374_3 := arithEq_of_rows h (row := 374) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_3
  simp only [← c24354, ← c24355, k2, ← c24356] at e_374_3
  have hr := e_374_3
  simp only [e_374_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 374 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 500)) := by
  have c24357 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.1
  have c24358 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.1
  have c24359 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.1
  have c24360 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.1
  have c24361 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.1
  have c24362 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_374_4 := arithEq_of_rows h (row := 374) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_4
  simp only [← c24357, ← c24358, ← c24359] at e_374_4
  have e_374_5 := arithEq_of_rows h (row := 374) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_5
  simp only [← c24360, ← c24361, k2, ← c24362] at e_374_5
  have hr := e_374_5
  simp only [e_374_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 374 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 501)) := by
  have c24363 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24364 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24365 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24366 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24367 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24368 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_374_6 := arithEq_of_rows h (row := 374) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_6
  simp only [← c24363, ← c24364, ← c24365] at e_374_6
  have e_374_7 := arithEq_of_rows h (row := 374) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_7
  simp only [← c24366, ← c24367, k2, ← c24368] at e_374_7
  have hr := e_374_7
  simp only [e_374_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 374 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 502)) := by
  have c24369 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24370 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24371 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24372 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24373 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24374 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_374_8 := arithEq_of_rows h (row := 374) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_8
  simp only [← c24369, ← c24370, ← c24371] at e_374_8
  have e_374_9 := arithEq_of_rows h (row := 374) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_9
  simp only [← c24372, ← c24373, k2, ← c24374] at e_374_9
  have hr := e_374_9
  simp only [e_374_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 374 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 503)) := by
  have c24375 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24376 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24377 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24378 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24379 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24380 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_374_10 := arithEq_of_rows h (row := 374) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_10
  simp only [← c24375, ← c24376, ← c24377] at e_374_10
  have e_374_11 := arithEq_of_rows h (row := 374) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_11
  simp only [← c24378, ← c24379, k2, ← c24380] at e_374_11
  have hr := e_374_11
  simp only [e_374_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 374 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 504)) := by
  have c24381 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24382 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24383 := (publicBatchWrapper86_copies761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24384 := (publicBatchWrapper86_copies762 a h).1
  have c24385 := (publicBatchWrapper86_copies762 a h).2.1
  have c24386 := (publicBatchWrapper86_copies762 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_374_12 := arithEq_of_rows h (row := 374) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_12
  simp only [← c24381, ← c24382, ← c24383] at e_374_12
  have e_374_13 := arithEq_of_rows h (row := 374) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_13
  simp only [← c24384, ← c24385, k2, ← c24386] at e_374_13
  have hr := e_374_13
  simp only [e_374_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 374 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 505)) := by
  have c24387 := (publicBatchWrapper86_copies762 a h).2.2.2.1
  have c24388 := (publicBatchWrapper86_copies762 a h).2.2.2.2.1
  have c24389 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.1
  have c24390 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.1
  have c24391 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.1
  have c24392 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_374_14 := arithEq_of_rows h (row := 374) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_14
  simp only [← c24387, ← c24388, ← c24389] at e_374_14
  have e_374_15 := arithEq_of_rows h (row := 374) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_15
  simp only [← c24390, ← c24391, k2, ← c24392] at e_374_15
  have hr := e_374_15
  simp only [e_374_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 374 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 506)) := by
  have c24393 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.1
  have c24394 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24395 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24396 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24397 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24398 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_374_16 := arithEq_of_rows h (row := 374) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_16
  simp only [← c24393, ← c24394, ← c24395] at e_374_16
  have e_374_17 := arithEq_of_rows h (row := 374) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_17
  simp only [← c24396, ← c24397, k2, ← c24398] at e_374_17
  have hr := e_374_17
  simp only [e_374_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 374 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 507)) := by
  have c24399 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24400 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24401 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24402 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24403 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24404 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_374_18 := arithEq_of_rows h (row := 374) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_18
  simp only [← c24399, ← c24400, ← c24401] at e_374_18
  have e_374_19 := arithEq_of_rows h (row := 374) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_374_19
  simp only [← c24402, ← c24403, k2, ← c24404] at e_374_19
  have hr := e_374_19
  simp only [e_374_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 375 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 508)) := by
  have c24405 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24406 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24407 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24408 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24409 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24410 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_375_0 := arithEq_of_rows h (row := 375) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_0
  simp only [← c24405, ← c24406, ← c24407] at e_375_0
  have e_375_1 := arithEq_of_rows h (row := 375) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_1
  simp only [← c24408, ← c24409, k2, ← c24410] at e_375_1
  have hr := e_375_1
  simp only [e_375_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 375 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 509)) := by
  have c24411 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24412 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24413 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24414 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24415 := (publicBatchWrapper86_copies762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24416 := (publicBatchWrapper86_copies763 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_375_2 := arithEq_of_rows h (row := 375) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_2
  simp only [← c24411, ← c24412, ← c24413] at e_375_2
  have e_375_3 := arithEq_of_rows h (row := 375) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_3
  simp only [← c24414, ← c24415, k2, ← c24416] at e_375_3
  have hr := e_375_3
  simp only [e_375_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 375 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 510)) := by
  have c24417 := (publicBatchWrapper86_copies763 a h).2.1
  have c24418 := (publicBatchWrapper86_copies763 a h).2.2.1
  have c24419 := (publicBatchWrapper86_copies763 a h).2.2.2.1
  have c24420 := (publicBatchWrapper86_copies763 a h).2.2.2.2.1
  have c24421 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.1
  have c24422 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_375_4 := arithEq_of_rows h (row := 375) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_4
  simp only [← c24417, ← c24418, ← c24419] at e_375_4
  have e_375_5 := arithEq_of_rows h (row := 375) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_5
  simp only [← c24420, ← c24421, k2, ← c24422] at e_375_5
  have hr := e_375_5
  simp only [e_375_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 375 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 511)) := by
  have c24423 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.1
  have c24424 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.1
  have c24425 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.1
  have c24426 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24427 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24428 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_375_6 := arithEq_of_rows h (row := 375) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_6
  simp only [← c24423, ← c24424, ← c24425] at e_375_6
  have e_375_7 := arithEq_of_rows h (row := 375) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_7
  simp only [← c24426, ← c24427, k2, ← c24428] at e_375_7
  have hr := e_375_7
  simp only [e_375_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 375 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 512)) := by
  have c24429 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24430 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24431 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24432 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24433 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24434 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_375_8 := arithEq_of_rows h (row := 375) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_8
  simp only [← c24429, ← c24430, ← c24431] at e_375_8
  have e_375_9 := arithEq_of_rows h (row := 375) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_9
  simp only [← c24432, ← c24433, k2, ← c24434] at e_375_9
  have hr := e_375_9
  simp only [e_375_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 375 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 513)) := by
  have c24435 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24436 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24437 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24438 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24439 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24440 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_375_10 := arithEq_of_rows h (row := 375) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_10
  simp only [← c24435, ← c24436, ← c24437] at e_375_10
  have e_375_11 := arithEq_of_rows h (row := 375) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_11
  simp only [← c24438, ← c24439, k2, ← c24440] at e_375_11
  have hr := e_375_11
  simp only [e_375_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 375 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 514)) := by
  have c24441 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24442 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24443 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24444 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24445 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24446 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_375_12 := arithEq_of_rows h (row := 375) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_12
  simp only [← c24441, ← c24442, ← c24443] at e_375_12
  have e_375_13 := arithEq_of_rows h (row := 375) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_13
  simp only [← c24444, ← c24445, k2, ← c24446] at e_375_13
  have hr := e_375_13
  simp only [e_375_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 375 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 515)) := by
  have c24447 := (publicBatchWrapper86_copies763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24448 := (publicBatchWrapper86_copies764 a h).1
  have c24449 := (publicBatchWrapper86_copies764 a h).2.1
  have c24450 := (publicBatchWrapper86_copies764 a h).2.2.1
  have c24451 := (publicBatchWrapper86_copies764 a h).2.2.2.1
  have c24452 := (publicBatchWrapper86_copies764 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_375_14 := arithEq_of_rows h (row := 375) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_14
  simp only [← c24447, ← c24448, ← c24449] at e_375_14
  have e_375_15 := arithEq_of_rows h (row := 375) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_15
  simp only [← c24450, ← c24451, k2, ← c24452] at e_375_15
  have hr := e_375_15
  simp only [e_375_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 375 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 516)) := by
  have c24453 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.1
  have c24454 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.1
  have c24455 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.1
  have c24456 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.1
  have c24457 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.1
  have c24458 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_375_16 := arithEq_of_rows h (row := 375) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_16
  simp only [← c24453, ← c24454, ← c24455] at e_375_16
  have e_375_17 := arithEq_of_rows h (row := 375) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_17
  simp only [← c24456, ← c24457, k2, ← c24458] at e_375_17
  have hr := e_375_17
  simp only [e_375_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 375 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 517)) := by
  have c24459 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24460 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24461 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24462 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24463 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24464 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_375_18 := arithEq_of_rows h (row := 375) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_18
  simp only [← c24459, ← c24460, ← c24461] at e_375_18
  have e_375_19 := arithEq_of_rows h (row := 375) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_375_19
  simp only [← c24462, ← c24463, k2, ← c24464] at e_375_19
  have hr := e_375_19
  simp only [e_375_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 376 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 518)) := by
  have c24465 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24466 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24467 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24468 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24469 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24470 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_376_0 := arithEq_of_rows h (row := 376) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_0
  simp only [← c24465, ← c24466, ← c24467] at e_376_0
  have e_376_1 := arithEq_of_rows h (row := 376) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_1
  simp only [← c24468, ← c24469, k2, ← c24470] at e_376_1
  have hr := e_376_1
  simp only [e_376_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 376 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 519)) := by
  have c24471 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24472 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24473 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24474 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24475 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24476 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_376_2 := arithEq_of_rows h (row := 376) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_2
  simp only [← c24471, ← c24472, ← c24473] at e_376_2
  have e_376_3 := arithEq_of_rows h (row := 376) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_3
  simp only [← c24474, ← c24475, k2, ← c24476] at e_376_3
  have hr := e_376_3
  simp only [e_376_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 376 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 520)) := by
  have c24477 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24478 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24479 := (publicBatchWrapper86_copies764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24480 := (publicBatchWrapper86_copies765 a h).1
  have c24481 := (publicBatchWrapper86_copies765 a h).2.1
  have c24482 := (publicBatchWrapper86_copies765 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_376_4 := arithEq_of_rows h (row := 376) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_4
  simp only [← c24477, ← c24478, ← c24479] at e_376_4
  have e_376_5 := arithEq_of_rows h (row := 376) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_5
  simp only [← c24480, ← c24481, k2, ← c24482] at e_376_5
  have hr := e_376_5
  simp only [e_376_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 376 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 521)) := by
  have c24483 := (publicBatchWrapper86_copies765 a h).2.2.2.1
  have c24484 := (publicBatchWrapper86_copies765 a h).2.2.2.2.1
  have c24485 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.1
  have c24486 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.1
  have c24487 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.1
  have c24488 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_376_6 := arithEq_of_rows h (row := 376) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_6
  simp only [← c24483, ← c24484, ← c24485] at e_376_6
  have e_376_7 := arithEq_of_rows h (row := 376) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_7
  simp only [← c24486, ← c24487, k2, ← c24488] at e_376_7
  have hr := e_376_7
  simp only [e_376_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 376 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 522)) := by
  have c24489 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.1
  have c24490 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24491 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24492 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24493 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24494 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_376_8 := arithEq_of_rows h (row := 376) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_8
  simp only [← c24489, ← c24490, ← c24491] at e_376_8
  have e_376_9 := arithEq_of_rows h (row := 376) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_9
  simp only [← c24492, ← c24493, k2, ← c24494] at e_376_9
  have hr := e_376_9
  simp only [e_376_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 376 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 523)) := by
  have c24495 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24496 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24497 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24498 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24499 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24500 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_376_10 := arithEq_of_rows h (row := 376) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_10
  simp only [← c24495, ← c24496, ← c24497] at e_376_10
  have e_376_11 := arithEq_of_rows h (row := 376) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_11
  simp only [← c24498, ← c24499, k2, ← c24500] at e_376_11
  have hr := e_376_11
  simp only [e_376_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 376 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 524)) := by
  have c24501 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24502 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24503 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24504 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24505 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24506 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_376_12 := arithEq_of_rows h (row := 376) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_12
  simp only [← c24501, ← c24502, ← c24503] at e_376_12
  have e_376_13 := arithEq_of_rows h (row := 376) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_13
  simp only [← c24504, ← c24505, k2, ← c24506] at e_376_13
  have hr := e_376_13
  simp only [e_376_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 376 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 525)) := by
  have c24507 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24508 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24509 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24510 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24511 := (publicBatchWrapper86_copies765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24512 := (publicBatchWrapper86_copies766 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_376_14 := arithEq_of_rows h (row := 376) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_14
  simp only [← c24507, ← c24508, ← c24509] at e_376_14
  have e_376_15 := arithEq_of_rows h (row := 376) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_15
  simp only [← c24510, ← c24511, k2, ← c24512] at e_376_15
  have hr := e_376_15
  simp only [e_376_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 376 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 526)) := by
  have c24513 := (publicBatchWrapper86_copies766 a h).2.1
  have c24514 := (publicBatchWrapper86_copies766 a h).2.2.1
  have c24515 := (publicBatchWrapper86_copies766 a h).2.2.2.1
  have c24516 := (publicBatchWrapper86_copies766 a h).2.2.2.2.1
  have c24517 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.1
  have c24518 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_376_16 := arithEq_of_rows h (row := 376) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_16
  simp only [← c24513, ← c24514, ← c24515] at e_376_16
  have e_376_17 := arithEq_of_rows h (row := 376) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_17
  simp only [← c24516, ← c24517, k2, ← c24518] at e_376_17
  have hr := e_376_17
  simp only [e_376_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 376 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 527)) := by
  have c24519 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.1
  have c24520 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.1
  have c24521 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.1
  have c24522 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24523 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24524 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_376_18 := arithEq_of_rows h (row := 376) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_18
  simp only [← c24519, ← c24520, ← c24521] at e_376_18
  have e_376_19 := arithEq_of_rows h (row := 376) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_376_19
  simp only [← c24522, ← c24523, k2, ← c24524] at e_376_19
  have hr := e_376_19
  simp only [e_376_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 377 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 528)) := by
  have c24525 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24526 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24527 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24528 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24529 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24530 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_377_0 := arithEq_of_rows h (row := 377) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_0
  simp only [← c24525, ← c24526, ← c24527] at e_377_0
  have e_377_1 := arithEq_of_rows h (row := 377) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_1
  simp only [← c24528, ← c24529, k2, ← c24530] at e_377_1
  have hr := e_377_1
  simp only [e_377_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 377 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 529)) := by
  have c24531 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24532 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24533 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24534 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24535 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24536 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_377_2 := arithEq_of_rows h (row := 377) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_2
  simp only [← c24531, ← c24532, ← c24533] at e_377_2
  have e_377_3 := arithEq_of_rows h (row := 377) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_3
  simp only [← c24534, ← c24535, k2, ← c24536] at e_377_3
  have hr := e_377_3
  simp only [e_377_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 377 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 530)) := by
  have c24537 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24538 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24539 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24540 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24541 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24542 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_377_4 := arithEq_of_rows h (row := 377) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_4
  simp only [← c24537, ← c24538, ← c24539] at e_377_4
  have e_377_5 := arithEq_of_rows h (row := 377) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_5
  simp only [← c24540, ← c24541, k2, ← c24542] at e_377_5
  have hr := e_377_5
  simp only [e_377_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 377 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 531)) := by
  have c24543 := (publicBatchWrapper86_copies766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24544 := (publicBatchWrapper86_copies767 a h).1
  have c24545 := (publicBatchWrapper86_copies767 a h).2.1
  have c24546 := (publicBatchWrapper86_copies767 a h).2.2.1
  have c24547 := (publicBatchWrapper86_copies767 a h).2.2.2.1
  have c24548 := (publicBatchWrapper86_copies767 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_377_6 := arithEq_of_rows h (row := 377) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_6
  simp only [← c24543, ← c24544, ← c24545] at e_377_6
  have e_377_7 := arithEq_of_rows h (row := 377) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_7
  simp only [← c24546, ← c24547, k2, ← c24548] at e_377_7
  have hr := e_377_7
  simp only [e_377_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 377 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 532)) := by
  have c24549 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.1
  have c24550 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.1
  have c24551 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.1
  have c24552 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.1
  have c24553 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.1
  have c24554 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_377_8 := arithEq_of_rows h (row := 377) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_8
  simp only [← c24549, ← c24550, ← c24551] at e_377_8
  have e_377_9 := arithEq_of_rows h (row := 377) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_9
  simp only [← c24552, ← c24553, k2, ← c24554] at e_377_9
  have hr := e_377_9
  simp only [e_377_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 377 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 533)) := by
  have c24555 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24556 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24557 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24558 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24559 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24560 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_377_10 := arithEq_of_rows h (row := 377) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_10
  simp only [← c24555, ← c24556, ← c24557] at e_377_10
  have e_377_11 := arithEq_of_rows h (row := 377) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_11
  simp only [← c24558, ← c24559, k2, ← c24560] at e_377_11
  have hr := e_377_11
  simp only [e_377_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 377 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 534)) := by
  have c24561 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24562 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24563 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24564 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24565 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24566 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_377_12 := arithEq_of_rows h (row := 377) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_12
  simp only [← c24561, ← c24562, ← c24563] at e_377_12
  have e_377_13 := arithEq_of_rows h (row := 377) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_13
  simp only [← c24564, ← c24565, k2, ← c24566] at e_377_13
  have hr := e_377_13
  simp only [e_377_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 377 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 535)) := by
  have c24567 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24568 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24569 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24570 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24571 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24572 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_377_14 := arithEq_of_rows h (row := 377) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_14
  simp only [← c24567, ← c24568, ← c24569] at e_377_14
  have e_377_15 := arithEq_of_rows h (row := 377) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_15
  simp only [← c24570, ← c24571, k2, ← c24572] at e_377_15
  have hr := e_377_15
  simp only [e_377_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 377 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 536)) := by
  have c24573 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24574 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24575 := (publicBatchWrapper86_copies767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24576 := (publicBatchWrapper86_copies768 a h).1
  have c24577 := (publicBatchWrapper86_copies768 a h).2.1
  have c24578 := (publicBatchWrapper86_copies768 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_377_16 := arithEq_of_rows h (row := 377) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_16
  simp only [← c24573, ← c24574, ← c24575] at e_377_16
  have e_377_17 := arithEq_of_rows h (row := 377) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_17
  simp only [← c24576, ← c24577, k2, ← c24578] at e_377_17
  have hr := e_377_17
  simp only [e_377_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 377 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 537)) := by
  have c24579 := (publicBatchWrapper86_copies768 a h).2.2.2.1
  have c24580 := (publicBatchWrapper86_copies768 a h).2.2.2.2.1
  have c24581 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.1
  have c24582 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.1
  have c24583 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.1
  have c24584 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_377_18 := arithEq_of_rows h (row := 377) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_18
  simp only [← c24579, ← c24580, ← c24581] at e_377_18
  have e_377_19 := arithEq_of_rows h (row := 377) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_377_19
  simp only [← c24582, ← c24583, k2, ← c24584] at e_377_19
  have hr := e_377_19
  simp only [e_377_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 378 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 538)) := by
  have c24585 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.1
  have c24586 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24587 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24588 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24589 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24590 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_378_0 := arithEq_of_rows h (row := 378) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_0
  simp only [← c24585, ← c24586, ← c24587] at e_378_0
  have e_378_1 := arithEq_of_rows h (row := 378) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_1
  simp only [← c24588, ← c24589, k2, ← c24590] at e_378_1
  have hr := e_378_1
  simp only [e_378_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 378 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 539)) := by
  have c24591 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24592 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24593 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24594 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24595 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24596 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_378_2 := arithEq_of_rows h (row := 378) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_2
  simp only [← c24591, ← c24592, ← c24593] at e_378_2
  have e_378_3 := arithEq_of_rows h (row := 378) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_3
  simp only [← c24594, ← c24595, k2, ← c24596] at e_378_3
  have hr := e_378_3
  simp only [e_378_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 378 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 540)) := by
  have c24597 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24598 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24599 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24600 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24601 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24602 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_378_4 := arithEq_of_rows h (row := 378) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_4
  simp only [← c24597, ← c24598, ← c24599] at e_378_4
  have e_378_5 := arithEq_of_rows h (row := 378) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_5
  simp only [← c24600, ← c24601, k2, ← c24602] at e_378_5
  have hr := e_378_5
  simp only [e_378_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 378 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 541)) := by
  have c24603 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24604 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24605 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24606 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24607 := (publicBatchWrapper86_copies768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24608 := (publicBatchWrapper86_copies769 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_378_6 := arithEq_of_rows h (row := 378) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_6
  simp only [← c24603, ← c24604, ← c24605] at e_378_6
  have e_378_7 := arithEq_of_rows h (row := 378) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_7
  simp only [← c24606, ← c24607, k2, ← c24608] at e_378_7
  have hr := e_378_7
  simp only [e_378_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 378 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 542)) := by
  have c24609 := (publicBatchWrapper86_copies769 a h).2.1
  have c24610 := (publicBatchWrapper86_copies769 a h).2.2.1
  have c24611 := (publicBatchWrapper86_copies769 a h).2.2.2.1
  have c24612 := (publicBatchWrapper86_copies769 a h).2.2.2.2.1
  have c24613 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.1
  have c24614 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_378_8 := arithEq_of_rows h (row := 378) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_8
  simp only [← c24609, ← c24610, ← c24611] at e_378_8
  have e_378_9 := arithEq_of_rows h (row := 378) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_9
  simp only [← c24612, ← c24613, k2, ← c24614] at e_378_9
  have hr := e_378_9
  simp only [e_378_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 378 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 543)) := by
  have c24615 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.1
  have c24616 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.1
  have c24617 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.1
  have c24618 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24619 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24620 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_378_10 := arithEq_of_rows h (row := 378) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_10
  simp only [← c24615, ← c24616, ← c24617] at e_378_10
  have e_378_11 := arithEq_of_rows h (row := 378) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_11
  simp only [← c24618, ← c24619, k2, ← c24620] at e_378_11
  have hr := e_378_11
  simp only [e_378_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 378 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 544)) := by
  have c24621 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24622 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24623 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24624 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24625 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24626 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_378_12 := arithEq_of_rows h (row := 378) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_12
  simp only [← c24621, ← c24622, ← c24623] at e_378_12
  have e_378_13 := arithEq_of_rows h (row := 378) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_13
  simp only [← c24624, ← c24625, k2, ← c24626] at e_378_13
  have hr := e_378_13
  simp only [e_378_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 378 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 545)) := by
  have c24627 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24628 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24629 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24630 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24631 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24632 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_378_14 := arithEq_of_rows h (row := 378) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_14
  simp only [← c24627, ← c24628, ← c24629] at e_378_14
  have e_378_15 := arithEq_of_rows h (row := 378) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_15
  simp only [← c24630, ← c24631, k2, ← c24632] at e_378_15
  have hr := e_378_15
  simp only [e_378_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 378 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 546)) := by
  have c24633 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24634 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24635 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24636 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24637 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24638 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_378_16 := arithEq_of_rows h (row := 378) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_16
  simp only [← c24633, ← c24634, ← c24635] at e_378_16
  have e_378_17 := arithEq_of_rows h (row := 378) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_17
  simp only [← c24636, ← c24637, k2, ← c24638] at e_378_17
  have hr := e_378_17
  simp only [e_378_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 378 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 547)) := by
  have c24639 := (publicBatchWrapper86_copies769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24640 := (publicBatchWrapper86_copies770 a h).1
  have c24641 := (publicBatchWrapper86_copies770 a h).2.1
  have c24642 := (publicBatchWrapper86_copies770 a h).2.2.1
  have c24643 := (publicBatchWrapper86_copies770 a h).2.2.2.1
  have c24644 := (publicBatchWrapper86_copies770 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_378_18 := arithEq_of_rows h (row := 378) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_18
  simp only [← c24639, ← c24640, ← c24641] at e_378_18
  have e_378_19 := arithEq_of_rows h (row := 378) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_378_19
  simp only [← c24642, ← c24643, k2, ← c24644] at e_378_19
  have hr := e_378_19
  simp only [e_378_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 379 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 548)) := by
  have c24645 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.1
  have c24646 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.1
  have c24647 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.1
  have c24648 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.1
  have c24649 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.1
  have c24650 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_379_0 := arithEq_of_rows h (row := 379) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_0
  simp only [← c24645, ← c24646, ← c24647] at e_379_0
  have e_379_1 := arithEq_of_rows h (row := 379) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_1
  simp only [← c24648, ← c24649, k2, ← c24650] at e_379_1
  have hr := e_379_1
  simp only [e_379_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 379 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 549)) := by
  have c24651 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24652 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24653 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24654 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24655 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24656 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_379_2 := arithEq_of_rows h (row := 379) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_2
  simp only [← c24651, ← c24652, ← c24653] at e_379_2
  have e_379_3 := arithEq_of_rows h (row := 379) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_3
  simp only [← c24654, ← c24655, k2, ← c24656] at e_379_3
  have hr := e_379_3
  simp only [e_379_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 379 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 550)) := by
  have c24657 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24658 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24659 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24660 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24661 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24662 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_379_4 := arithEq_of_rows h (row := 379) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_4
  simp only [← c24657, ← c24658, ← c24659] at e_379_4
  have e_379_5 := arithEq_of_rows h (row := 379) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_5
  simp only [← c24660, ← c24661, k2, ← c24662] at e_379_5
  have hr := e_379_5
  simp only [e_379_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 379 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 551)) := by
  have c24663 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24664 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24665 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24666 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24667 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24668 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_379_6 := arithEq_of_rows h (row := 379) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_6
  simp only [← c24663, ← c24664, ← c24665] at e_379_6
  have e_379_7 := arithEq_of_rows h (row := 379) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_7
  simp only [← c24666, ← c24667, k2, ← c24668] at e_379_7
  have hr := e_379_7
  simp only [e_379_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 379 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 552)) := by
  have c24669 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24670 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24671 := (publicBatchWrapper86_copies770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24672 := (publicBatchWrapper86_copies771 a h).1
  have c24673 := (publicBatchWrapper86_copies771 a h).2.1
  have c24674 := (publicBatchWrapper86_copies771 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_379_8 := arithEq_of_rows h (row := 379) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_8
  simp only [← c24669, ← c24670, ← c24671] at e_379_8
  have e_379_9 := arithEq_of_rows h (row := 379) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_9
  simp only [← c24672, ← c24673, k2, ← c24674] at e_379_9
  have hr := e_379_9
  simp only [e_379_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 379 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 553)) := by
  have c24675 := (publicBatchWrapper86_copies771 a h).2.2.2.1
  have c24676 := (publicBatchWrapper86_copies771 a h).2.2.2.2.1
  have c24677 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.1
  have c24678 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.1
  have c24679 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.1
  have c24680 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_379_10 := arithEq_of_rows h (row := 379) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_10
  simp only [← c24675, ← c24676, ← c24677] at e_379_10
  have e_379_11 := arithEq_of_rows h (row := 379) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_11
  simp only [← c24678, ← c24679, k2, ← c24680] at e_379_11
  have hr := e_379_11
  simp only [e_379_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 379 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 554)) := by
  have c24681 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.1
  have c24682 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24683 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24684 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24685 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24686 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_379_12 := arithEq_of_rows h (row := 379) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_12
  simp only [← c24681, ← c24682, ← c24683] at e_379_12
  have e_379_13 := arithEq_of_rows h (row := 379) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_13
  simp only [← c24684, ← c24685, k2, ← c24686] at e_379_13
  have hr := e_379_13
  simp only [e_379_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 379 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 555)) := by
  have c24687 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24688 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24689 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24690 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24691 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24692 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_379_14 := arithEq_of_rows h (row := 379) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_14
  simp only [← c24687, ← c24688, ← c24689] at e_379_14
  have e_379_15 := arithEq_of_rows h (row := 379) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_15
  simp only [← c24690, ← c24691, k2, ← c24692] at e_379_15
  have hr := e_379_15
  simp only [e_379_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 379 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 556)) := by
  have c24693 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24694 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24695 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24696 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24697 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24698 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_379_16 := arithEq_of_rows h (row := 379) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_16
  simp only [← c24693, ← c24694, ← c24695] at e_379_16
  have e_379_17 := arithEq_of_rows h (row := 379) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_17
  simp only [← c24696, ← c24697, k2, ← c24698] at e_379_17
  have hr := e_379_17
  simp only [e_379_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 379 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 557)) := by
  have c24699 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24700 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24701 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24702 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24703 := (publicBatchWrapper86_copies771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24704 := (publicBatchWrapper86_copies772 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_379_18 := arithEq_of_rows h (row := 379) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_18
  simp only [← c24699, ← c24700, ← c24701] at e_379_18
  have e_379_19 := arithEq_of_rows h (row := 379) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_379_19
  simp only [← c24702, ← c24703, k2, ← c24704] at e_379_19
  have hr := e_379_19
  simp only [e_379_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 380 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 558)) := by
  have c24705 := (publicBatchWrapper86_copies772 a h).2.1
  have c24706 := (publicBatchWrapper86_copies772 a h).2.2.1
  have c24707 := (publicBatchWrapper86_copies772 a h).2.2.2.1
  have c24708 := (publicBatchWrapper86_copies772 a h).2.2.2.2.1
  have c24709 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.1
  have c24710 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_380_0 := arithEq_of_rows h (row := 380) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_0
  simp only [← c24705, ← c24706, ← c24707] at e_380_0
  have e_380_1 := arithEq_of_rows h (row := 380) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_1
  simp only [← c24708, ← c24709, k2, ← c24710] at e_380_1
  have hr := e_380_1
  simp only [e_380_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 380 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 559)) := by
  have c24711 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.1
  have c24712 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.1
  have c24713 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.1
  have c24714 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24715 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24716 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_380_2 := arithEq_of_rows h (row := 380) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_2
  simp only [← c24711, ← c24712, ← c24713] at e_380_2
  have e_380_3 := arithEq_of_rows h (row := 380) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_3
  simp only [← c24714, ← c24715, k2, ← c24716] at e_380_3
  have hr := e_380_3
  simp only [e_380_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 380 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 560)) := by
  have c24717 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24718 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24719 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24720 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24721 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24722 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_380_4 := arithEq_of_rows h (row := 380) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_4
  simp only [← c24717, ← c24718, ← c24719] at e_380_4
  have e_380_5 := arithEq_of_rows h (row := 380) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_5
  simp only [← c24720, ← c24721, k2, ← c24722] at e_380_5
  have hr := e_380_5
  simp only [e_380_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 380 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 561)) := by
  have c24723 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24724 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24725 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24726 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24727 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24728 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_380_6 := arithEq_of_rows h (row := 380) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_6
  simp only [← c24723, ← c24724, ← c24725] at e_380_6
  have e_380_7 := arithEq_of_rows h (row := 380) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_7
  simp only [← c24726, ← c24727, k2, ← c24728] at e_380_7
  have hr := e_380_7
  simp only [e_380_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 380 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 562)) := by
  have c24729 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24730 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24731 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24732 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24733 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24734 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_380_8 := arithEq_of_rows h (row := 380) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_8
  simp only [← c24729, ← c24730, ← c24731] at e_380_8
  have e_380_9 := arithEq_of_rows h (row := 380) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_9
  simp only [← c24732, ← c24733, k2, ← c24734] at e_380_9
  have hr := e_380_9
  simp only [e_380_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 380 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 563)) := by
  have c24735 := (publicBatchWrapper86_copies772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24736 := (publicBatchWrapper86_copies773 a h).1
  have c24737 := (publicBatchWrapper86_copies773 a h).2.1
  have c24738 := (publicBatchWrapper86_copies773 a h).2.2.1
  have c24739 := (publicBatchWrapper86_copies773 a h).2.2.2.1
  have c24740 := (publicBatchWrapper86_copies773 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_380_10 := arithEq_of_rows h (row := 380) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_10
  simp only [← c24735, ← c24736, ← c24737] at e_380_10
  have e_380_11 := arithEq_of_rows h (row := 380) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_11
  simp only [← c24738, ← c24739, k2, ← c24740] at e_380_11
  have hr := e_380_11
  simp only [e_380_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 380 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 564)) := by
  have c24741 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.1
  have c24742 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.1
  have c24743 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.1
  have c24744 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.1
  have c24745 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.1
  have c24746 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_380_12 := arithEq_of_rows h (row := 380) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_12
  simp only [← c24741, ← c24742, ← c24743] at e_380_12
  have e_380_13 := arithEq_of_rows h (row := 380) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_13
  simp only [← c24744, ← c24745, k2, ← c24746] at e_380_13
  have hr := e_380_13
  simp only [e_380_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 380 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 565)) := by
  have c24747 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24748 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24749 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24750 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24751 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24752 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_380_14 := arithEq_of_rows h (row := 380) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_14
  simp only [← c24747, ← c24748, ← c24749] at e_380_14
  have e_380_15 := arithEq_of_rows h (row := 380) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_15
  simp only [← c24750, ← c24751, k2, ← c24752] at e_380_15
  have hr := e_380_15
  simp only [e_380_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 380 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 566)) := by
  have c24753 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24754 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24755 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24756 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24757 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24758 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_380_16 := arithEq_of_rows h (row := 380) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_16
  simp only [← c24753, ← c24754, ← c24755] at e_380_16
  have e_380_17 := arithEq_of_rows h (row := 380) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_17
  simp only [← c24756, ← c24757, k2, ← c24758] at e_380_17
  have hr := e_380_17
  simp only [e_380_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 380 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 567)) := by
  have c24759 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24760 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24761 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24762 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24763 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24764 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_380_18 := arithEq_of_rows h (row := 380) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_18
  simp only [← c24759, ← c24760, ← c24761] at e_380_18
  have e_380_19 := arithEq_of_rows h (row := 380) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_380_19
  simp only [← c24762, ← c24763, k2, ← c24764] at e_380_19
  have hr := e_380_19
  simp only [e_380_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 381 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 568)) := by
  have c24765 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24766 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24767 := (publicBatchWrapper86_copies773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24768 := (publicBatchWrapper86_copies774 a h).1
  have c24769 := (publicBatchWrapper86_copies774 a h).2.1
  have c24770 := (publicBatchWrapper86_copies774 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_381_0 := arithEq_of_rows h (row := 381) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_0
  simp only [← c24765, ← c24766, ← c24767] at e_381_0
  have e_381_1 := arithEq_of_rows h (row := 381) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_1
  simp only [← c24768, ← c24769, k2, ← c24770] at e_381_1
  have hr := e_381_1
  simp only [e_381_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 381 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 569)) := by
  have c24771 := (publicBatchWrapper86_copies774 a h).2.2.2.1
  have c24772 := (publicBatchWrapper86_copies774 a h).2.2.2.2.1
  have c24773 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.1
  have c24774 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.1
  have c24775 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.1
  have c24776 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_381_2 := arithEq_of_rows h (row := 381) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_2
  simp only [← c24771, ← c24772, ← c24773] at e_381_2
  have e_381_3 := arithEq_of_rows h (row := 381) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_3
  simp only [← c24774, ← c24775, k2, ← c24776] at e_381_3
  have hr := e_381_3
  simp only [e_381_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 381 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 610)) := by
  have c24777 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.1
  have c24778 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24779 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24780 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24781 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24782 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_381_4 := arithEq_of_rows h (row := 381) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_4
  simp only [← c24777, ← c24778, ← c24779] at e_381_4
  have e_381_5 := arithEq_of_rows h (row := 381) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_5
  simp only [← c24780, ← c24781, k2, ← c24782] at e_381_5
  have hr := e_381_5
  simp only [e_381_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 381 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 611)) := by
  have c24783 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24784 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24785 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24786 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24787 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24788 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_381_6 := arithEq_of_rows h (row := 381) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_6
  simp only [← c24783, ← c24784, ← c24785] at e_381_6
  have e_381_7 := arithEq_of_rows h (row := 381) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_7
  simp only [← c24786, ← c24787, k2, ← c24788] at e_381_7
  have hr := e_381_7
  simp only [e_381_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 381 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 612)) := by
  have c24789 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24790 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24791 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24792 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24793 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24794 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_381_8 := arithEq_of_rows h (row := 381) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_8
  simp only [← c24789, ← c24790, ← c24791] at e_381_8
  have e_381_9 := arithEq_of_rows h (row := 381) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_9
  simp only [← c24792, ← c24793, k2, ← c24794] at e_381_9
  have hr := e_381_9
  simp only [e_381_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 381 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 613)) := by
  have c24795 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24796 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24797 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24798 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24799 := (publicBatchWrapper86_copies774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24800 := (publicBatchWrapper86_copies775 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_381_10 := arithEq_of_rows h (row := 381) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_10
  simp only [← c24795, ← c24796, ← c24797] at e_381_10
  have e_381_11 := arithEq_of_rows h (row := 381) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_11
  simp only [← c24798, ← c24799, k2, ← c24800] at e_381_11
  have hr := e_381_11
  simp only [e_381_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 381 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 614)) := by
  have c24801 := (publicBatchWrapper86_copies775 a h).2.1
  have c24802 := (publicBatchWrapper86_copies775 a h).2.2.1
  have c24803 := (publicBatchWrapper86_copies775 a h).2.2.2.1
  have c24804 := (publicBatchWrapper86_copies775 a h).2.2.2.2.1
  have c24805 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.1
  have c24806 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_381_12 := arithEq_of_rows h (row := 381) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_12
  simp only [← c24801, ← c24802, ← c24803] at e_381_12
  have e_381_13 := arithEq_of_rows h (row := 381) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_13
  simp only [← c24804, ← c24805, k2, ← c24806] at e_381_13
  have hr := e_381_13
  simp only [e_381_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 381 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 615)) := by
  have c24807 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.1
  have c24808 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.1
  have c24809 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.1
  have c24810 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24811 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24812 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_381_14 := arithEq_of_rows h (row := 381) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_14
  simp only [← c24807, ← c24808, ← c24809] at e_381_14
  have e_381_15 := arithEq_of_rows h (row := 381) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_15
  simp only [← c24810, ← c24811, k2, ← c24812] at e_381_15
  have hr := e_381_15
  simp only [e_381_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 381 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 616)) := by
  have c24813 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24814 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24815 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24816 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24817 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24818 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_381_16 := arithEq_of_rows h (row := 381) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_16
  simp only [← c24813, ← c24814, ← c24815] at e_381_16
  have e_381_17 := arithEq_of_rows h (row := 381) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_17
  simp only [← c24816, ← c24817, k2, ← c24818] at e_381_17
  have hr := e_381_17
  simp only [e_381_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 381 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 617)) := by
  have c24819 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24820 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24821 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24822 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24823 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24824 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_381_18 := arithEq_of_rows h (row := 381) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_18
  simp only [← c24819, ← c24820, ← c24821] at e_381_18
  have e_381_19 := arithEq_of_rows h (row := 381) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_381_19
  simp only [← c24822, ← c24823, k2, ← c24824] at e_381_19
  have hr := e_381_19
  simp only [e_381_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 382 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 618)) := by
  have c24825 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24826 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24827 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24828 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24829 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24830 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_382_0 := arithEq_of_rows h (row := 382) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_0
  simp only [← c24825, ← c24826, ← c24827] at e_382_0
  have e_382_1 := arithEq_of_rows h (row := 382) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_1
  simp only [← c24828, ← c24829, k2, ← c24830] at e_382_1
  have hr := e_382_1
  simp only [e_382_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 382 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 619)) := by
  have c24831 := (publicBatchWrapper86_copies775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24832 := (publicBatchWrapper86_copies776 a h).1
  have c24833 := (publicBatchWrapper86_copies776 a h).2.1
  have c24834 := (publicBatchWrapper86_copies776 a h).2.2.1
  have c24835 := (publicBatchWrapper86_copies776 a h).2.2.2.1
  have c24836 := (publicBatchWrapper86_copies776 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_382_2 := arithEq_of_rows h (row := 382) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_2
  simp only [← c24831, ← c24832, ← c24833] at e_382_2
  have e_382_3 := arithEq_of_rows h (row := 382) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_3
  simp only [← c24834, ← c24835, k2, ← c24836] at e_382_3
  have hr := e_382_3
  simp only [e_382_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 382 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 620)) := by
  have c24837 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.1
  have c24838 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.1
  have c24839 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.1
  have c24840 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.1
  have c24841 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.1
  have c24842 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_382_4 := arithEq_of_rows h (row := 382) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_4
  simp only [← c24837, ← c24838, ← c24839] at e_382_4
  have e_382_5 := arithEq_of_rows h (row := 382) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_5
  simp only [← c24840, ← c24841, k2, ← c24842] at e_382_5
  have hr := e_382_5
  simp only [e_382_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 382 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 621)) := by
  have c24843 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24844 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24845 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24846 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24847 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24848 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_382_6 := arithEq_of_rows h (row := 382) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_6
  simp only [← c24843, ← c24844, ← c24845] at e_382_6
  have e_382_7 := arithEq_of_rows h (row := 382) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_7
  simp only [← c24846, ← c24847, k2, ← c24848] at e_382_7
  have hr := e_382_7
  simp only [e_382_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 382 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 622)) := by
  have c24849 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24850 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24851 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24852 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24853 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24854 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_382_8 := arithEq_of_rows h (row := 382) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_8
  simp only [← c24849, ← c24850, ← c24851] at e_382_8
  have e_382_9 := arithEq_of_rows h (row := 382) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_9
  simp only [← c24852, ← c24853, k2, ← c24854] at e_382_9
  have hr := e_382_9
  simp only [e_382_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 382 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 623)) := by
  have c24855 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24856 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24857 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24858 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24859 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24860 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_382_10 := arithEq_of_rows h (row := 382) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_10
  simp only [← c24855, ← c24856, ← c24857] at e_382_10
  have e_382_11 := arithEq_of_rows h (row := 382) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_11
  simp only [← c24858, ← c24859, k2, ← c24860] at e_382_11
  have hr := e_382_11
  simp only [e_382_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 382 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 624)) := by
  have c24861 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24862 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24863 := (publicBatchWrapper86_copies776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24864 := (publicBatchWrapper86_copies777 a h).1
  have c24865 := (publicBatchWrapper86_copies777 a h).2.1
  have c24866 := (publicBatchWrapper86_copies777 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_382_12 := arithEq_of_rows h (row := 382) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_12
  simp only [← c24861, ← c24862, ← c24863] at e_382_12
  have e_382_13 := arithEq_of_rows h (row := 382) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_13
  simp only [← c24864, ← c24865, k2, ← c24866] at e_382_13
  have hr := e_382_13
  simp only [e_382_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 382 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 625)) := by
  have c24867 := (publicBatchWrapper86_copies777 a h).2.2.2.1
  have c24868 := (publicBatchWrapper86_copies777 a h).2.2.2.2.1
  have c24869 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.1
  have c24870 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.1
  have c24871 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.1
  have c24872 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_382_14 := arithEq_of_rows h (row := 382) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_14
  simp only [← c24867, ← c24868, ← c24869] at e_382_14
  have e_382_15 := arithEq_of_rows h (row := 382) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_15
  simp only [← c24870, ← c24871, k2, ← c24872] at e_382_15
  have hr := e_382_15
  simp only [e_382_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 382 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 626)) := by
  have c24873 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.1
  have c24874 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24875 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24876 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24877 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24878 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_382_16 := arithEq_of_rows h (row := 382) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_16
  simp only [← c24873, ← c24874, ← c24875] at e_382_16
  have e_382_17 := arithEq_of_rows h (row := 382) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_17
  simp only [← c24876, ← c24877, k2, ← c24878] at e_382_17
  have hr := e_382_17
  simp only [e_382_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 382 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 627)) := by
  have c24879 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24880 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24881 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24882 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24883 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24884 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_382_18 := arithEq_of_rows h (row := 382) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_18
  simp only [← c24879, ← c24880, ← c24881] at e_382_18
  have e_382_19 := arithEq_of_rows h (row := 382) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_382_19
  simp only [← c24882, ← c24883, k2, ← c24884] at e_382_19
  have hr := e_382_19
  simp only [e_382_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 383 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 628)) := by
  have c24885 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24886 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24887 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24888 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24889 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24890 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_383_0 := arithEq_of_rows h (row := 383) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_0
  simp only [← c24885, ← c24886, ← c24887] at e_383_0
  have e_383_1 := arithEq_of_rows h (row := 383) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_1
  simp only [← c24888, ← c24889, k2, ← c24890] at e_383_1
  have hr := e_383_1
  simp only [e_383_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 383 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 629)) := by
  have c24891 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24892 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24893 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24894 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24895 := (publicBatchWrapper86_copies777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24896 := (publicBatchWrapper86_copies778 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_383_2 := arithEq_of_rows h (row := 383) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_2
  simp only [← c24891, ← c24892, ← c24893] at e_383_2
  have e_383_3 := arithEq_of_rows h (row := 383) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_3
  simp only [← c24894, ← c24895, k2, ← c24896] at e_383_3
  have hr := e_383_3
  simp only [e_383_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 383 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 630)) := by
  have c24897 := (publicBatchWrapper86_copies778 a h).2.1
  have c24898 := (publicBatchWrapper86_copies778 a h).2.2.1
  have c24899 := (publicBatchWrapper86_copies778 a h).2.2.2.1
  have c24900 := (publicBatchWrapper86_copies778 a h).2.2.2.2.1
  have c24901 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.1
  have c24902 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_383_4 := arithEq_of_rows h (row := 383) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_4
  simp only [← c24897, ← c24898, ← c24899] at e_383_4
  have e_383_5 := arithEq_of_rows h (row := 383) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_5
  simp only [← c24900, ← c24901, k2, ← c24902] at e_383_5
  have hr := e_383_5
  simp only [e_383_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 383 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 631)) := by
  have c24903 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.1
  have c24904 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.1
  have c24905 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.1
  have c24906 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24907 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24908 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_383_6 := arithEq_of_rows h (row := 383) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_6
  simp only [← c24903, ← c24904, ← c24905] at e_383_6
  have e_383_7 := arithEq_of_rows h (row := 383) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_7
  simp only [← c24906, ← c24907, k2, ← c24908] at e_383_7
  have hr := e_383_7
  simp only [e_383_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 383 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 632)) := by
  have c24909 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24910 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24911 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24912 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24913 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24914 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_383_8 := arithEq_of_rows h (row := 383) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_8
  simp only [← c24909, ← c24910, ← c24911] at e_383_8
  have e_383_9 := arithEq_of_rows h (row := 383) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_9
  simp only [← c24912, ← c24913, k2, ← c24914] at e_383_9
  have hr := e_383_9
  simp only [e_383_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 383 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 633)) := by
  have c24915 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24916 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24917 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24918 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24919 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24920 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_383_10 := arithEq_of_rows h (row := 383) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_10
  simp only [← c24915, ← c24916, ← c24917] at e_383_10
  have e_383_11 := arithEq_of_rows h (row := 383) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_11
  simp only [← c24918, ← c24919, k2, ← c24920] at e_383_11
  have hr := e_383_11
  simp only [e_383_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 383 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 634)) := by
  have c24921 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24922 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24923 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24924 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24925 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24926 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_383_12 := arithEq_of_rows h (row := 383) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_12
  simp only [← c24921, ← c24922, ← c24923] at e_383_12
  have e_383_13 := arithEq_of_rows h (row := 383) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_13
  simp only [← c24924, ← c24925, k2, ← c24926] at e_383_13
  have hr := e_383_13
  simp only [e_383_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 383 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 635)) := by
  have c24927 := (publicBatchWrapper86_copies778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24928 := (publicBatchWrapper86_copies779 a h).1
  have c24929 := (publicBatchWrapper86_copies779 a h).2.1
  have c24930 := (publicBatchWrapper86_copies779 a h).2.2.1
  have c24931 := (publicBatchWrapper86_copies779 a h).2.2.2.1
  have c24932 := (publicBatchWrapper86_copies779 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_383_14 := arithEq_of_rows h (row := 383) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_14
  simp only [← c24927, ← c24928, ← c24929] at e_383_14
  have e_383_15 := arithEq_of_rows h (row := 383) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_15
  simp only [← c24930, ← c24931, k2, ← c24932] at e_383_15
  have hr := e_383_15
  simp only [e_383_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 383 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 636)) := by
  have c24933 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.1
  have c24934 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.1
  have c24935 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.1
  have c24936 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.1
  have c24937 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.1
  have c24938 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_383_16 := arithEq_of_rows h (row := 383) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_16
  simp only [← c24933, ← c24934, ← c24935] at e_383_16
  have e_383_17 := arithEq_of_rows h (row := 383) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_17
  simp only [← c24936, ← c24937, k2, ← c24938] at e_383_17
  have hr := e_383_17
  simp only [e_383_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 383 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 637)) := by
  have c24939 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24940 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24941 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24942 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24943 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24944 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_383_18 := arithEq_of_rows h (row := 383) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_18
  simp only [← c24939, ← c24940, ← c24941] at e_383_18
  have e_383_19 := arithEq_of_rows h (row := 383) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_383_19
  simp only [← c24942, ← c24943, k2, ← c24944] at e_383_19
  have hr := e_383_19
  simp only [e_383_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 384 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 638)) := by
  have c24945 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24946 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24947 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24948 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24949 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24950 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_384_0 := arithEq_of_rows h (row := 384) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_0
  simp only [← c24945, ← c24946, ← c24947] at e_384_0
  have e_384_1 := arithEq_of_rows h (row := 384) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_1
  simp only [← c24948, ← c24949, k2, ← c24950] at e_384_1
  have hr := e_384_1
  simp only [e_384_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 384 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 639)) := by
  have c24951 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24952 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24953 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24954 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24955 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24956 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_384_2 := arithEq_of_rows h (row := 384) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_2
  simp only [← c24951, ← c24952, ← c24953] at e_384_2
  have e_384_3 := arithEq_of_rows h (row := 384) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_3
  simp only [← c24954, ← c24955, k2, ← c24956] at e_384_3
  have hr := e_384_3
  simp only [e_384_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 384 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 640)) := by
  have c24957 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24958 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24959 := (publicBatchWrapper86_copies779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24960 := (publicBatchWrapper86_copies780 a h).1
  have c24961 := (publicBatchWrapper86_copies780 a h).2.1
  have c24962 := (publicBatchWrapper86_copies780 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_384_4 := arithEq_of_rows h (row := 384) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_4
  simp only [← c24957, ← c24958, ← c24959] at e_384_4
  have e_384_5 := arithEq_of_rows h (row := 384) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_5
  simp only [← c24960, ← c24961, k2, ← c24962] at e_384_5
  have hr := e_384_5
  simp only [e_384_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 384 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 641)) := by
  have c24963 := (publicBatchWrapper86_copies780 a h).2.2.2.1
  have c24964 := (publicBatchWrapper86_copies780 a h).2.2.2.2.1
  have c24965 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.1
  have c24966 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.1
  have c24967 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.1
  have c24968 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_384_6 := arithEq_of_rows h (row := 384) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_6
  simp only [← c24963, ← c24964, ← c24965] at e_384_6
  have e_384_7 := arithEq_of_rows h (row := 384) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_7
  simp only [← c24966, ← c24967, k2, ← c24968] at e_384_7
  have hr := e_384_7
  simp only [e_384_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 384 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 642)) := by
  have c24969 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.1
  have c24970 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.1
  have c24971 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c24972 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24973 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24974 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_384_8 := arithEq_of_rows h (row := 384) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_8
  simp only [← c24969, ← c24970, ← c24971] at e_384_8
  have e_384_9 := arithEq_of_rows h (row := 384) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_9
  simp only [← c24972, ← c24973, k2, ← c24974] at e_384_9
  have hr := e_384_9
  simp only [e_384_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 384 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 643)) := by
  have c24975 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24976 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24977 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24978 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24979 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24980 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_384_10 := arithEq_of_rows h (row := 384) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_10
  simp only [← c24975, ← c24976, ← c24977] at e_384_10
  have e_384_11 := arithEq_of_rows h (row := 384) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_11
  simp only [← c24978, ← c24979, k2, ← c24980] at e_384_11
  have hr := e_384_11
  simp only [e_384_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 384 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 644)) := by
  have c24981 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24982 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24983 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24984 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24985 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24986 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_384_12 := arithEq_of_rows h (row := 384) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_12
  simp only [← c24981, ← c24982, ← c24983] at e_384_12
  have e_384_13 := arithEq_of_rows h (row := 384) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_13
  simp only [← c24984, ← c24985, k2, ← c24986] at e_384_13
  have hr := e_384_13
  simp only [e_384_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 384 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 645)) := by
  have c24987 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24988 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24989 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24990 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c24991 := (publicBatchWrapper86_copies780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c24992 := (publicBatchWrapper86_copies781 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_384_14 := arithEq_of_rows h (row := 384) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_14
  simp only [← c24987, ← c24988, ← c24989] at e_384_14
  have e_384_15 := arithEq_of_rows h (row := 384) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_15
  simp only [← c24990, ← c24991, k2, ← c24992] at e_384_15
  have hr := e_384_15
  simp only [e_384_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 384 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 646)) := by
  have c24993 := (publicBatchWrapper86_copies781 a h).2.1
  have c24994 := (publicBatchWrapper86_copies781 a h).2.2.1
  have c24995 := (publicBatchWrapper86_copies781 a h).2.2.2.1
  have c24996 := (publicBatchWrapper86_copies781 a h).2.2.2.2.1
  have c24997 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.1
  have c24998 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_384_16 := arithEq_of_rows h (row := 384) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_16
  simp only [← c24993, ← c24994, ← c24995] at e_384_16
  have e_384_17 := arithEq_of_rows h (row := 384) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_17
  simp only [← c24996, ← c24997, k2, ← c24998] at e_384_17
  have hr := e_384_17
  simp only [e_384_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 384 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 647)) := by
  have c24999 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.1
  have c25000 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.1
  have c25001 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.1
  have c25002 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25003 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25004 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_384_18 := arithEq_of_rows h (row := 384) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_18
  simp only [← c24999, ← c25000, ← c25001] at e_384_18
  have e_384_19 := arithEq_of_rows h (row := 384) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_384_19
  simp only [← c25002, ← c25003, k2, ← c25004] at e_384_19
  have hr := e_384_19
  simp only [e_384_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 385 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 648)) := by
  have c25005 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25006 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25007 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25008 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25009 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25010 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_385_0 := arithEq_of_rows h (row := 385) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_0
  simp only [← c25005, ← c25006, ← c25007] at e_385_0
  have e_385_1 := arithEq_of_rows h (row := 385) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_1
  simp only [← c25008, ← c25009, k2, ← c25010] at e_385_1
  have hr := e_385_1
  simp only [e_385_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 385 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 649)) := by
  have c25011 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25012 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25013 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25014 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25015 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25016 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_385_2 := arithEq_of_rows h (row := 385) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_2
  simp only [← c25011, ← c25012, ← c25013] at e_385_2
  have e_385_3 := arithEq_of_rows h (row := 385) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_3
  simp only [← c25014, ← c25015, k2, ← c25016] at e_385_3
  have hr := e_385_3
  simp only [e_385_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 385 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 650)) := by
  have c25017 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25018 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25019 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25020 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25021 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25022 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_385_4 := arithEq_of_rows h (row := 385) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_4
  simp only [← c25017, ← c25018, ← c25019] at e_385_4
  have e_385_5 := arithEq_of_rows h (row := 385) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_5
  simp only [← c25020, ← c25021, k2, ← c25022] at e_385_5
  have hr := e_385_5
  simp only [e_385_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 385 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 651)) := by
  have c25023 := (publicBatchWrapper86_copies781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25024 := (publicBatchWrapper86_copies782 a h).1
  have c25025 := (publicBatchWrapper86_copies782 a h).2.1
  have c25026 := (publicBatchWrapper86_copies782 a h).2.2.1
  have c25027 := (publicBatchWrapper86_copies782 a h).2.2.2.1
  have c25028 := (publicBatchWrapper86_copies782 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_385_6 := arithEq_of_rows h (row := 385) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_6
  simp only [← c25023, ← c25024, ← c25025] at e_385_6
  have e_385_7 := arithEq_of_rows h (row := 385) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_7
  simp only [← c25026, ← c25027, k2, ← c25028] at e_385_7
  have hr := e_385_7
  simp only [e_385_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 385 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 652)) := by
  have c25029 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.1
  have c25030 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.1
  have c25031 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.1
  have c25032 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.1
  have c25033 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.1
  have c25034 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_385_8 := arithEq_of_rows h (row := 385) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_8
  simp only [← c25029, ← c25030, ← c25031] at e_385_8
  have e_385_9 := arithEq_of_rows h (row := 385) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_9
  simp only [← c25032, ← c25033, k2, ← c25034] at e_385_9
  have hr := e_385_9
  simp only [e_385_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 385 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 653)) := by
  have c25035 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25036 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25037 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25038 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25039 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25040 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_385_10 := arithEq_of_rows h (row := 385) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_10
  simp only [← c25035, ← c25036, ← c25037] at e_385_10
  have e_385_11 := arithEq_of_rows h (row := 385) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_11
  simp only [← c25038, ← c25039, k2, ← c25040] at e_385_11
  have hr := e_385_11
  simp only [e_385_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 385 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 654)) := by
  have c25041 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25042 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25043 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25044 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25045 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25046 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_385_12 := arithEq_of_rows h (row := 385) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_12
  simp only [← c25041, ← c25042, ← c25043] at e_385_12
  have e_385_13 := arithEq_of_rows h (row := 385) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_13
  simp only [← c25044, ← c25045, k2, ← c25046] at e_385_13
  have hr := e_385_13
  simp only [e_385_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 385 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 655)) := by
  have c25047 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25048 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25049 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25050 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25051 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25052 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_385_14 := arithEq_of_rows h (row := 385) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_14
  simp only [← c25047, ← c25048, ← c25049] at e_385_14
  have e_385_15 := arithEq_of_rows h (row := 385) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_15
  simp only [← c25050, ← c25051, k2, ← c25052] at e_385_15
  have hr := e_385_15
  simp only [e_385_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 385 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 656)) := by
  have c25053 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25054 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25055 := (publicBatchWrapper86_copies782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25056 := (publicBatchWrapper86_copies783 a h).1
  have c25057 := (publicBatchWrapper86_copies783 a h).2.1
  have c25058 := (publicBatchWrapper86_copies783 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_385_16 := arithEq_of_rows h (row := 385) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_16
  simp only [← c25053, ← c25054, ← c25055] at e_385_16
  have e_385_17 := arithEq_of_rows h (row := 385) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_17
  simp only [← c25056, ← c25057, k2, ← c25058] at e_385_17
  have hr := e_385_17
  simp only [e_385_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 385 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 657)) := by
  have c25059 := (publicBatchWrapper86_copies783 a h).2.2.2.1
  have c25060 := (publicBatchWrapper86_copies783 a h).2.2.2.2.1
  have c25061 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.1
  have c25062 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.1
  have c25063 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.1
  have c25064 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_385_18 := arithEq_of_rows h (row := 385) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_18
  simp only [← c25059, ← c25060, ← c25061] at e_385_18
  have e_385_19 := arithEq_of_rows h (row := 385) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_385_19
  simp only [← c25062, ← c25063, k2, ← c25064] at e_385_19
  have hr := e_385_19
  simp only [e_385_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 386 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 658)) := by
  have c25065 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.1
  have c25066 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25067 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25068 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25069 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25070 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_386_0 := arithEq_of_rows h (row := 386) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_0
  simp only [← c25065, ← c25066, ← c25067] at e_386_0
  have e_386_1 := arithEq_of_rows h (row := 386) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_1
  simp only [← c25068, ← c25069, k2, ← c25070] at e_386_1
  have hr := e_386_1
  simp only [e_386_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 386 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 659)) := by
  have c25071 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25072 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25073 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25074 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25075 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25076 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_386_2 := arithEq_of_rows h (row := 386) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_2
  simp only [← c25071, ← c25072, ← c25073] at e_386_2
  have e_386_3 := arithEq_of_rows h (row := 386) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_3
  simp only [← c25074, ← c25075, k2, ← c25076] at e_386_3
  have hr := e_386_3
  simp only [e_386_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 386 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 660)) := by
  have c25077 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25078 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25079 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25080 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25081 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25082 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_386_4 := arithEq_of_rows h (row := 386) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_4
  simp only [← c25077, ← c25078, ← c25079] at e_386_4
  have e_386_5 := arithEq_of_rows h (row := 386) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_5
  simp only [← c25080, ← c25081, k2, ← c25082] at e_386_5
  have hr := e_386_5
  simp only [e_386_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 386 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 661)) := by
  have c25083 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25084 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25085 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25086 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25087 := (publicBatchWrapper86_copies783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25088 := (publicBatchWrapper86_copies784 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_386_6 := arithEq_of_rows h (row := 386) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_6
  simp only [← c25083, ← c25084, ← c25085] at e_386_6
  have e_386_7 := arithEq_of_rows h (row := 386) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_7
  simp only [← c25086, ← c25087, k2, ← c25088] at e_386_7
  have hr := e_386_7
  simp only [e_386_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 386 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 662)) := by
  have c25089 := (publicBatchWrapper86_copies784 a h).2.1
  have c25090 := (publicBatchWrapper86_copies784 a h).2.2.1
  have c25091 := (publicBatchWrapper86_copies784 a h).2.2.2.1
  have c25092 := (publicBatchWrapper86_copies784 a h).2.2.2.2.1
  have c25093 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.1
  have c25094 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_386_8 := arithEq_of_rows h (row := 386) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_8
  simp only [← c25089, ← c25090, ← c25091] at e_386_8
  have e_386_9 := arithEq_of_rows h (row := 386) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_9
  simp only [← c25092, ← c25093, k2, ← c25094] at e_386_9
  have hr := e_386_9
  simp only [e_386_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 386 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 663)) := by
  have c25095 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.1
  have c25096 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.1
  have c25097 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.1
  have c25098 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25099 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25100 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_386_10 := arithEq_of_rows h (row := 386) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_10
  simp only [← c25095, ← c25096, ← c25097] at e_386_10
  have e_386_11 := arithEq_of_rows h (row := 386) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_11
  simp only [← c25098, ← c25099, k2, ← c25100] at e_386_11
  have hr := e_386_11
  simp only [e_386_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 386 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 664)) := by
  have c25101 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25102 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25103 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25104 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25105 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25106 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_386_12 := arithEq_of_rows h (row := 386) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_12
  simp only [← c25101, ← c25102, ← c25103] at e_386_12
  have e_386_13 := arithEq_of_rows h (row := 386) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_13
  simp only [← c25104, ← c25105, k2, ← c25106] at e_386_13
  have hr := e_386_13
  simp only [e_386_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 386 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 665)) := by
  have c25107 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25108 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25109 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25110 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25111 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25112 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_386_14 := arithEq_of_rows h (row := 386) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_14
  simp only [← c25107, ← c25108, ← c25109] at e_386_14
  have e_386_15 := arithEq_of_rows h (row := 386) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_15
  simp only [← c25110, ← c25111, k2, ← c25112] at e_386_15
  have hr := e_386_15
  simp only [e_386_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 386 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 666)) := by
  have c25113 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25114 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25115 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25116 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25117 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25118 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_386_16 := arithEq_of_rows h (row := 386) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_16
  simp only [← c25113, ← c25114, ← c25115] at e_386_16
  have e_386_17 := arithEq_of_rows h (row := 386) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_17
  simp only [← c25116, ← c25117, k2, ← c25118] at e_386_17
  have hr := e_386_17
  simp only [e_386_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 386 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 667)) := by
  have c25119 := (publicBatchWrapper86_copies784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25120 := (publicBatchWrapper86_copies785 a h).1
  have c25121 := (publicBatchWrapper86_copies785 a h).2.1
  have c25122 := (publicBatchWrapper86_copies785 a h).2.2.1
  have c25123 := (publicBatchWrapper86_copies785 a h).2.2.2.1
  have c25124 := (publicBatchWrapper86_copies785 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_386_18 := arithEq_of_rows h (row := 386) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_18
  simp only [← c25119, ← c25120, ← c25121] at e_386_18
  have e_386_19 := arithEq_of_rows h (row := 386) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_386_19
  simp only [← c25122, ← c25123, k2, ← c25124] at e_386_19
  have hr := e_386_19
  simp only [e_386_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 387 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 668)) := by
  have c25125 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.1
  have c25126 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.1
  have c25127 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.1
  have c25128 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.1
  have c25129 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.1
  have c25130 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_387_0 := arithEq_of_rows h (row := 387) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_0
  simp only [← c25125, ← c25126, ← c25127] at e_387_0
  have e_387_1 := arithEq_of_rows h (row := 387) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_1
  simp only [← c25128, ← c25129, k2, ← c25130] at e_387_1
  have hr := e_387_1
  simp only [e_387_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 387 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 669)) := by
  have c25131 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25132 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25133 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25134 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25135 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25136 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_387_2 := arithEq_of_rows h (row := 387) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_2
  simp only [← c25131, ← c25132, ← c25133] at e_387_2
  have e_387_3 := arithEq_of_rows h (row := 387) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_3
  simp only [← c25134, ← c25135, k2, ← c25136] at e_387_3
  have hr := e_387_3
  simp only [e_387_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 387 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 670)) := by
  have c25137 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25138 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25139 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25140 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25141 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25142 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_387_4 := arithEq_of_rows h (row := 387) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_4
  simp only [← c25137, ← c25138, ← c25139] at e_387_4
  have e_387_5 := arithEq_of_rows h (row := 387) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_5
  simp only [← c25140, ← c25141, k2, ← c25142] at e_387_5
  have hr := e_387_5
  simp only [e_387_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 387 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 671)) := by
  have c25143 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25144 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25145 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25146 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25147 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25148 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_387_6 := arithEq_of_rows h (row := 387) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_6
  simp only [← c25143, ← c25144, ← c25145] at e_387_6
  have e_387_7 := arithEq_of_rows h (row := 387) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_7
  simp only [← c25146, ← c25147, k2, ← c25148] at e_387_7
  have hr := e_387_7
  simp only [e_387_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 387 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 672)) := by
  have c25149 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25150 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25151 := (publicBatchWrapper86_copies785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25152 := (publicBatchWrapper86_copies786 a h).1
  have c25153 := (publicBatchWrapper86_copies786 a h).2.1
  have c25154 := (publicBatchWrapper86_copies786 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_387_8 := arithEq_of_rows h (row := 387) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_8
  simp only [← c25149, ← c25150, ← c25151] at e_387_8
  have e_387_9 := arithEq_of_rows h (row := 387) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_9
  simp only [← c25152, ← c25153, k2, ← c25154] at e_387_9
  have hr := e_387_9
  simp only [e_387_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 387 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 673)) := by
  have c25155 := (publicBatchWrapper86_copies786 a h).2.2.2.1
  have c25156 := (publicBatchWrapper86_copies786 a h).2.2.2.2.1
  have c25157 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.1
  have c25158 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.1
  have c25159 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.1
  have c25160 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_387_10 := arithEq_of_rows h (row := 387) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_10
  simp only [← c25155, ← c25156, ← c25157] at e_387_10
  have e_387_11 := arithEq_of_rows h (row := 387) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_11
  simp only [← c25158, ← c25159, k2, ← c25160] at e_387_11
  have hr := e_387_11
  simp only [e_387_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 387 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 674)) := by
  have c25161 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.1
  have c25162 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25163 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25164 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25165 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25166 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_387_12 := arithEq_of_rows h (row := 387) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_12
  simp only [← c25161, ← c25162, ← c25163] at e_387_12
  have e_387_13 := arithEq_of_rows h (row := 387) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_13
  simp only [← c25164, ← c25165, k2, ← c25166] at e_387_13
  have hr := e_387_13
  simp only [e_387_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 387 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 675)) := by
  have c25167 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25168 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25169 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25170 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25171 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25172 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_387_14 := arithEq_of_rows h (row := 387) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_14
  simp only [← c25167, ← c25168, ← c25169] at e_387_14
  have e_387_15 := arithEq_of_rows h (row := 387) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_15
  simp only [← c25170, ← c25171, k2, ← c25172] at e_387_15
  have hr := e_387_15
  simp only [e_387_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 387 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 676)) := by
  have c25173 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25174 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25175 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25176 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25177 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25178 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_387_16 := arithEq_of_rows h (row := 387) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_16
  simp only [← c25173, ← c25174, ← c25175] at e_387_16
  have e_387_17 := arithEq_of_rows h (row := 387) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_17
  simp only [← c25176, ← c25177, k2, ← c25178] at e_387_17
  have hr := e_387_17
  simp only [e_387_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 387 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 677)) := by
  have c25179 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25180 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25181 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25182 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25183 := (publicBatchWrapper86_copies786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25184 := (publicBatchWrapper86_copies787 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_387_18 := arithEq_of_rows h (row := 387) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_18
  simp only [← c25179, ← c25180, ← c25181] at e_387_18
  have e_387_19 := arithEq_of_rows h (row := 387) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_387_19
  simp only [← c25182, ← c25183, k2, ← c25184] at e_387_19
  have hr := e_387_19
  simp only [e_387_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 388 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 678)) := by
  have c25185 := (publicBatchWrapper86_copies787 a h).2.1
  have c25186 := (publicBatchWrapper86_copies787 a h).2.2.1
  have c25187 := (publicBatchWrapper86_copies787 a h).2.2.2.1
  have c25188 := (publicBatchWrapper86_copies787 a h).2.2.2.2.1
  have c25189 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.1
  have c25190 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_388_0 := arithEq_of_rows h (row := 388) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_0
  simp only [← c25185, ← c25186, ← c25187] at e_388_0
  have e_388_1 := arithEq_of_rows h (row := 388) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_1
  simp only [← c25188, ← c25189, k2, ← c25190] at e_388_1
  have hr := e_388_1
  simp only [e_388_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 388 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 679)) := by
  have c25191 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.1
  have c25192 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.1
  have c25193 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.1
  have c25194 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25195 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25196 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_388_2 := arithEq_of_rows h (row := 388) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_2
  simp only [← c25191, ← c25192, ← c25193] at e_388_2
  have e_388_3 := arithEq_of_rows h (row := 388) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_3
  simp only [← c25194, ← c25195, k2, ← c25196] at e_388_3
  have hr := e_388_3
  simp only [e_388_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 388 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 680)) := by
  have c25197 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25198 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25199 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25200 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25201 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25202 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_388_4 := arithEq_of_rows h (row := 388) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_4
  simp only [← c25197, ← c25198, ← c25199] at e_388_4
  have e_388_5 := arithEq_of_rows h (row := 388) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_5
  simp only [← c25200, ← c25201, k2, ← c25202] at e_388_5
  have hr := e_388_5
  simp only [e_388_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 388 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 681)) := by
  have c25203 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25204 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25205 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25206 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25207 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25208 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_388_6 := arithEq_of_rows h (row := 388) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_6
  simp only [← c25203, ← c25204, ← c25205] at e_388_6
  have e_388_7 := arithEq_of_rows h (row := 388) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_7
  simp only [← c25206, ← c25207, k2, ← c25208] at e_388_7
  have hr := e_388_7
  simp only [e_388_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 388 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 682)) := by
  have c25209 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25210 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25211 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25212 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25213 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25214 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_388_8 := arithEq_of_rows h (row := 388) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_8
  simp only [← c25209, ← c25210, ← c25211] at e_388_8
  have e_388_9 := arithEq_of_rows h (row := 388) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_9
  simp only [← c25212, ← c25213, k2, ← c25214] at e_388_9
  have hr := e_388_9
  simp only [e_388_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 388 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 683)) := by
  have c25215 := (publicBatchWrapper86_copies787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25216 := (publicBatchWrapper86_copies788 a h).1
  have c25217 := (publicBatchWrapper86_copies788 a h).2.1
  have c25218 := (publicBatchWrapper86_copies788 a h).2.2.1
  have c25219 := (publicBatchWrapper86_copies788 a h).2.2.2.1
  have c25220 := (publicBatchWrapper86_copies788 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_388_10 := arithEq_of_rows h (row := 388) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_10
  simp only [← c25215, ← c25216, ← c25217] at e_388_10
  have e_388_11 := arithEq_of_rows h (row := 388) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_11
  simp only [← c25218, ← c25219, k2, ← c25220] at e_388_11
  have hr := e_388_11
  simp only [e_388_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 388 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 684)) := by
  have c25221 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.1
  have c25222 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.1
  have c25223 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.1
  have c25224 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.1
  have c25225 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.1
  have c25226 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_388_12 := arithEq_of_rows h (row := 388) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_12
  simp only [← c25221, ← c25222, ← c25223] at e_388_12
  have e_388_13 := arithEq_of_rows h (row := 388) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_13
  simp only [← c25224, ← c25225, k2, ← c25226] at e_388_13
  have hr := e_388_13
  simp only [e_388_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 388 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 685)) := by
  have c25227 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25228 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25229 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25230 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25231 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25232 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_388_14 := arithEq_of_rows h (row := 388) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_14
  simp only [← c25227, ← c25228, ← c25229] at e_388_14
  have e_388_15 := arithEq_of_rows h (row := 388) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_15
  simp only [← c25230, ← c25231, k2, ← c25232] at e_388_15
  have hr := e_388_15
  simp only [e_388_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 388 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 686)) := by
  have c25233 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25234 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25235 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25236 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25237 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25238 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_388_16 := arithEq_of_rows h (row := 388) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_16
  simp only [← c25233, ← c25234, ← c25235] at e_388_16
  have e_388_17 := arithEq_of_rows h (row := 388) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_17
  simp only [← c25236, ← c25237, k2, ← c25238] at e_388_17
  have hr := e_388_17
  simp only [e_388_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 388 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 687)) := by
  have c25239 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25240 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25241 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25242 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25243 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25244 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_388_18 := arithEq_of_rows h (row := 388) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_18
  simp only [← c25239, ← c25240, ← c25241] at e_388_18
  have e_388_19 := arithEq_of_rows h (row := 388) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_388_19
  simp only [← c25242, ← c25243, k2, ← c25244] at e_388_19
  have hr := e_388_19
  simp only [e_388_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 389 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 688)) := by
  have c25245 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25246 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25247 := (publicBatchWrapper86_copies788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25248 := (publicBatchWrapper86_copies789 a h).1
  have c25249 := (publicBatchWrapper86_copies789 a h).2.1
  have c25250 := (publicBatchWrapper86_copies789 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_389_0 := arithEq_of_rows h (row := 389) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_0
  simp only [← c25245, ← c25246, ← c25247] at e_389_0
  have e_389_1 := arithEq_of_rows h (row := 389) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_1
  simp only [← c25248, ← c25249, k2, ← c25250] at e_389_1
  have hr := e_389_1
  simp only [e_389_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 389 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 689)) := by
  have c25251 := (publicBatchWrapper86_copies789 a h).2.2.2.1
  have c25252 := (publicBatchWrapper86_copies789 a h).2.2.2.2.1
  have c25253 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.1
  have c25254 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.1
  have c25255 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.1
  have c25256 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_389_2 := arithEq_of_rows h (row := 389) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_2
  simp only [← c25251, ← c25252, ← c25253] at e_389_2
  have e_389_3 := arithEq_of_rows h (row := 389) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_3
  simp only [← c25254, ← c25255, k2, ← c25256] at e_389_3
  have hr := e_389_3
  simp only [e_389_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 389 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 690)) := by
  have c25257 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.1
  have c25258 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25259 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25260 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25261 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25262 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_389_4 := arithEq_of_rows h (row := 389) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_4
  simp only [← c25257, ← c25258, ← c25259] at e_389_4
  have e_389_5 := arithEq_of_rows h (row := 389) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_5
  simp only [← c25260, ← c25261, k2, ← c25262] at e_389_5
  have hr := e_389_5
  simp only [e_389_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 389 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 691)) := by
  have c25263 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25264 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25265 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25266 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25267 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25268 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_389_6 := arithEq_of_rows h (row := 389) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_6
  simp only [← c25263, ← c25264, ← c25265] at e_389_6
  have e_389_7 := arithEq_of_rows h (row := 389) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_7
  simp only [← c25266, ← c25267, k2, ← c25268] at e_389_7
  have hr := e_389_7
  simp only [e_389_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 389 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 692)) := by
  have c25269 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25270 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25271 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25272 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25273 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25274 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_389_8 := arithEq_of_rows h (row := 389) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_8
  simp only [← c25269, ← c25270, ← c25271] at e_389_8
  have e_389_9 := arithEq_of_rows h (row := 389) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_9
  simp only [← c25272, ← c25273, k2, ← c25274] at e_389_9
  have hr := e_389_9
  simp only [e_389_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 389 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 693)) := by
  have c25275 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25276 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25277 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25278 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25279 := (publicBatchWrapper86_copies789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25280 := (publicBatchWrapper86_copies790 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_389_10 := arithEq_of_rows h (row := 389) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_10
  simp only [← c25275, ← c25276, ← c25277] at e_389_10
  have e_389_11 := arithEq_of_rows h (row := 389) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_11
  simp only [← c25278, ← c25279, k2, ← c25280] at e_389_11
  have hr := e_389_11
  simp only [e_389_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 389 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 694)) := by
  have c25281 := (publicBatchWrapper86_copies790 a h).2.1
  have c25282 := (publicBatchWrapper86_copies790 a h).2.2.1
  have c25283 := (publicBatchWrapper86_copies790 a h).2.2.2.1
  have c25284 := (publicBatchWrapper86_copies790 a h).2.2.2.2.1
  have c25285 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.1
  have c25286 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_389_12 := arithEq_of_rows h (row := 389) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_12
  simp only [← c25281, ← c25282, ← c25283] at e_389_12
  have e_389_13 := arithEq_of_rows h (row := 389) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_13
  simp only [← c25284, ← c25285, k2, ← c25286] at e_389_13
  have hr := e_389_13
  simp only [e_389_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 389 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 695)) := by
  have c25287 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.1
  have c25288 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.1
  have c25289 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.1
  have c25290 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25291 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25292 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_389_14 := arithEq_of_rows h (row := 389) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_14
  simp only [← c25287, ← c25288, ← c25289] at e_389_14
  have e_389_15 := arithEq_of_rows h (row := 389) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_15
  simp only [← c25290, ← c25291, k2, ← c25292] at e_389_15
  have hr := e_389_15
  simp only [e_389_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 389 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 696)) := by
  have c25293 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25294 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25295 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25296 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25297 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25298 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_389_16 := arithEq_of_rows h (row := 389) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_16
  simp only [← c25293, ← c25294, ← c25295] at e_389_16
  have e_389_17 := arithEq_of_rows h (row := 389) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_17
  simp only [← c25296, ← c25297, k2, ← c25298] at e_389_17
  have hr := e_389_17
  simp only [e_389_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 389 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 697)) := by
  have c25299 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25300 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25301 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25302 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25303 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25304 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_389_18 := arithEq_of_rows h (row := 389) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_18
  simp only [← c25299, ← c25300, ← c25301] at e_389_18
  have e_389_19 := arithEq_of_rows h (row := 389) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_389_19
  simp only [← c25302, ← c25303, k2, ← c25304] at e_389_19
  have hr := e_389_19
  simp only [e_389_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 390 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 698)) := by
  have c25305 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25306 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25307 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25308 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25309 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25310 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_390_0 := arithEq_of_rows h (row := 390) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_0
  simp only [← c25305, ← c25306, ← c25307] at e_390_0
  have e_390_1 := arithEq_of_rows h (row := 390) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_1
  simp only [← c25308, ← c25309, k2, ← c25310] at e_390_1
  have hr := e_390_1
  simp only [e_390_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 390 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 699)) := by
  have c25311 := (publicBatchWrapper86_copies790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25312 := (publicBatchWrapper86_copies791 a h).1
  have c25313 := (publicBatchWrapper86_copies791 a h).2.1
  have c25314 := (publicBatchWrapper86_copies791 a h).2.2.1
  have c25315 := (publicBatchWrapper86_copies791 a h).2.2.2.1
  have c25316 := (publicBatchWrapper86_copies791 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_390_2 := arithEq_of_rows h (row := 390) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_2
  simp only [← c25311, ← c25312, ← c25313] at e_390_2
  have e_390_3 := arithEq_of_rows h (row := 390) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_3
  simp only [← c25314, ← c25315, k2, ← c25316] at e_390_3
  have hr := e_390_3
  simp only [e_390_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 390 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 700)) := by
  have c25317 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.1
  have c25318 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.1
  have c25319 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.1
  have c25320 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.1
  have c25321 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.1
  have c25322 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_390_4 := arithEq_of_rows h (row := 390) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_4
  simp only [← c25317, ← c25318, ← c25319] at e_390_4
  have e_390_5 := arithEq_of_rows h (row := 390) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_5
  simp only [← c25320, ← c25321, k2, ← c25322] at e_390_5
  have hr := e_390_5
  simp only [e_390_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 390 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 701)) := by
  have c25323 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25324 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25325 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25326 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25327 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25328 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_390_6 := arithEq_of_rows h (row := 390) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_6
  simp only [← c25323, ← c25324, ← c25325] at e_390_6
  have e_390_7 := arithEq_of_rows h (row := 390) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_7
  simp only [← c25326, ← c25327, k2, ← c25328] at e_390_7
  have hr := e_390_7
  simp only [e_390_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 390 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 702)) := by
  have c25329 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25330 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25331 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25332 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25333 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25334 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_390_8 := arithEq_of_rows h (row := 390) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_8
  simp only [← c25329, ← c25330, ← c25331] at e_390_8
  have e_390_9 := arithEq_of_rows h (row := 390) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_9
  simp only [← c25332, ← c25333, k2, ← c25334] at e_390_9
  have hr := e_390_9
  simp only [e_390_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 390 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 703)) := by
  have c25335 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25336 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25337 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25338 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25339 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25340 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_390_10 := arithEq_of_rows h (row := 390) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_10
  simp only [← c25335, ← c25336, ← c25337] at e_390_10
  have e_390_11 := arithEq_of_rows h (row := 390) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_11
  simp only [← c25338, ← c25339, k2, ← c25340] at e_390_11
  have hr := e_390_11
  simp only [e_390_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 390 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 704)) := by
  have c25341 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25342 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25343 := (publicBatchWrapper86_copies791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25344 := (publicBatchWrapper86_copies792 a h).1
  have c25345 := (publicBatchWrapper86_copies792 a h).2.1
  have c25346 := (publicBatchWrapper86_copies792 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_390_12 := arithEq_of_rows h (row := 390) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_12
  simp only [← c25341, ← c25342, ← c25343] at e_390_12
  have e_390_13 := arithEq_of_rows h (row := 390) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_13
  simp only [← c25344, ← c25345, k2, ← c25346] at e_390_13
  have hr := e_390_13
  simp only [e_390_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 390 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 705)) := by
  have c25347 := (publicBatchWrapper86_copies792 a h).2.2.2.1
  have c25348 := (publicBatchWrapper86_copies792 a h).2.2.2.2.1
  have c25349 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.1
  have c25350 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.1
  have c25351 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.1
  have c25352 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_390_14 := arithEq_of_rows h (row := 390) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_14
  simp only [← c25347, ← c25348, ← c25349] at e_390_14
  have e_390_15 := arithEq_of_rows h (row := 390) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_15
  simp only [← c25350, ← c25351, k2, ← c25352] at e_390_15
  have hr := e_390_15
  simp only [e_390_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 390 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 706)) := by
  have c25353 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.1
  have c25354 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25355 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25356 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25357 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25358 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_390_16 := arithEq_of_rows h (row := 390) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_16
  simp only [← c25353, ← c25354, ← c25355] at e_390_16
  have e_390_17 := arithEq_of_rows h (row := 390) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_17
  simp only [← c25356, ← c25357, k2, ← c25358] at e_390_17
  have hr := e_390_17
  simp only [e_390_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 390 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 707)) := by
  have c25359 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25360 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25361 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25362 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25363 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25364 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_390_18 := arithEq_of_rows h (row := 390) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_18
  simp only [← c25359, ← c25360, ← c25361] at e_390_18
  have e_390_19 := arithEq_of_rows h (row := 390) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_390_19
  simp only [← c25362, ← c25363, k2, ← c25364] at e_390_19
  have hr := e_390_19
  simp only [e_390_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 391 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 708)) := by
  have c25365 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25366 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25367 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25368 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25369 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25370 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_391_0 := arithEq_of_rows h (row := 391) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_0
  simp only [← c25365, ← c25366, ← c25367] at e_391_0
  have e_391_1 := arithEq_of_rows h (row := 391) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_1
  simp only [← c25368, ← c25369, k2, ← c25370] at e_391_1
  have hr := e_391_1
  simp only [e_391_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 391 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 709)) := by
  have c25371 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25372 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25373 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25374 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25375 := (publicBatchWrapper86_copies792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25376 := (publicBatchWrapper86_copies793 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_391_2 := arithEq_of_rows h (row := 391) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_2
  simp only [← c25371, ← c25372, ← c25373] at e_391_2
  have e_391_3 := arithEq_of_rows h (row := 391) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_3
  simp only [← c25374, ← c25375, k2, ← c25376] at e_391_3
  have hr := e_391_3
  simp only [e_391_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 391 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 710)) := by
  have c25377 := (publicBatchWrapper86_copies793 a h).2.1
  have c25378 := (publicBatchWrapper86_copies793 a h).2.2.1
  have c25379 := (publicBatchWrapper86_copies793 a h).2.2.2.1
  have c25380 := (publicBatchWrapper86_copies793 a h).2.2.2.2.1
  have c25381 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.1
  have c25382 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_391_4 := arithEq_of_rows h (row := 391) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_4
  simp only [← c25377, ← c25378, ← c25379] at e_391_4
  have e_391_5 := arithEq_of_rows h (row := 391) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_5
  simp only [← c25380, ← c25381, k2, ← c25382] at e_391_5
  have hr := e_391_5
  simp only [e_391_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 391 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 711)) := by
  have c25383 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.1
  have c25384 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.1
  have c25385 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.1
  have c25386 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25387 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25388 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_391_6 := arithEq_of_rows h (row := 391) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_6
  simp only [← c25383, ← c25384, ← c25385] at e_391_6
  have e_391_7 := arithEq_of_rows h (row := 391) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_7
  simp only [← c25386, ← c25387, k2, ← c25388] at e_391_7
  have hr := e_391_7
  simp only [e_391_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 391 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 712)) := by
  have c25389 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25390 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25391 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25392 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25393 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25394 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_391_8 := arithEq_of_rows h (row := 391) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_8
  simp only [← c25389, ← c25390, ← c25391] at e_391_8
  have e_391_9 := arithEq_of_rows h (row := 391) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_9
  simp only [← c25392, ← c25393, k2, ← c25394] at e_391_9
  have hr := e_391_9
  simp only [e_391_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 391 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 713)) := by
  have c25395 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25396 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25397 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25398 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25399 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25400 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_391_10 := arithEq_of_rows h (row := 391) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_10
  simp only [← c25395, ← c25396, ← c25397] at e_391_10
  have e_391_11 := arithEq_of_rows h (row := 391) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_11
  simp only [← c25398, ← c25399, k2, ← c25400] at e_391_11
  have hr := e_391_11
  simp only [e_391_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 391 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 714)) := by
  have c25401 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25402 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25403 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25404 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25405 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25406 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_391_12 := arithEq_of_rows h (row := 391) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_12
  simp only [← c25401, ← c25402, ← c25403] at e_391_12
  have e_391_13 := arithEq_of_rows h (row := 391) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_13
  simp only [← c25404, ← c25405, k2, ← c25406] at e_391_13
  have hr := e_391_13
  simp only [e_391_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 391 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 715)) := by
  have c25407 := (publicBatchWrapper86_copies793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25408 := (publicBatchWrapper86_copies794 a h).1
  have c25409 := (publicBatchWrapper86_copies794 a h).2.1
  have c25410 := (publicBatchWrapper86_copies794 a h).2.2.1
  have c25411 := (publicBatchWrapper86_copies794 a h).2.2.2.1
  have c25412 := (publicBatchWrapper86_copies794 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_391_14 := arithEq_of_rows h (row := 391) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_14
  simp only [← c25407, ← c25408, ← c25409] at e_391_14
  have e_391_15 := arithEq_of_rows h (row := 391) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_15
  simp only [← c25410, ← c25411, k2, ← c25412] at e_391_15
  have hr := e_391_15
  simp only [e_391_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 391 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 716)) := by
  have c25413 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.1
  have c25414 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.1
  have c25415 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.1
  have c25416 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.1
  have c25417 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.1
  have c25418 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_391_16 := arithEq_of_rows h (row := 391) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_16
  simp only [← c25413, ← c25414, ← c25415] at e_391_16
  have e_391_17 := arithEq_of_rows h (row := 391) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_17
  simp only [← c25416, ← c25417, k2, ← c25418] at e_391_17
  have hr := e_391_17
  simp only [e_391_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 391 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 717)) := by
  have c25419 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25420 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25421 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25422 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25423 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25424 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_391_18 := arithEq_of_rows h (row := 391) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_18
  simp only [← c25419, ← c25420, ← c25421] at e_391_18
  have e_391_19 := arithEq_of_rows h (row := 391) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_391_19
  simp only [← c25422, ← c25423, k2, ← c25424] at e_391_19
  have hr := e_391_19
  simp only [e_391_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 392 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 718)) := by
  have c25425 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25426 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25427 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25428 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25429 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25430 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_392_0 := arithEq_of_rows h (row := 392) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_0
  simp only [← c25425, ← c25426, ← c25427] at e_392_0
  have e_392_1 := arithEq_of_rows h (row := 392) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_1
  simp only [← c25428, ← c25429, k2, ← c25430] at e_392_1
  have hr := e_392_1
  simp only [e_392_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 392 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 719)) := by
  have c25431 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25432 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25433 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25434 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25435 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25436 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_392_2 := arithEq_of_rows h (row := 392) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_2
  simp only [← c25431, ← c25432, ← c25433] at e_392_2
  have e_392_3 := arithEq_of_rows h (row := 392) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_3
  simp only [← c25434, ← c25435, k2, ← c25436] at e_392_3
  have hr := e_392_3
  simp only [e_392_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 392 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 720)) := by
  have c25437 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25438 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25439 := (publicBatchWrapper86_copies794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25440 := (publicBatchWrapper86_copies795 a h).1
  have c25441 := (publicBatchWrapper86_copies795 a h).2.1
  have c25442 := (publicBatchWrapper86_copies795 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_392_4 := arithEq_of_rows h (row := 392) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_4
  simp only [← c25437, ← c25438, ← c25439] at e_392_4
  have e_392_5 := arithEq_of_rows h (row := 392) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_5
  simp only [← c25440, ← c25441, k2, ← c25442] at e_392_5
  have hr := e_392_5
  simp only [e_392_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 392 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 721)) := by
  have c25443 := (publicBatchWrapper86_copies795 a h).2.2.2.1
  have c25444 := (publicBatchWrapper86_copies795 a h).2.2.2.2.1
  have c25445 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.1
  have c25446 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.1
  have c25447 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.1
  have c25448 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_392_6 := arithEq_of_rows h (row := 392) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_6
  simp only [← c25443, ← c25444, ← c25445] at e_392_6
  have e_392_7 := arithEq_of_rows h (row := 392) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_7
  simp only [← c25446, ← c25447, k2, ← c25448] at e_392_7
  have hr := e_392_7
  simp only [e_392_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 392 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 722)) := by
  have c25449 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.1
  have c25450 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25451 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25452 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25453 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25454 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_392_8 := arithEq_of_rows h (row := 392) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_8
  simp only [← c25449, ← c25450, ← c25451] at e_392_8
  have e_392_9 := arithEq_of_rows h (row := 392) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_9
  simp only [← c25452, ← c25453, k2, ← c25454] at e_392_9
  have hr := e_392_9
  simp only [e_392_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 392 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 723)) := by
  have c25455 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25456 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25457 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25458 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25459 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25460 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_392_10 := arithEq_of_rows h (row := 392) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_10
  simp only [← c25455, ← c25456, ← c25457] at e_392_10
  have e_392_11 := arithEq_of_rows h (row := 392) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_11
  simp only [← c25458, ← c25459, k2, ← c25460] at e_392_11
  have hr := e_392_11
  simp only [e_392_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 392 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 724)) := by
  have c25461 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25462 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25463 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25464 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25465 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25466 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_392_12 := arithEq_of_rows h (row := 392) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_12
  simp only [← c25461, ← c25462, ← c25463] at e_392_12
  have e_392_13 := arithEq_of_rows h (row := 392) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_13
  simp only [← c25464, ← c25465, k2, ← c25466] at e_392_13
  have hr := e_392_13
  simp only [e_392_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 392 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 725)) := by
  have c25467 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25468 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25469 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25470 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25471 := (publicBatchWrapper86_copies795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25472 := (publicBatchWrapper86_copies796 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_392_14 := arithEq_of_rows h (row := 392) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_14
  simp only [← c25467, ← c25468, ← c25469] at e_392_14
  have e_392_15 := arithEq_of_rows h (row := 392) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_15
  simp only [← c25470, ← c25471, k2, ← c25472] at e_392_15
  have hr := e_392_15
  simp only [e_392_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 392 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 726)) := by
  have c25473 := (publicBatchWrapper86_copies796 a h).2.1
  have c25474 := (publicBatchWrapper86_copies796 a h).2.2.1
  have c25475 := (publicBatchWrapper86_copies796 a h).2.2.2.1
  have c25476 := (publicBatchWrapper86_copies796 a h).2.2.2.2.1
  have c25477 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.1
  have c25478 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_392_16 := arithEq_of_rows h (row := 392) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_16
  simp only [← c25473, ← c25474, ← c25475] at e_392_16
  have e_392_17 := arithEq_of_rows h (row := 392) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_17
  simp only [← c25476, ← c25477, k2, ← c25478] at e_392_17
  have hr := e_392_17
  simp only [e_392_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 392 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 727)) := by
  have c25479 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.1
  have c25480 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.1
  have c25481 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.1
  have c25482 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25483 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25484 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_392_18 := arithEq_of_rows h (row := 392) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_18
  simp only [← c25479, ← c25480, ← c25481] at e_392_18
  have e_392_19 := arithEq_of_rows h (row := 392) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_392_19
  simp only [← c25482, ← c25483, k2, ← c25484] at e_392_19
  have hr := e_392_19
  simp only [e_392_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 393 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 728)) := by
  have c25485 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25486 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25487 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25488 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25489 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25490 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_393_0 := arithEq_of_rows h (row := 393) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_0
  simp only [← c25485, ← c25486, ← c25487] at e_393_0
  have e_393_1 := arithEq_of_rows h (row := 393) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_1
  simp only [← c25488, ← c25489, k2, ← c25490] at e_393_1
  have hr := e_393_1
  simp only [e_393_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 393 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 729)) := by
  have c25491 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25492 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25493 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25494 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25495 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25496 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_393_2 := arithEq_of_rows h (row := 393) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_2
  simp only [← c25491, ← c25492, ← c25493] at e_393_2
  have e_393_3 := arithEq_of_rows h (row := 393) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_3
  simp only [← c25494, ← c25495, k2, ← c25496] at e_393_3
  have hr := e_393_3
  simp only [e_393_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 393 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 730)) := by
  have c25497 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25498 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25499 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25500 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25501 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25502 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_393_4 := arithEq_of_rows h (row := 393) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_4
  simp only [← c25497, ← c25498, ← c25499] at e_393_4
  have e_393_5 := arithEq_of_rows h (row := 393) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_5
  simp only [← c25500, ← c25501, k2, ← c25502] at e_393_5
  have hr := e_393_5
  simp only [e_393_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 393 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 731)) := by
  have c25503 := (publicBatchWrapper86_copies796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25504 := (publicBatchWrapper86_copies797 a h).1
  have c25505 := (publicBatchWrapper86_copies797 a h).2.1
  have c25506 := (publicBatchWrapper86_copies797 a h).2.2.1
  have c25507 := (publicBatchWrapper86_copies797 a h).2.2.2.1
  have c25508 := (publicBatchWrapper86_copies797 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_393_6 := arithEq_of_rows h (row := 393) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_6
  simp only [← c25503, ← c25504, ← c25505] at e_393_6
  have e_393_7 := arithEq_of_rows h (row := 393) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_7
  simp only [← c25506, ← c25507, k2, ← c25508] at e_393_7
  have hr := e_393_7
  simp only [e_393_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 393 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 732)) := by
  have c25509 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.1
  have c25510 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.1
  have c25511 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.1
  have c25512 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.1
  have c25513 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.1
  have c25514 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_393_8 := arithEq_of_rows h (row := 393) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_8
  simp only [← c25509, ← c25510, ← c25511] at e_393_8
  have e_393_9 := arithEq_of_rows h (row := 393) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_9
  simp only [← c25512, ← c25513, k2, ← c25514] at e_393_9
  have hr := e_393_9
  simp only [e_393_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 393 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 733)) := by
  have c25515 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25516 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25517 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25518 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25519 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25520 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_393_10 := arithEq_of_rows h (row := 393) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_10
  simp only [← c25515, ← c25516, ← c25517] at e_393_10
  have e_393_11 := arithEq_of_rows h (row := 393) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_11
  simp only [← c25518, ← c25519, k2, ← c25520] at e_393_11
  have hr := e_393_11
  simp only [e_393_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 393 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 734)) := by
  have c25521 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25522 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25523 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25524 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25525 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25526 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_393_12 := arithEq_of_rows h (row := 393) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_12
  simp only [← c25521, ← c25522, ← c25523] at e_393_12
  have e_393_13 := arithEq_of_rows h (row := 393) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_13
  simp only [← c25524, ← c25525, k2, ← c25526] at e_393_13
  have hr := e_393_13
  simp only [e_393_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 393 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 735)) := by
  have c25527 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25528 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25529 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25530 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25531 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25532 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_393_14 := arithEq_of_rows h (row := 393) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_14
  simp only [← c25527, ← c25528, ← c25529] at e_393_14
  have e_393_15 := arithEq_of_rows h (row := 393) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_15
  simp only [← c25530, ← c25531, k2, ← c25532] at e_393_15
  have hr := e_393_15
  simp only [e_393_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 393 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 736)) := by
  have c25533 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25534 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25535 := (publicBatchWrapper86_copies797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25536 := (publicBatchWrapper86_copies798 a h).1
  have c25537 := (publicBatchWrapper86_copies798 a h).2.1
  have c25538 := (publicBatchWrapper86_copies798 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_393_16 := arithEq_of_rows h (row := 393) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_16
  simp only [← c25533, ← c25534, ← c25535] at e_393_16
  have e_393_17 := arithEq_of_rows h (row := 393) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_17
  simp only [← c25536, ← c25537, k2, ← c25538] at e_393_17
  have hr := e_393_17
  simp only [e_393_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 393 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 737)) := by
  have c25539 := (publicBatchWrapper86_copies798 a h).2.2.2.1
  have c25540 := (publicBatchWrapper86_copies798 a h).2.2.2.2.1
  have c25541 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.1
  have c25542 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.1
  have c25543 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.1
  have c25544 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_393_18 := arithEq_of_rows h (row := 393) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_18
  simp only [← c25539, ← c25540, ← c25541] at e_393_18
  have e_393_19 := arithEq_of_rows h (row := 393) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_393_19
  simp only [← c25542, ← c25543, k2, ← c25544] at e_393_19
  have hr := e_393_19
  simp only [e_393_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 394 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 738)) := by
  have c25545 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.1
  have c25546 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25547 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25548 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25549 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25550 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_394_0 := arithEq_of_rows h (row := 394) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_0
  simp only [← c25545, ← c25546, ← c25547] at e_394_0
  have e_394_1 := arithEq_of_rows h (row := 394) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_1
  simp only [← c25548, ← c25549, k2, ← c25550] at e_394_1
  have hr := e_394_1
  simp only [e_394_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 394 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 739)) := by
  have c25551 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25552 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25553 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25554 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25555 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25556 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_394_2 := arithEq_of_rows h (row := 394) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_2
  simp only [← c25551, ← c25552, ← c25553] at e_394_2
  have e_394_3 := arithEq_of_rows h (row := 394) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_3
  simp only [← c25554, ← c25555, k2, ← c25556] at e_394_3
  have hr := e_394_3
  simp only [e_394_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 394 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 740)) := by
  have c25557 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25558 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25559 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25560 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25561 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25562 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_394_4 := arithEq_of_rows h (row := 394) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_4
  simp only [← c25557, ← c25558, ← c25559] at e_394_4
  have e_394_5 := arithEq_of_rows h (row := 394) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_5
  simp only [← c25560, ← c25561, k2, ← c25562] at e_394_5
  have hr := e_394_5
  simp only [e_394_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 394 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 741)) := by
  have c25563 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25564 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25565 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25566 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25567 := (publicBatchWrapper86_copies798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25568 := (publicBatchWrapper86_copies799 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_394_6 := arithEq_of_rows h (row := 394) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_6
  simp only [← c25563, ← c25564, ← c25565] at e_394_6
  have e_394_7 := arithEq_of_rows h (row := 394) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_7
  simp only [← c25566, ← c25567, k2, ← c25568] at e_394_7
  have hr := e_394_7
  simp only [e_394_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 394 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 742)) := by
  have c25569 := (publicBatchWrapper86_copies799 a h).2.1
  have c25570 := (publicBatchWrapper86_copies799 a h).2.2.1
  have c25571 := (publicBatchWrapper86_copies799 a h).2.2.2.1
  have c25572 := (publicBatchWrapper86_copies799 a h).2.2.2.2.1
  have c25573 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.1
  have c25574 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_394_8 := arithEq_of_rows h (row := 394) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_8
  simp only [← c25569, ← c25570, ← c25571] at e_394_8
  have e_394_9 := arithEq_of_rows h (row := 394) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_9
  simp only [← c25572, ← c25573, k2, ← c25574] at e_394_9
  have hr := e_394_9
  simp only [e_394_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 394 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 743)) := by
  have c25575 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.1
  have c25576 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.1
  have c25577 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.1
  have c25578 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25579 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25580 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_394_10 := arithEq_of_rows h (row := 394) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_10
  simp only [← c25575, ← c25576, ← c25577] at e_394_10
  have e_394_11 := arithEq_of_rows h (row := 394) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_11
  simp only [← c25578, ← c25579, k2, ← c25580] at e_394_11
  have hr := e_394_11
  simp only [e_394_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 394 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 744)) := by
  have c25581 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25582 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25583 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25584 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25585 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25586 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_394_12 := arithEq_of_rows h (row := 394) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_12
  simp only [← c25581, ← c25582, ← c25583] at e_394_12
  have e_394_13 := arithEq_of_rows h (row := 394) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_13
  simp only [← c25584, ← c25585, k2, ← c25586] at e_394_13
  have hr := e_394_13
  simp only [e_394_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 394 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 745)) := by
  have c25587 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25588 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25589 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25590 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25591 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25592 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_394_14 := arithEq_of_rows h (row := 394) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_14
  simp only [← c25587, ← c25588, ← c25589] at e_394_14
  have e_394_15 := arithEq_of_rows h (row := 394) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_15
  simp only [← c25590, ← c25591, k2, ← c25592] at e_394_15
  have hr := e_394_15
  simp only [e_394_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 394 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 746)) := by
  have c25593 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25594 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25595 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25596 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25597 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25598 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_394_16 := arithEq_of_rows h (row := 394) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_16
  simp only [← c25593, ← c25594, ← c25595] at e_394_16
  have e_394_17 := arithEq_of_rows h (row := 394) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_17
  simp only [← c25596, ← c25597, k2, ← c25598] at e_394_17
  have hr := e_394_17
  simp only [e_394_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 394 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 747)) := by
  have c25599 := (publicBatchWrapper86_copies799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25600 := (publicBatchWrapper86_copies800 a h).1
  have c25601 := (publicBatchWrapper86_copies800 a h).2.1
  have c25602 := (publicBatchWrapper86_copies800 a h).2.2.1
  have c25603 := (publicBatchWrapper86_copies800 a h).2.2.2.1
  have c25604 := (publicBatchWrapper86_copies800 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_394_18 := arithEq_of_rows h (row := 394) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_18
  simp only [← c25599, ← c25600, ← c25601] at e_394_18
  have e_394_19 := arithEq_of_rows h (row := 394) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_394_19
  simp only [← c25602, ← c25603, k2, ← c25604] at e_394_19
  have hr := e_394_19
  simp only [e_394_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 395 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 748)) := by
  have c25605 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.1
  have c25606 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.1
  have c25607 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.1
  have c25608 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.1
  have c25609 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.1
  have c25610 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_395_0 := arithEq_of_rows h (row := 395) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_0
  simp only [← c25605, ← c25606, ← c25607] at e_395_0
  have e_395_1 := arithEq_of_rows h (row := 395) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_1
  simp only [← c25608, ← c25609, k2, ← c25610] at e_395_1
  have hr := e_395_1
  simp only [e_395_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 395 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 749)) := by
  have c25611 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25612 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25613 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25614 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25615 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25616 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_395_2 := arithEq_of_rows h (row := 395) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_2
  simp only [← c25611, ← c25612, ← c25613] at e_395_2
  have e_395_3 := arithEq_of_rows h (row := 395) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_3
  simp only [← c25614, ← c25615, k2, ← c25616] at e_395_3
  have hr := e_395_3
  simp only [e_395_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 395 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 750)) := by
  have c25617 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25618 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25619 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25620 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25621 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25622 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_395_4 := arithEq_of_rows h (row := 395) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_4
  simp only [← c25617, ← c25618, ← c25619] at e_395_4
  have e_395_5 := arithEq_of_rows h (row := 395) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_5
  simp only [← c25620, ← c25621, k2, ← c25622] at e_395_5
  have hr := e_395_5
  simp only [e_395_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 395 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 751)) := by
  have c25623 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25624 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25625 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25626 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25627 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25628 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_395_6 := arithEq_of_rows h (row := 395) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_6
  simp only [← c25623, ← c25624, ← c25625] at e_395_6
  have e_395_7 := arithEq_of_rows h (row := 395) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_7
  simp only [← c25626, ← c25627, k2, ← c25628] at e_395_7
  have hr := e_395_7
  simp only [e_395_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 395 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 752)) := by
  have c25629 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25630 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25631 := (publicBatchWrapper86_copies800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25632 := (publicBatchWrapper86_copies801 a h).1
  have c25633 := (publicBatchWrapper86_copies801 a h).2.1
  have c25634 := (publicBatchWrapper86_copies801 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_395_8 := arithEq_of_rows h (row := 395) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_8
  simp only [← c25629, ← c25630, ← c25631] at e_395_8
  have e_395_9 := arithEq_of_rows h (row := 395) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_9
  simp only [← c25632, ← c25633, k2, ← c25634] at e_395_9
  have hr := e_395_9
  simp only [e_395_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 395 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 753)) := by
  have c25635 := (publicBatchWrapper86_copies801 a h).2.2.2.1
  have c25636 := (publicBatchWrapper86_copies801 a h).2.2.2.2.1
  have c25637 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.1
  have c25638 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.1
  have c25639 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.1
  have c25640 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_395_10 := arithEq_of_rows h (row := 395) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_10
  simp only [← c25635, ← c25636, ← c25637] at e_395_10
  have e_395_11 := arithEq_of_rows h (row := 395) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_11
  simp only [← c25638, ← c25639, k2, ← c25640] at e_395_11
  have hr := e_395_11
  simp only [e_395_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 395 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 754)) := by
  have c25641 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.1
  have c25642 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25643 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25644 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25645 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25646 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_395_12 := arithEq_of_rows h (row := 395) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_12
  simp only [← c25641, ← c25642, ← c25643] at e_395_12
  have e_395_13 := arithEq_of_rows h (row := 395) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_13
  simp only [← c25644, ← c25645, k2, ← c25646] at e_395_13
  have hr := e_395_13
  simp only [e_395_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 395 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 755)) := by
  have c25647 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25648 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25649 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25650 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25651 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25652 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_395_14 := arithEq_of_rows h (row := 395) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_14
  simp only [← c25647, ← c25648, ← c25649] at e_395_14
  have e_395_15 := arithEq_of_rows h (row := 395) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_15
  simp only [← c25650, ← c25651, k2, ← c25652] at e_395_15
  have hr := e_395_15
  simp only [e_395_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 395 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 756)) := by
  have c25653 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25654 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25655 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25656 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25657 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25658 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_395_16 := arithEq_of_rows h (row := 395) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_16
  simp only [← c25653, ← c25654, ← c25655] at e_395_16
  have e_395_17 := arithEq_of_rows h (row := 395) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_17
  simp only [← c25656, ← c25657, k2, ← c25658] at e_395_17
  have hr := e_395_17
  simp only [e_395_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 395 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 757)) := by
  have c25659 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25660 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25661 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25662 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25663 := (publicBatchWrapper86_copies801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25664 := (publicBatchWrapper86_copies802 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_395_18 := arithEq_of_rows h (row := 395) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_18
  simp only [← c25659, ← c25660, ← c25661] at e_395_18
  have e_395_19 := arithEq_of_rows h (row := 395) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_395_19
  simp only [← c25662, ← c25663, k2, ← c25664] at e_395_19
  have hr := e_395_19
  simp only [e_395_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 396 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 758)) := by
  have c25665 := (publicBatchWrapper86_copies802 a h).2.1
  have c25666 := (publicBatchWrapper86_copies802 a h).2.2.1
  have c25667 := (publicBatchWrapper86_copies802 a h).2.2.2.1
  have c25668 := (publicBatchWrapper86_copies802 a h).2.2.2.2.1
  have c25669 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.1
  have c25670 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_396_0 := arithEq_of_rows h (row := 396) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_0
  simp only [← c25665, ← c25666, ← c25667] at e_396_0
  have e_396_1 := arithEq_of_rows h (row := 396) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_1
  simp only [← c25668, ← c25669, k2, ← c25670] at e_396_1
  have hr := e_396_1
  simp only [e_396_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 396 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 759)) := by
  have c25671 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.1
  have c25672 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.1
  have c25673 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.1
  have c25674 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25675 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25676 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_396_2 := arithEq_of_rows h (row := 396) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_2
  simp only [← c25671, ← c25672, ← c25673] at e_396_2
  have e_396_3 := arithEq_of_rows h (row := 396) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_3
  simp only [← c25674, ← c25675, k2, ← c25676] at e_396_3
  have hr := e_396_3
  simp only [e_396_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 396 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 760)) := by
  have c25677 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25678 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25679 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25680 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25681 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25682 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_396_4 := arithEq_of_rows h (row := 396) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_4
  simp only [← c25677, ← c25678, ← c25679] at e_396_4
  have e_396_5 := arithEq_of_rows h (row := 396) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_5
  simp only [← c25680, ← c25681, k2, ← c25682] at e_396_5
  have hr := e_396_5
  simp only [e_396_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 396 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 761)) := by
  have c25683 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25684 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25685 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25686 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25687 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25688 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_396_6 := arithEq_of_rows h (row := 396) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_6
  simp only [← c25683, ← c25684, ← c25685] at e_396_6
  have e_396_7 := arithEq_of_rows h (row := 396) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_7
  simp only [← c25686, ← c25687, k2, ← c25688] at e_396_7
  have hr := e_396_7
  simp only [e_396_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 396 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 762)) := by
  have c25689 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25690 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25691 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25692 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25693 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25694 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_396_8 := arithEq_of_rows h (row := 396) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_8
  simp only [← c25689, ← c25690, ← c25691] at e_396_8
  have e_396_9 := arithEq_of_rows h (row := 396) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_9
  simp only [← c25692, ← c25693, k2, ← c25694] at e_396_9
  have hr := e_396_9
  simp only [e_396_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 396 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 763)) := by
  have c25695 := (publicBatchWrapper86_copies802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25696 := (publicBatchWrapper86_copies803 a h).1
  have c25697 := (publicBatchWrapper86_copies803 a h).2.1
  have c25698 := (publicBatchWrapper86_copies803 a h).2.2.1
  have c25699 := (publicBatchWrapper86_copies803 a h).2.2.2.1
  have c25700 := (publicBatchWrapper86_copies803 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_396_10 := arithEq_of_rows h (row := 396) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_10
  simp only [← c25695, ← c25696, ← c25697] at e_396_10
  have e_396_11 := arithEq_of_rows h (row := 396) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_11
  simp only [← c25698, ← c25699, k2, ← c25700] at e_396_11
  have hr := e_396_11
  simp only [e_396_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 396 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 764)) := by
  have c25701 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.1
  have c25702 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.1
  have c25703 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.1
  have c25704 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.1
  have c25705 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.1
  have c25706 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_396_12 := arithEq_of_rows h (row := 396) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_12
  simp only [← c25701, ← c25702, ← c25703] at e_396_12
  have e_396_13 := arithEq_of_rows h (row := 396) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_13
  simp only [← c25704, ← c25705, k2, ← c25706] at e_396_13
  have hr := e_396_13
  simp only [e_396_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 396 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 765)) := by
  have c25707 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25708 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25709 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25710 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25711 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25712 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_396_14 := arithEq_of_rows h (row := 396) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_14
  simp only [← c25707, ← c25708, ← c25709] at e_396_14
  have e_396_15 := arithEq_of_rows h (row := 396) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_15
  simp only [← c25710, ← c25711, k2, ← c25712] at e_396_15
  have hr := e_396_15
  simp only [e_396_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 396 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 766)) := by
  have c25713 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25714 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25715 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25716 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25717 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25718 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_396_16 := arithEq_of_rows h (row := 396) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_16
  simp only [← c25713, ← c25714, ← c25715] at e_396_16
  have e_396_17 := arithEq_of_rows h (row := 396) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_17
  simp only [← c25716, ← c25717, k2, ← c25718] at e_396_17
  have hr := e_396_17
  simp only [e_396_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 396 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 767)) := by
  have c25719 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25720 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25721 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25722 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25723 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25724 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_396_18 := arithEq_of_rows h (row := 396) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_18
  simp only [← c25719, ← c25720, ← c25721] at e_396_18
  have e_396_19 := arithEq_of_rows h (row := 396) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_396_19
  simp only [← c25722, ← c25723, k2, ← c25724] at e_396_19
  have hr := e_396_19
  simp only [e_396_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 397 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 768)) := by
  have c25725 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25726 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25727 := (publicBatchWrapper86_copies803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25728 := (publicBatchWrapper86_copies804 a h).1
  have c25729 := (publicBatchWrapper86_copies804 a h).2.1
  have c25730 := (publicBatchWrapper86_copies804 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_397_0 := arithEq_of_rows h (row := 397) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_0
  simp only [← c25725, ← c25726, ← c25727] at e_397_0
  have e_397_1 := arithEq_of_rows h (row := 397) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_1
  simp only [← c25728, ← c25729, k2, ← c25730] at e_397_1
  have hr := e_397_1
  simp only [e_397_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 397 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 769)) := by
  have c25731 := (publicBatchWrapper86_copies804 a h).2.2.2.1
  have c25732 := (publicBatchWrapper86_copies804 a h).2.2.2.2.1
  have c25733 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.1
  have c25734 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.1
  have c25735 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.1
  have c25736 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_397_2 := arithEq_of_rows h (row := 397) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_2
  simp only [← c25731, ← c25732, ← c25733] at e_397_2
  have e_397_3 := arithEq_of_rows h (row := 397) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_3
  simp only [← c25734, ← c25735, k2, ← c25736] at e_397_3
  have hr := e_397_3
  simp only [e_397_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 397 23) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 810)) := by
  have c25737 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.1
  have c25738 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25739 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25740 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25741 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25742 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_397_4 := arithEq_of_rows h (row := 397) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_4
  simp only [← c25737, ← c25738, ← c25739] at e_397_4
  have e_397_5 := arithEq_of_rows h (row := 397) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_5
  simp only [← c25740, ← c25741, k2, ← c25742] at e_397_5
  have hr := e_397_5
  simp only [e_397_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 397 31) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 811)) := by
  have c25743 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25744 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25745 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25746 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25747 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25748 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_397_6 := arithEq_of_rows h (row := 397) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_6
  simp only [← c25743, ← c25744, ← c25745] at e_397_6
  have e_397_7 := arithEq_of_rows h (row := 397) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_7
  simp only [← c25746, ← c25747, k2, ← c25748] at e_397_7
  have hr := e_397_7
  simp only [e_397_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 397 39) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 812)) := by
  have c25749 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25750 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25751 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25752 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25753 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25754 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_397_8 := arithEq_of_rows h (row := 397) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_8
  simp only [← c25749, ← c25750, ← c25751] at e_397_8
  have e_397_9 := arithEq_of_rows h (row := 397) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_9
  simp only [← c25752, ← c25753, k2, ← c25754] at e_397_9
  have hr := e_397_9
  simp only [e_397_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 397 47) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 813)) := by
  have c25755 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25756 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25757 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25758 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25759 := (publicBatchWrapper86_copies804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25760 := (publicBatchWrapper86_copies805 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_397_10 := arithEq_of_rows h (row := 397) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_10
  simp only [← c25755, ← c25756, ← c25757] at e_397_10
  have e_397_11 := arithEq_of_rows h (row := 397) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_11
  simp only [← c25758, ← c25759, k2, ← c25760] at e_397_11
  have hr := e_397_11
  simp only [e_397_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 397 55) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 814)) := by
  have c25761 := (publicBatchWrapper86_copies805 a h).2.1
  have c25762 := (publicBatchWrapper86_copies805 a h).2.2.1
  have c25763 := (publicBatchWrapper86_copies805 a h).2.2.2.1
  have c25764 := (publicBatchWrapper86_copies805 a h).2.2.2.2.1
  have c25765 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.1
  have c25766 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_397_12 := arithEq_of_rows h (row := 397) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_12
  simp only [← c25761, ← c25762, ← c25763] at e_397_12
  have e_397_13 := arithEq_of_rows h (row := 397) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_13
  simp only [← c25764, ← c25765, k2, ← c25766] at e_397_13
  have hr := e_397_13
  simp only [e_397_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 397 63) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 815)) := by
  have c25767 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.1
  have c25768 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.1
  have c25769 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.1
  have c25770 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25771 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25772 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_397_14 := arithEq_of_rows h (row := 397) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_14
  simp only [← c25767, ← c25768, ← c25769] at e_397_14
  have e_397_15 := arithEq_of_rows h (row := 397) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_15
  simp only [← c25770, ← c25771, k2, ← c25772] at e_397_15
  have hr := e_397_15
  simp only [e_397_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 397 71) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 816)) := by
  have c25773 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25774 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25775 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25776 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25777 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25778 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_397_16 := arithEq_of_rows h (row := 397) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_16
  simp only [← c25773, ← c25774, ← c25775] at e_397_16
  have e_397_17 := arithEq_of_rows h (row := 397) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_17
  simp only [← c25776, ← c25777, k2, ← c25778] at e_397_17
  have hr := e_397_17
  simp only [e_397_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 397 79) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 817)) := by
  have c25779 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25780 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25781 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25782 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25783 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25784 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_397_18 := arithEq_of_rows h (row := 397) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_18
  simp only [← c25779, ← c25780, ← c25781] at e_397_18
  have e_397_19 := arithEq_of_rows h (row := 397) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_397_19
  simp only [← c25782, ← c25783, k2, ← c25784] at e_397_19
  have hr := e_397_19
  simp only [e_397_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 398 7) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 818)) := by
  have c25785 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25786 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25787 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25788 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25789 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25790 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_398_0 := arithEq_of_rows h (row := 398) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_0
  simp only [← c25785, ← c25786, ← c25787] at e_398_0
  have e_398_1 := arithEq_of_rows h (row := 398) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_1
  simp only [← c25788, ← c25789, k2, ← c25790] at e_398_1
  have hr := e_398_1
  simp only [e_398_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 398 15) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 819)) := by
  have c25791 := (publicBatchWrapper86_copies805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25792 := (publicBatchWrapper86_copies806 a h).1
  have c25793 := (publicBatchWrapper86_copies806 a h).2.1
  have c25794 := (publicBatchWrapper86_copies806 a h).2.2.1
  have c25795 := (publicBatchWrapper86_copies806 a h).2.2.2.1
  have c25796 := (publicBatchWrapper86_copies806 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_398_2 := arithEq_of_rows h (row := 398) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_2
  simp only [← c25791, ← c25792, ← c25793] at e_398_2
  have e_398_3 := arithEq_of_rows h (row := 398) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_3
  simp only [← c25794, ← c25795, k2, ← c25796] at e_398_3
  have hr := e_398_3
  simp only [e_398_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 398 23) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 820)) := by
  have c25797 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.1
  have c25798 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.1
  have c25799 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.1
  have c25800 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.1
  have c25801 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.1
  have c25802 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_398_4 := arithEq_of_rows h (row := 398) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_4
  simp only [← c25797, ← c25798, ← c25799] at e_398_4
  have e_398_5 := arithEq_of_rows h (row := 398) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_5
  simp only [← c25800, ← c25801, k2, ← c25802] at e_398_5
  have hr := e_398_5
  simp only [e_398_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 398 31) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 821)) := by
  have c25803 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25804 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25805 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25806 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25807 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25808 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_398_6 := arithEq_of_rows h (row := 398) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_6
  simp only [← c25803, ← c25804, ← c25805] at e_398_6
  have e_398_7 := arithEq_of_rows h (row := 398) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_7
  simp only [← c25806, ← c25807, k2, ← c25808] at e_398_7
  have hr := e_398_7
  simp only [e_398_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 398 39) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 822)) := by
  have c25809 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25810 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25811 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25812 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25813 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25814 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_398_8 := arithEq_of_rows h (row := 398) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_8
  simp only [← c25809, ← c25810, ← c25811] at e_398_8
  have e_398_9 := arithEq_of_rows h (row := 398) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_9
  simp only [← c25812, ← c25813, k2, ← c25814] at e_398_9
  have hr := e_398_9
  simp only [e_398_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 398 47) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 823)) := by
  have c25815 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25816 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25817 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25818 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25819 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25820 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_398_10 := arithEq_of_rows h (row := 398) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_10
  simp only [← c25815, ← c25816, ← c25817] at e_398_10
  have e_398_11 := arithEq_of_rows h (row := 398) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_11
  simp only [← c25818, ← c25819, k2, ← c25820] at e_398_11
  have hr := e_398_11
  simp only [e_398_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 398 55) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 824)) := by
  have c25821 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25822 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25823 := (publicBatchWrapper86_copies806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25824 := (publicBatchWrapper86_copies807 a h).1
  have c25825 := (publicBatchWrapper86_copies807 a h).2.1
  have c25826 := (publicBatchWrapper86_copies807 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_398_12 := arithEq_of_rows h (row := 398) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_12
  simp only [← c25821, ← c25822, ← c25823] at e_398_12
  have e_398_13 := arithEq_of_rows h (row := 398) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_13
  simp only [← c25824, ← c25825, k2, ← c25826] at e_398_13
  have hr := e_398_13
  simp only [e_398_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 398 63) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 825)) := by
  have c25827 := (publicBatchWrapper86_copies807 a h).2.2.2.1
  have c25828 := (publicBatchWrapper86_copies807 a h).2.2.2.2.1
  have c25829 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.1
  have c25830 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.1
  have c25831 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.1
  have c25832 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_398_14 := arithEq_of_rows h (row := 398) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_14
  simp only [← c25827, ← c25828, ← c25829] at e_398_14
  have e_398_15 := arithEq_of_rows h (row := 398) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_15
  simp only [← c25830, ← c25831, k2, ← c25832] at e_398_15
  have hr := e_398_15
  simp only [e_398_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 398 71) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 826)) := by
  have c25833 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.1
  have c25834 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25835 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25836 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25837 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25838 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_398_16 := arithEq_of_rows h (row := 398) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_16
  simp only [← c25833, ← c25834, ← c25835] at e_398_16
  have e_398_17 := arithEq_of_rows h (row := 398) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_17
  simp only [← c25836, ← c25837, k2, ← c25838] at e_398_17
  have hr := e_398_17
  simp only [e_398_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 398 79) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 827)) := by
  have c25839 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25840 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25841 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25842 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25843 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25844 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_398_18 := arithEq_of_rows h (row := 398) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_18
  simp only [← c25839, ← c25840, ← c25841] at e_398_18
  have e_398_19 := arithEq_of_rows h (row := 398) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_398_19
  simp only [← c25842, ← c25843, k2, ← c25844] at e_398_19
  have hr := e_398_19
  simp only [e_398_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 399 7) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 828)) := by
  have c25845 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25846 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25847 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25848 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25849 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25850 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_399_0 := arithEq_of_rows h (row := 399) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_0
  simp only [← c25845, ← c25846, ← c25847] at e_399_0
  have e_399_1 := arithEq_of_rows h (row := 399) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_1
  simp only [← c25848, ← c25849, k2, ← c25850] at e_399_1
  have hr := e_399_1
  simp only [e_399_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 399 15) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 829)) := by
  have c25851 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25852 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25853 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25854 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25855 := (publicBatchWrapper86_copies807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25856 := (publicBatchWrapper86_copies808 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_399_2 := arithEq_of_rows h (row := 399) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_2
  simp only [← c25851, ← c25852, ← c25853] at e_399_2
  have e_399_3 := arithEq_of_rows h (row := 399) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_3
  simp only [← c25854, ← c25855, k2, ← c25856] at e_399_3
  have hr := e_399_3
  simp only [e_399_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 399 23) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 830)) := by
  have c25857 := (publicBatchWrapper86_copies808 a h).2.1
  have c25858 := (publicBatchWrapper86_copies808 a h).2.2.1
  have c25859 := (publicBatchWrapper86_copies808 a h).2.2.2.1
  have c25860 := (publicBatchWrapper86_copies808 a h).2.2.2.2.1
  have c25861 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.1
  have c25862 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_399_4 := arithEq_of_rows h (row := 399) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_4
  simp only [← c25857, ← c25858, ← c25859] at e_399_4
  have e_399_5 := arithEq_of_rows h (row := 399) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_5
  simp only [← c25860, ← c25861, k2, ← c25862] at e_399_5
  have hr := e_399_5
  simp only [e_399_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 399 31) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 831)) := by
  have c25863 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.1
  have c25864 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.1
  have c25865 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.1
  have c25866 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25867 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25868 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_399_6 := arithEq_of_rows h (row := 399) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_6
  simp only [← c25863, ← c25864, ← c25865] at e_399_6
  have e_399_7 := arithEq_of_rows h (row := 399) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_7
  simp only [← c25866, ← c25867, k2, ← c25868] at e_399_7
  have hr := e_399_7
  simp only [e_399_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 399 39) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 832)) := by
  have c25869 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25870 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25871 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25872 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25873 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25874 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_399_8 := arithEq_of_rows h (row := 399) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_8
  simp only [← c25869, ← c25870, ← c25871] at e_399_8
  have e_399_9 := arithEq_of_rows h (row := 399) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_9
  simp only [← c25872, ← c25873, k2, ← c25874] at e_399_9
  have hr := e_399_9
  simp only [e_399_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 399 47) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 833)) := by
  have c25875 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25876 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25877 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25878 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25879 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25880 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_399_10 := arithEq_of_rows h (row := 399) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_10
  simp only [← c25875, ← c25876, ← c25877] at e_399_10
  have e_399_11 := arithEq_of_rows h (row := 399) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_11
  simp only [← c25878, ← c25879, k2, ← c25880] at e_399_11
  have hr := e_399_11
  simp only [e_399_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 399 55) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 834)) := by
  have c25881 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25882 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25883 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25884 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25885 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25886 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_399_12 := arithEq_of_rows h (row := 399) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_12
  simp only [← c25881, ← c25882, ← c25883] at e_399_12
  have e_399_13 := arithEq_of_rows h (row := 399) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_13
  simp only [← c25884, ← c25885, k2, ← c25886] at e_399_13
  have hr := e_399_13
  simp only [e_399_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 399 63) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 835)) := by
  have c25887 := (publicBatchWrapper86_copies808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25888 := (publicBatchWrapper86_copies809 a h).1
  have c25889 := (publicBatchWrapper86_copies809 a h).2.1
  have c25890 := (publicBatchWrapper86_copies809 a h).2.2.1
  have c25891 := (publicBatchWrapper86_copies809 a h).2.2.2.1
  have c25892 := (publicBatchWrapper86_copies809 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_399_14 := arithEq_of_rows h (row := 399) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_14
  simp only [← c25887, ← c25888, ← c25889] at e_399_14
  have e_399_15 := arithEq_of_rows h (row := 399) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_15
  simp only [← c25890, ← c25891, k2, ← c25892] at e_399_15
  have hr := e_399_15
  simp only [e_399_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 399 71) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 836)) := by
  have c25893 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.1
  have c25894 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.1
  have c25895 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.1
  have c25896 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.1
  have c25897 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.1
  have c25898 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_399_16 := arithEq_of_rows h (row := 399) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_16
  simp only [← c25893, ← c25894, ← c25895] at e_399_16
  have e_399_17 := arithEq_of_rows h (row := 399) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_17
  simp only [← c25896, ← c25897, k2, ← c25898] at e_399_17
  have hr := e_399_17
  simp only [e_399_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 399 79) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 837)) := by
  have c25899 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25900 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25901 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25902 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25903 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25904 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_399_18 := arithEq_of_rows h (row := 399) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_18
  simp only [← c25899, ← c25900, ← c25901] at e_399_18
  have e_399_19 := arithEq_of_rows h (row := 399) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_399_19
  simp only [← c25902, ← c25903, k2, ← c25904] at e_399_19
  have hr := e_399_19
  simp only [e_399_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 400 7) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 838)) := by
  have c25905 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25906 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25907 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25908 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25909 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25910 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_400_0 := arithEq_of_rows h (row := 400) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_0
  simp only [← c25905, ← c25906, ← c25907] at e_400_0
  have e_400_1 := arithEq_of_rows h (row := 400) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_1
  simp only [← c25908, ← c25909, k2, ← c25910] at e_400_1
  have hr := e_400_1
  simp only [e_400_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 400 15) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 839)) := by
  have c25911 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25912 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25913 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25914 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25915 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25916 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_400_2 := arithEq_of_rows h (row := 400) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_2
  simp only [← c25911, ← c25912, ← c25913] at e_400_2
  have e_400_3 := arithEq_of_rows h (row := 400) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_3
  simp only [← c25914, ← c25915, k2, ← c25916] at e_400_3
  have hr := e_400_3
  simp only [e_400_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 400 23) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 840)) := by
  have c25917 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25918 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25919 := (publicBatchWrapper86_copies809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25920 := (publicBatchWrapper86_copies810 a h).1
  have c25921 := (publicBatchWrapper86_copies810 a h).2.1
  have c25922 := (publicBatchWrapper86_copies810 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_400_4 := arithEq_of_rows h (row := 400) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_4
  simp only [← c25917, ← c25918, ← c25919] at e_400_4
  have e_400_5 := arithEq_of_rows h (row := 400) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_5
  simp only [← c25920, ← c25921, k2, ← c25922] at e_400_5
  have hr := e_400_5
  simp only [e_400_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 400 31) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 841)) := by
  have c25923 := (publicBatchWrapper86_copies810 a h).2.2.2.1
  have c25924 := (publicBatchWrapper86_copies810 a h).2.2.2.2.1
  have c25925 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.1
  have c25926 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.1
  have c25927 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.1
  have c25928 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_400_6 := arithEq_of_rows h (row := 400) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_6
  simp only [← c25923, ← c25924, ← c25925] at e_400_6
  have e_400_7 := arithEq_of_rows h (row := 400) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_7
  simp only [← c25926, ← c25927, k2, ← c25928] at e_400_7
  have hr := e_400_7
  simp only [e_400_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 400 39) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 842)) := by
  have c25929 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.1
  have c25930 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25931 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25932 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25933 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25934 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_400_8 := arithEq_of_rows h (row := 400) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_8
  simp only [← c25929, ← c25930, ← c25931] at e_400_8
  have e_400_9 := arithEq_of_rows h (row := 400) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_9
  simp only [← c25932, ← c25933, k2, ← c25934] at e_400_9
  have hr := e_400_9
  simp only [e_400_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 400 47) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 843)) := by
  have c25935 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25936 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25937 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25938 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25939 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25940 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_400_10 := arithEq_of_rows h (row := 400) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_10
  simp only [← c25935, ← c25936, ← c25937] at e_400_10
  have e_400_11 := arithEq_of_rows h (row := 400) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_11
  simp only [← c25938, ← c25939, k2, ← c25940] at e_400_11
  have hr := e_400_11
  simp only [e_400_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 400 55) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 844)) := by
  have c25941 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25942 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25943 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25944 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25945 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25946 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_400_12 := arithEq_of_rows h (row := 400) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_12
  simp only [← c25941, ← c25942, ← c25943] at e_400_12
  have e_400_13 := arithEq_of_rows h (row := 400) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_13
  simp only [← c25944, ← c25945, k2, ← c25946] at e_400_13
  have hr := e_400_13
  simp only [e_400_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 400 63) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 845)) := by
  have c25947 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25948 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25949 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25950 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25951 := (publicBatchWrapper86_copies810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25952 := (publicBatchWrapper86_copies811 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_400_14 := arithEq_of_rows h (row := 400) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_14
  simp only [← c25947, ← c25948, ← c25949] at e_400_14
  have e_400_15 := arithEq_of_rows h (row := 400) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_15
  simp only [← c25950, ← c25951, k2, ← c25952] at e_400_15
  have hr := e_400_15
  simp only [e_400_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 400 71) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 846)) := by
  have c25953 := (publicBatchWrapper86_copies811 a h).2.1
  have c25954 := (publicBatchWrapper86_copies811 a h).2.2.1
  have c25955 := (publicBatchWrapper86_copies811 a h).2.2.2.1
  have c25956 := (publicBatchWrapper86_copies811 a h).2.2.2.2.1
  have c25957 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.1
  have c25958 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_400_16 := arithEq_of_rows h (row := 400) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_16
  simp only [← c25953, ← c25954, ← c25955] at e_400_16
  have e_400_17 := arithEq_of_rows h (row := 400) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_17
  simp only [← c25956, ← c25957, k2, ← c25958] at e_400_17
  have hr := e_400_17
  simp only [e_400_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 400 79) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 847)) := by
  have c25959 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.1
  have c25960 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.1
  have c25961 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.1
  have c25962 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.1
  have c25963 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25964 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_400_18 := arithEq_of_rows h (row := 400) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_18
  simp only [← c25959, ← c25960, ← c25961] at e_400_18
  have e_400_19 := arithEq_of_rows h (row := 400) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_400_19
  simp only [← c25962, ← c25963, k2, ← c25964] at e_400_19
  have hr := e_400_19
  simp only [e_400_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 401 7) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 848)) := by
  have c25965 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25966 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25967 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25968 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25969 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25970 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_401_0 := arithEq_of_rows h (row := 401) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_0
  simp only [← c25965, ← c25966, ← c25967] at e_401_0
  have e_401_1 := arithEq_of_rows h (row := 401) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_1
  simp only [← c25968, ← c25969, k2, ← c25970] at e_401_1
  have hr := e_401_1
  simp only [e_401_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 401 15) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 849)) := by
  have c25971 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25972 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25973 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25974 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25975 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25976 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_401_2 := arithEq_of_rows h (row := 401) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_2
  simp only [← c25971, ← c25972, ← c25973] at e_401_2
  have e_401_3 := arithEq_of_rows h (row := 401) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_3
  simp only [← c25974, ← c25975, k2, ← c25976] at e_401_3
  have hr := e_401_3
  simp only [e_401_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 401 23) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 850)) := by
  have c25977 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25978 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25979 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25980 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25981 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25982 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_401_4 := arithEq_of_rows h (row := 401) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_4
  simp only [← c25977, ← c25978, ← c25979] at e_401_4
  have e_401_5 := arithEq_of_rows h (row := 401) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_5
  simp only [← c25980, ← c25981, k2, ← c25982] at e_401_5
  have hr := e_401_5
  simp only [e_401_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 401 31) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 851)) := by
  have c25983 := (publicBatchWrapper86_copies811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c25984 := (publicBatchWrapper86_copies812 a h).1
  have c25985 := (publicBatchWrapper86_copies812 a h).2.1
  have c25986 := (publicBatchWrapper86_copies812 a h).2.2.1
  have c25987 := (publicBatchWrapper86_copies812 a h).2.2.2.1
  have c25988 := (publicBatchWrapper86_copies812 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_401_6 := arithEq_of_rows h (row := 401) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_6
  simp only [← c25983, ← c25984, ← c25985] at e_401_6
  have e_401_7 := arithEq_of_rows h (row := 401) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_7
  simp only [← c25986, ← c25987, k2, ← c25988] at e_401_7
  have hr := e_401_7
  simp only [e_401_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 401 39) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 852)) := by
  have c25989 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.1
  have c25990 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.1
  have c25991 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.1
  have c25992 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.1
  have c25993 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.1
  have c25994 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_401_8 := arithEq_of_rows h (row := 401) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_8
  simp only [← c25989, ← c25990, ← c25991] at e_401_8
  have e_401_9 := arithEq_of_rows h (row := 401) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_9
  simp only [← c25992, ← c25993, k2, ← c25994] at e_401_9
  have hr := e_401_9
  simp only [e_401_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 401 47) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 853)) := by
  have c25995 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c25996 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25997 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25998 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c25999 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26000 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_401_10 := arithEq_of_rows h (row := 401) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_10
  simp only [← c25995, ← c25996, ← c25997] at e_401_10
  have e_401_11 := arithEq_of_rows h (row := 401) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_11
  simp only [← c25998, ← c25999, k2, ← c26000] at e_401_11
  have hr := e_401_11
  simp only [e_401_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 401 55) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 854)) := by
  have c26001 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26002 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26003 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26004 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26005 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26006 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_401_12 := arithEq_of_rows h (row := 401) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_12
  simp only [← c26001, ← c26002, ← c26003] at e_401_12
  have e_401_13 := arithEq_of_rows h (row := 401) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_13
  simp only [← c26004, ← c26005, k2, ← c26006] at e_401_13
  have hr := e_401_13
  simp only [e_401_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 401 63) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 855)) := by
  have c26007 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26008 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26009 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26010 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26011 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26012 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_401_14 := arithEq_of_rows h (row := 401) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_14
  simp only [← c26007, ← c26008, ← c26009] at e_401_14
  have e_401_15 := arithEq_of_rows h (row := 401) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_15
  simp only [← c26010, ← c26011, k2, ← c26012] at e_401_15
  have hr := e_401_15
  simp only [e_401_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 401 71) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 856)) := by
  have c26013 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26014 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26015 := (publicBatchWrapper86_copies812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26016 := (publicBatchWrapper86_copies813 a h).1
  have c26017 := (publicBatchWrapper86_copies813 a h).2.1
  have c26018 := (publicBatchWrapper86_copies813 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_401_16 := arithEq_of_rows h (row := 401) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_16
  simp only [← c26013, ← c26014, ← c26015] at e_401_16
  have e_401_17 := arithEq_of_rows h (row := 401) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_17
  simp only [← c26016, ← c26017, k2, ← c26018] at e_401_17
  have hr := e_401_17
  simp only [e_401_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 401 79) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 857)) := by
  have c26019 := (publicBatchWrapper86_copies813 a h).2.2.2.1
  have c26020 := (publicBatchWrapper86_copies813 a h).2.2.2.2.1
  have c26021 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.1
  have c26022 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.1
  have c26023 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.1
  have c26024 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_401_18 := arithEq_of_rows h (row := 401) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_18
  simp only [← c26019, ← c26020, ← c26021] at e_401_18
  have e_401_19 := arithEq_of_rows h (row := 401) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_401_19
  simp only [← c26022, ← c26023, k2, ← c26024] at e_401_19
  have hr := e_401_19
  simp only [e_401_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 402 7) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 858)) := by
  have c26025 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.1
  have c26026 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26027 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26028 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26029 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26030 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_402_0 := arithEq_of_rows h (row := 402) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_0
  simp only [← c26025, ← c26026, ← c26027] at e_402_0
  have e_402_1 := arithEq_of_rows h (row := 402) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_1
  simp only [← c26028, ← c26029, k2, ← c26030] at e_402_1
  have hr := e_402_1
  simp only [e_402_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 402 15) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 859)) := by
  have c26031 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26032 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26033 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26034 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26035 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26036 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_402_2 := arithEq_of_rows h (row := 402) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_2
  simp only [← c26031, ← c26032, ← c26033] at e_402_2
  have e_402_3 := arithEq_of_rows h (row := 402) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_3
  simp only [← c26034, ← c26035, k2, ← c26036] at e_402_3
  have hr := e_402_3
  simp only [e_402_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 402 23) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 860)) := by
  have c26037 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26038 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26039 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26040 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26041 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26042 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_402_4 := arithEq_of_rows h (row := 402) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_4
  simp only [← c26037, ← c26038, ← c26039] at e_402_4
  have e_402_5 := arithEq_of_rows h (row := 402) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_5
  simp only [← c26040, ← c26041, k2, ← c26042] at e_402_5
  have hr := e_402_5
  simp only [e_402_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 402 31) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 861)) := by
  have c26043 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26044 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26045 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26046 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26047 := (publicBatchWrapper86_copies813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26048 := (publicBatchWrapper86_copies814 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_402_6 := arithEq_of_rows h (row := 402) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_6
  simp only [← c26043, ← c26044, ← c26045] at e_402_6
  have e_402_7 := arithEq_of_rows h (row := 402) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_7
  simp only [← c26046, ← c26047, k2, ← c26048] at e_402_7
  have hr := e_402_7
  simp only [e_402_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 402 39) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 862)) := by
  have c26049 := (publicBatchWrapper86_copies814 a h).2.1
  have c26050 := (publicBatchWrapper86_copies814 a h).2.2.1
  have c26051 := (publicBatchWrapper86_copies814 a h).2.2.2.1
  have c26052 := (publicBatchWrapper86_copies814 a h).2.2.2.2.1
  have c26053 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.1
  have c26054 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_402_8 := arithEq_of_rows h (row := 402) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_8
  simp only [← c26049, ← c26050, ← c26051] at e_402_8
  have e_402_9 := arithEq_of_rows h (row := 402) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_9
  simp only [← c26052, ← c26053, k2, ← c26054] at e_402_9
  have hr := e_402_9
  simp only [e_402_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 402 47) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 863)) := by
  have c26055 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.1
  have c26056 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.1
  have c26057 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.1
  have c26058 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26059 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26060 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_402_10 := arithEq_of_rows h (row := 402) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_10
  simp only [← c26055, ← c26056, ← c26057] at e_402_10
  have e_402_11 := arithEq_of_rows h (row := 402) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_11
  simp only [← c26058, ← c26059, k2, ← c26060] at e_402_11
  have hr := e_402_11
  simp only [e_402_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 402 55) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 864)) := by
  have c26061 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26062 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26063 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26064 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26065 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26066 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_402_12 := arithEq_of_rows h (row := 402) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_12
  simp only [← c26061, ← c26062, ← c26063] at e_402_12
  have e_402_13 := arithEq_of_rows h (row := 402) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_13
  simp only [← c26064, ← c26065, k2, ← c26066] at e_402_13
  have hr := e_402_13
  simp only [e_402_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 402 63) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 865)) := by
  have c26067 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26068 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26069 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26070 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26071 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26072 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_402_14 := arithEq_of_rows h (row := 402) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_14
  simp only [← c26067, ← c26068, ← c26069] at e_402_14
  have e_402_15 := arithEq_of_rows h (row := 402) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_15
  simp only [← c26070, ← c26071, k2, ← c26072] at e_402_15
  have hr := e_402_15
  simp only [e_402_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 402 71) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 866)) := by
  have c26073 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26074 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26075 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26076 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26077 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26078 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_402_16 := arithEq_of_rows h (row := 402) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_16
  simp only [← c26073, ← c26074, ← c26075] at e_402_16
  have e_402_17 := arithEq_of_rows h (row := 402) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_17
  simp only [← c26076, ← c26077, k2, ← c26078] at e_402_17
  have hr := e_402_17
  simp only [e_402_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 402 79) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 867)) := by
  have c26079 := (publicBatchWrapper86_copies814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26080 := (publicBatchWrapper86_copies815 a h).1
  have c26081 := (publicBatchWrapper86_copies815 a h).2.1
  have c26082 := (publicBatchWrapper86_copies815 a h).2.2.1
  have c26083 := (publicBatchWrapper86_copies815 a h).2.2.2.1
  have c26084 := (publicBatchWrapper86_copies815 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_402_18 := arithEq_of_rows h (row := 402) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_18
  simp only [← c26079, ← c26080, ← c26081] at e_402_18
  have e_402_19 := arithEq_of_rows h (row := 402) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_402_19
  simp only [← c26082, ← c26083, k2, ← c26084] at e_402_19
  have hr := e_402_19
  simp only [e_402_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 403 7) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 868)) := by
  have c26085 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.1
  have c26086 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.1
  have c26087 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.1
  have c26088 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.1
  have c26089 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.1
  have c26090 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_403_0 := arithEq_of_rows h (row := 403) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_0
  simp only [← c26085, ← c26086, ← c26087] at e_403_0
  have e_403_1 := arithEq_of_rows h (row := 403) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_1
  simp only [← c26088, ← c26089, k2, ← c26090] at e_403_1
  have hr := e_403_1
  simp only [e_403_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 403 15) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 869)) := by
  have c26091 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26092 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26093 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26094 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26095 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26096 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_403_2 := arithEq_of_rows h (row := 403) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_2
  simp only [← c26091, ← c26092, ← c26093] at e_403_2
  have e_403_3 := arithEq_of_rows h (row := 403) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_3
  simp only [← c26094, ← c26095, k2, ← c26096] at e_403_3
  have hr := e_403_3
  simp only [e_403_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 403 23) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 870)) := by
  have c26097 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26098 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26099 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26100 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26101 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26102 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_403_4 := arithEq_of_rows h (row := 403) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_4
  simp only [← c26097, ← c26098, ← c26099] at e_403_4
  have e_403_5 := arithEq_of_rows h (row := 403) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_5
  simp only [← c26100, ← c26101, k2, ← c26102] at e_403_5
  have hr := e_403_5
  simp only [e_403_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 403 31) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 871)) := by
  have c26103 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26104 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26105 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26106 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26107 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26108 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_403_6 := arithEq_of_rows h (row := 403) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_6
  simp only [← c26103, ← c26104, ← c26105] at e_403_6
  have e_403_7 := arithEq_of_rows h (row := 403) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_7
  simp only [← c26106, ← c26107, k2, ← c26108] at e_403_7
  have hr := e_403_7
  simp only [e_403_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 403 39) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 872)) := by
  have c26109 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26110 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26111 := (publicBatchWrapper86_copies815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26112 := (publicBatchWrapper86_copies816 a h).1
  have c26113 := (publicBatchWrapper86_copies816 a h).2.1
  have c26114 := (publicBatchWrapper86_copies816 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_403_8 := arithEq_of_rows h (row := 403) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_8
  simp only [← c26109, ← c26110, ← c26111] at e_403_8
  have e_403_9 := arithEq_of_rows h (row := 403) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_9
  simp only [← c26112, ← c26113, k2, ← c26114] at e_403_9
  have hr := e_403_9
  simp only [e_403_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 403 47) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 873)) := by
  have c26115 := (publicBatchWrapper86_copies816 a h).2.2.2.1
  have c26116 := (publicBatchWrapper86_copies816 a h).2.2.2.2.1
  have c26117 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.1
  have c26118 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.1
  have c26119 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.1
  have c26120 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_403_10 := arithEq_of_rows h (row := 403) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_10
  simp only [← c26115, ← c26116, ← c26117] at e_403_10
  have e_403_11 := arithEq_of_rows h (row := 403) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_11
  simp only [← c26118, ← c26119, k2, ← c26120] at e_403_11
  have hr := e_403_11
  simp only [e_403_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 403 55) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 874)) := by
  have c26121 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.1
  have c26122 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26123 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26124 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26125 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26126 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_403_12 := arithEq_of_rows h (row := 403) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_12
  simp only [← c26121, ← c26122, ← c26123] at e_403_12
  have e_403_13 := arithEq_of_rows h (row := 403) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_13
  simp only [← c26124, ← c26125, k2, ← c26126] at e_403_13
  have hr := e_403_13
  simp only [e_403_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 403 63) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 875)) := by
  have c26127 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26128 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26129 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26130 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26131 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26132 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_403_14 := arithEq_of_rows h (row := 403) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_14
  simp only [← c26127, ← c26128, ← c26129] at e_403_14
  have e_403_15 := arithEq_of_rows h (row := 403) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_15
  simp only [← c26130, ← c26131, k2, ← c26132] at e_403_15
  have hr := e_403_15
  simp only [e_403_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 403 71) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 876)) := by
  have c26133 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26134 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26135 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26136 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26137 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26138 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_403_16 := arithEq_of_rows h (row := 403) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_16
  simp only [← c26133, ← c26134, ← c26135] at e_403_16
  have e_403_17 := arithEq_of_rows h (row := 403) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_17
  simp only [← c26136, ← c26137, k2, ← c26138] at e_403_17
  have hr := e_403_17
  simp only [e_403_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 403 79) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 877)) := by
  have c26139 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26140 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26141 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26142 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26143 := (publicBatchWrapper86_copies816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26144 := (publicBatchWrapper86_copies817 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_403_18 := arithEq_of_rows h (row := 403) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_18
  simp only [← c26139, ← c26140, ← c26141] at e_403_18
  have e_403_19 := arithEq_of_rows h (row := 403) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_403_19
  simp only [← c26142, ← c26143, k2, ← c26144] at e_403_19
  have hr := e_403_19
  simp only [e_403_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 404 7) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 878)) := by
  have c26145 := (publicBatchWrapper86_copies817 a h).2.1
  have c26146 := (publicBatchWrapper86_copies817 a h).2.2.1
  have c26147 := (publicBatchWrapper86_copies817 a h).2.2.2.1
  have c26148 := (publicBatchWrapper86_copies817 a h).2.2.2.2.1
  have c26149 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.1
  have c26150 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_404_0 := arithEq_of_rows h (row := 404) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_0
  simp only [← c26145, ← c26146, ← c26147] at e_404_0
  have e_404_1 := arithEq_of_rows h (row := 404) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_1
  simp only [← c26148, ← c26149, k2, ← c26150] at e_404_1
  have hr := e_404_1
  simp only [e_404_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 404 15) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 879)) := by
  have c26151 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.1
  have c26152 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.1
  have c26153 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.1
  have c26154 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26155 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26156 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_404_2 := arithEq_of_rows h (row := 404) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_2
  simp only [← c26151, ← c26152, ← c26153] at e_404_2
  have e_404_3 := arithEq_of_rows h (row := 404) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_3
  simp only [← c26154, ← c26155, k2, ← c26156] at e_404_3
  have hr := e_404_3
  simp only [e_404_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 404 23) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 880)) := by
  have c26157 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26158 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26159 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26160 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26161 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26162 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_404_4 := arithEq_of_rows h (row := 404) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_4
  simp only [← c26157, ← c26158, ← c26159] at e_404_4
  have e_404_5 := arithEq_of_rows h (row := 404) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_5
  simp only [← c26160, ← c26161, k2, ← c26162] at e_404_5
  have hr := e_404_5
  simp only [e_404_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 404 31) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 881)) := by
  have c26163 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26164 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26165 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26166 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26167 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26168 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_404_6 := arithEq_of_rows h (row := 404) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_6
  simp only [← c26163, ← c26164, ← c26165] at e_404_6
  have e_404_7 := arithEq_of_rows h (row := 404) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_7
  simp only [← c26166, ← c26167, k2, ← c26168] at e_404_7
  have hr := e_404_7
  simp only [e_404_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 404 39) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 882)) := by
  have c26169 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26170 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26171 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26172 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26173 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26174 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_404_8 := arithEq_of_rows h (row := 404) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_8
  simp only [← c26169, ← c26170, ← c26171] at e_404_8
  have e_404_9 := arithEq_of_rows h (row := 404) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_9
  simp only [← c26172, ← c26173, k2, ← c26174] at e_404_9
  have hr := e_404_9
  simp only [e_404_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 404 47) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 883)) := by
  have c26175 := (publicBatchWrapper86_copies817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26176 := (publicBatchWrapper86_copies818 a h).1
  have c26177 := (publicBatchWrapper86_copies818 a h).2.1
  have c26178 := (publicBatchWrapper86_copies818 a h).2.2.1
  have c26179 := (publicBatchWrapper86_copies818 a h).2.2.2.1
  have c26180 := (publicBatchWrapper86_copies818 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_404_10 := arithEq_of_rows h (row := 404) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_10
  simp only [← c26175, ← c26176, ← c26177] at e_404_10
  have e_404_11 := arithEq_of_rows h (row := 404) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_11
  simp only [← c26178, ← c26179, k2, ← c26180] at e_404_11
  have hr := e_404_11
  simp only [e_404_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 404 55) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 884)) := by
  have c26181 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.1
  have c26182 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.1
  have c26183 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.1
  have c26184 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.1
  have c26185 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.1
  have c26186 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_404_12 := arithEq_of_rows h (row := 404) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_12
  simp only [← c26181, ← c26182, ← c26183] at e_404_12
  have e_404_13 := arithEq_of_rows h (row := 404) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_13
  simp only [← c26184, ← c26185, k2, ← c26186] at e_404_13
  have hr := e_404_13
  simp only [e_404_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 404 63) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 885)) := by
  have c26187 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26188 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26189 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26190 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26191 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26192 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_404_14 := arithEq_of_rows h (row := 404) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_14
  simp only [← c26187, ← c26188, ← c26189] at e_404_14
  have e_404_15 := arithEq_of_rows h (row := 404) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_15
  simp only [← c26190, ← c26191, k2, ← c26192] at e_404_15
  have hr := e_404_15
  simp only [e_404_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 404 71) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 886)) := by
  have c26193 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26194 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26195 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26196 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26197 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26198 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_404_16 := arithEq_of_rows h (row := 404) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_16
  simp only [← c26193, ← c26194, ← c26195] at e_404_16
  have e_404_17 := arithEq_of_rows h (row := 404) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_17
  simp only [← c26196, ← c26197, k2, ← c26198] at e_404_17
  have hr := e_404_17
  simp only [e_404_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 404 79) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 887)) := by
  have c26199 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26200 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26201 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26202 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26203 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26204 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_404_18 := arithEq_of_rows h (row := 404) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_18
  simp only [← c26199, ← c26200, ← c26201] at e_404_18
  have e_404_19 := arithEq_of_rows h (row := 404) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_404_19
  simp only [← c26202, ← c26203, k2, ← c26204] at e_404_19
  have hr := e_404_19
  simp only [e_404_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 405 7) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 888)) := by
  have c26205 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26206 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26207 := (publicBatchWrapper86_copies818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26208 := (publicBatchWrapper86_copies819 a h).1
  have c26209 := (publicBatchWrapper86_copies819 a h).2.1
  have c26210 := (publicBatchWrapper86_copies819 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_405_0 := arithEq_of_rows h (row := 405) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_0
  simp only [← c26205, ← c26206, ← c26207] at e_405_0
  have e_405_1 := arithEq_of_rows h (row := 405) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_1
  simp only [← c26208, ← c26209, k2, ← c26210] at e_405_1
  have hr := e_405_1
  simp only [e_405_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 405 15) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 889)) := by
  have c26211 := (publicBatchWrapper86_copies819 a h).2.2.2.1
  have c26212 := (publicBatchWrapper86_copies819 a h).2.2.2.2.1
  have c26213 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.1
  have c26214 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.1
  have c26215 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.1
  have c26216 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_405_2 := arithEq_of_rows h (row := 405) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_2
  simp only [← c26211, ← c26212, ← c26213] at e_405_2
  have e_405_3 := arithEq_of_rows h (row := 405) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_3
  simp only [← c26214, ← c26215, k2, ← c26216] at e_405_3
  have hr := e_405_3
  simp only [e_405_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 405 23) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 890)) := by
  have c26217 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.1
  have c26218 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26219 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26220 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26221 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26222 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_405_4 := arithEq_of_rows h (row := 405) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_4
  simp only [← c26217, ← c26218, ← c26219] at e_405_4
  have e_405_5 := arithEq_of_rows h (row := 405) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_5
  simp only [← c26220, ← c26221, k2, ← c26222] at e_405_5
  have hr := e_405_5
  simp only [e_405_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 405 31) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 891)) := by
  have c26223 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26224 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26225 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26226 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26227 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26228 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_405_6 := arithEq_of_rows h (row := 405) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_6
  simp only [← c26223, ← c26224, ← c26225] at e_405_6
  have e_405_7 := arithEq_of_rows h (row := 405) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_7
  simp only [← c26226, ← c26227, k2, ← c26228] at e_405_7
  have hr := e_405_7
  simp only [e_405_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 405 39) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 892)) := by
  have c26229 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26230 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26231 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26232 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26233 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26234 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_405_8 := arithEq_of_rows h (row := 405) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_8
  simp only [← c26229, ← c26230, ← c26231] at e_405_8
  have e_405_9 := arithEq_of_rows h (row := 405) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_9
  simp only [← c26232, ← c26233, k2, ← c26234] at e_405_9
  have hr := e_405_9
  simp only [e_405_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 405 47) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 893)) := by
  have c26235 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26236 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26237 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26238 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c26239 := (publicBatchWrapper86_copies819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c26240 := (publicBatchWrapper86_copies820 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_405_10 := arithEq_of_rows h (row := 405) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_10
  simp only [← c26235, ← c26236, ← c26237] at e_405_10
  have e_405_11 := arithEq_of_rows h (row := 405) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_11
  simp only [← c26238, ← c26239, k2, ← c26240] at e_405_11
  have hr := e_405_11
  simp only [e_405_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 405 55) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 894)) := by
  have c26241 := (publicBatchWrapper86_copies820 a h).2.1
  have c26242 := (publicBatchWrapper86_copies820 a h).2.2.1
  have c26243 := (publicBatchWrapper86_copies820 a h).2.2.2.1
  have c26244 := (publicBatchWrapper86_copies820 a h).2.2.2.2.1
  have c26245 := (publicBatchWrapper86_copies820 a h).2.2.2.2.2.1
  have c26246 := (publicBatchWrapper86_copies820 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_405_12 := arithEq_of_rows h (row := 405) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_12
  simp only [← c26241, ← c26242, ← c26243] at e_405_12
  have e_405_13 := arithEq_of_rows h (row := 405) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_13
  simp only [← c26244, ← c26245, k2, ← c26246] at e_405_13
  have hr := e_405_13
  simp only [e_405_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f3649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 405 63) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 895)) := by
  have c26247 := (publicBatchWrapper86_copies820 a h).2.2.2.2.2.2.2.1
  have c26248 := (publicBatchWrapper86_copies820 a h).2.2.2.2.2.2.2.2.1
  have c26249 := (publicBatchWrapper86_copies820 a h).2.2.2.2.2.2.2.2.2.1
  have c26250 := (publicBatchWrapper86_copies820 a h).2.2.2.2.2.2.2.2.2.2.1
  have c26251 := (publicBatchWrapper86_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c26252 := (publicBatchWrapper86_copies820 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_405_14 := arithEq_of_rows h (row := 405) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_14
  simp only [← c26247, ← c26248, ← c26249] at e_405_14
  have e_405_15 := arithEq_of_rows h (row := 405) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_405_15
  simp only [← c26250, ← c26251, k2, ← c26252] at e_405_15
  have hr := e_405_15
  simp only [e_405_14] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
