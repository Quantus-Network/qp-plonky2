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

theorem publicBatchWrapper86_f14667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1507 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14673)) := by
  have c92355 := (publicBatchWrapper86_copies2886 a h).2.2.2.1
  have c92356 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.1
  have c92357 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.1
  have c92358 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.1
  have c92359 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.1
  have c92360 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1507_10 := arithEq_of_rows h (row := 1507) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_10
  simp only [← c92355, ← c92356, ← c92357] at e_1507_10
  have e_1507_11 := arithEq_of_rows h (row := 1507) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_11
  simp only [← c92358, ← c92359, k2, ← c92360] at e_1507_11
  have hr := e_1507_11
  simp only [e_1507_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1507 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14674)) := by
  have c92361 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.1
  have c92362 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92363 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92364 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92365 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92366 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1507_12 := arithEq_of_rows h (row := 1507) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_12
  simp only [← c92361, ← c92362, ← c92363] at e_1507_12
  have e_1507_13 := arithEq_of_rows h (row := 1507) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_13
  simp only [← c92364, ← c92365, k2, ← c92366] at e_1507_13
  have hr := e_1507_13
  simp only [e_1507_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1507 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14675)) := by
  have c92367 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92368 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92369 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92370 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92371 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92372 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1507_14 := arithEq_of_rows h (row := 1507) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_14
  simp only [← c92367, ← c92368, ← c92369] at e_1507_14
  have e_1507_15 := arithEq_of_rows h (row := 1507) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_15
  simp only [← c92370, ← c92371, k2, ← c92372] at e_1507_15
  have hr := e_1507_15
  simp only [e_1507_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1507 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14676)) := by
  have c92373 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92374 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92375 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92376 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92377 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92378 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1507_16 := arithEq_of_rows h (row := 1507) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_16
  simp only [← c92373, ← c92374, ← c92375] at e_1507_16
  have e_1507_17 := arithEq_of_rows h (row := 1507) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_17
  simp only [← c92376, ← c92377, k2, ← c92378] at e_1507_17
  have hr := e_1507_17
  simp only [e_1507_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1507 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14677)) := by
  have c92379 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92380 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92381 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92382 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92383 := (publicBatchWrapper86_copies2886 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92384 := (publicBatchWrapper86_copies2887 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1507_18 := arithEq_of_rows h (row := 1507) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_18
  simp only [← c92379, ← c92380, ← c92381] at e_1507_18
  have e_1507_19 := arithEq_of_rows h (row := 1507) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_19
  simp only [← c92382, ← c92383, k2, ← c92384] at e_1507_19
  have hr := e_1507_19
  simp only [e_1507_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1508 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14678)) := by
  have c92385 := (publicBatchWrapper86_copies2887 a h).2.1
  have c92386 := (publicBatchWrapper86_copies2887 a h).2.2.1
  have c92387 := (publicBatchWrapper86_copies2887 a h).2.2.2.1
  have c92388 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.1
  have c92389 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.1
  have c92390 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1508_0 := arithEq_of_rows h (row := 1508) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_0
  simp only [← c92385, ← c92386, ← c92387] at e_1508_0
  have e_1508_1 := arithEq_of_rows h (row := 1508) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_1
  simp only [← c92388, ← c92389, k2, ← c92390] at e_1508_1
  have hr := e_1508_1
  simp only [e_1508_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1508 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14679)) := by
  have c92391 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.1
  have c92392 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.1
  have c92393 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.1
  have c92394 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92395 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92396 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1508_2 := arithEq_of_rows h (row := 1508) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_2
  simp only [← c92391, ← c92392, ← c92393] at e_1508_2
  have e_1508_3 := arithEq_of_rows h (row := 1508) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_3
  simp only [← c92394, ← c92395, k2, ← c92396] at e_1508_3
  have hr := e_1508_3
  simp only [e_1508_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1508 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14680)) := by
  have c92397 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92398 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92399 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92400 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92401 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92402 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1508_4 := arithEq_of_rows h (row := 1508) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_4
  simp only [← c92397, ← c92398, ← c92399] at e_1508_4
  have e_1508_5 := arithEq_of_rows h (row := 1508) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_5
  simp only [← c92400, ← c92401, k2, ← c92402] at e_1508_5
  have hr := e_1508_5
  simp only [e_1508_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1508 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14681)) := by
  have c92403 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92404 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92405 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92406 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92407 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92408 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1508_6 := arithEq_of_rows h (row := 1508) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_6
  simp only [← c92403, ← c92404, ← c92405] at e_1508_6
  have e_1508_7 := arithEq_of_rows h (row := 1508) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_7
  simp only [← c92406, ← c92407, k2, ← c92408] at e_1508_7
  have hr := e_1508_7
  simp only [e_1508_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1508 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14682)) := by
  have c92409 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92410 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92411 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92412 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92413 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92414 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1508_8 := arithEq_of_rows h (row := 1508) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_8
  simp only [← c92409, ← c92410, ← c92411] at e_1508_8
  have e_1508_9 := arithEq_of_rows h (row := 1508) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_9
  simp only [← c92412, ← c92413, k2, ← c92414] at e_1508_9
  have hr := e_1508_9
  simp only [e_1508_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1508 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14683)) := by
  have c92415 := (publicBatchWrapper86_copies2887 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92416 := (publicBatchWrapper86_copies2888 a h).1
  have c92417 := (publicBatchWrapper86_copies2888 a h).2.1
  have c92418 := (publicBatchWrapper86_copies2888 a h).2.2.1
  have c92419 := (publicBatchWrapper86_copies2888 a h).2.2.2.1
  have c92420 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1508_10 := arithEq_of_rows h (row := 1508) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_10
  simp only [← c92415, ← c92416, ← c92417] at e_1508_10
  have e_1508_11 := arithEq_of_rows h (row := 1508) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_11
  simp only [← c92418, ← c92419, k2, ← c92420] at e_1508_11
  have hr := e_1508_11
  simp only [e_1508_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1508 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14684)) := by
  have c92421 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.1
  have c92422 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.1
  have c92423 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.1
  have c92424 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.1
  have c92425 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.1
  have c92426 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1508_12 := arithEq_of_rows h (row := 1508) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_12
  simp only [← c92421, ← c92422, ← c92423] at e_1508_12
  have e_1508_13 := arithEq_of_rows h (row := 1508) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_13
  simp only [← c92424, ← c92425, k2, ← c92426] at e_1508_13
  have hr := e_1508_13
  simp only [e_1508_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1508 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14685)) := by
  have c92427 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92428 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92429 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92430 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92431 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92432 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1508_14 := arithEq_of_rows h (row := 1508) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_14
  simp only [← c92427, ← c92428, ← c92429] at e_1508_14
  have e_1508_15 := arithEq_of_rows h (row := 1508) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_15
  simp only [← c92430, ← c92431, k2, ← c92432] at e_1508_15
  have hr := e_1508_15
  simp only [e_1508_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1508 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14686)) := by
  have c92433 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92434 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92435 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92436 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92437 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92438 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1508_16 := arithEq_of_rows h (row := 1508) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_16
  simp only [← c92433, ← c92434, ← c92435] at e_1508_16
  have e_1508_17 := arithEq_of_rows h (row := 1508) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_17
  simp only [← c92436, ← c92437, k2, ← c92438] at e_1508_17
  have hr := e_1508_17
  simp only [e_1508_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1508 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14687)) := by
  have c92439 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92440 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92441 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92442 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92443 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92444 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1508_18 := arithEq_of_rows h (row := 1508) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_18
  simp only [← c92439, ← c92440, ← c92441] at e_1508_18
  have e_1508_19 := arithEq_of_rows h (row := 1508) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1508_19
  simp only [← c92442, ← c92443, k2, ← c92444] at e_1508_19
  have hr := e_1508_19
  simp only [e_1508_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1509 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14688)) := by
  have c92445 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92446 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92447 := (publicBatchWrapper86_copies2888 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92448 := (publicBatchWrapper86_copies2889 a h).1
  have c92449 := (publicBatchWrapper86_copies2889 a h).2.1
  have c92450 := (publicBatchWrapper86_copies2889 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1509_0 := arithEq_of_rows h (row := 1509) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_0
  simp only [← c92445, ← c92446, ← c92447] at e_1509_0
  have e_1509_1 := arithEq_of_rows h (row := 1509) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_1
  simp only [← c92448, ← c92449, k2, ← c92450] at e_1509_1
  have hr := e_1509_1
  simp only [e_1509_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1509 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14689)) := by
  have c92451 := (publicBatchWrapper86_copies2889 a h).2.2.2.1
  have c92452 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.1
  have c92453 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.1
  have c92454 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.1
  have c92455 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.1
  have c92456 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1509_2 := arithEq_of_rows h (row := 1509) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_2
  simp only [← c92451, ← c92452, ← c92453] at e_1509_2
  have e_1509_3 := arithEq_of_rows h (row := 1509) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_3
  simp only [← c92454, ← c92455, k2, ← c92456] at e_1509_3
  have hr := e_1509_3
  simp only [e_1509_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1509 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14690)) := by
  have c92457 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.1
  have c92458 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92459 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92460 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92461 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92462 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1509_4 := arithEq_of_rows h (row := 1509) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_4
  simp only [← c92457, ← c92458, ← c92459] at e_1509_4
  have e_1509_5 := arithEq_of_rows h (row := 1509) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_5
  simp only [← c92460, ← c92461, k2, ← c92462] at e_1509_5
  have hr := e_1509_5
  simp only [e_1509_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1509 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14691)) := by
  have c92463 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92464 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92465 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92466 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92467 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92468 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1509_6 := arithEq_of_rows h (row := 1509) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_6
  simp only [← c92463, ← c92464, ← c92465] at e_1509_6
  have e_1509_7 := arithEq_of_rows h (row := 1509) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_7
  simp only [← c92466, ← c92467, k2, ← c92468] at e_1509_7
  have hr := e_1509_7
  simp only [e_1509_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1509 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14692)) := by
  have c92469 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92470 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92471 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92472 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92473 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92474 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1509_8 := arithEq_of_rows h (row := 1509) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_8
  simp only [← c92469, ← c92470, ← c92471] at e_1509_8
  have e_1509_9 := arithEq_of_rows h (row := 1509) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_9
  simp only [← c92472, ← c92473, k2, ← c92474] at e_1509_9
  have hr := e_1509_9
  simp only [e_1509_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1509 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14693)) := by
  have c92475 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92476 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92477 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92478 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92479 := (publicBatchWrapper86_copies2889 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92480 := (publicBatchWrapper86_copies2890 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1509_10 := arithEq_of_rows h (row := 1509) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_10
  simp only [← c92475, ← c92476, ← c92477] at e_1509_10
  have e_1509_11 := arithEq_of_rows h (row := 1509) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_11
  simp only [← c92478, ← c92479, k2, ← c92480] at e_1509_11
  have hr := e_1509_11
  simp only [e_1509_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1509 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14694)) := by
  have c92481 := (publicBatchWrapper86_copies2890 a h).2.1
  have c92482 := (publicBatchWrapper86_copies2890 a h).2.2.1
  have c92483 := (publicBatchWrapper86_copies2890 a h).2.2.2.1
  have c92484 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.1
  have c92485 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.1
  have c92486 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1509_12 := arithEq_of_rows h (row := 1509) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_12
  simp only [← c92481, ← c92482, ← c92483] at e_1509_12
  have e_1509_13 := arithEq_of_rows h (row := 1509) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_13
  simp only [← c92484, ← c92485, k2, ← c92486] at e_1509_13
  have hr := e_1509_13
  simp only [e_1509_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1509 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14695)) := by
  have c92487 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.1
  have c92488 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.1
  have c92489 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.1
  have c92490 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92491 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92492 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1509_14 := arithEq_of_rows h (row := 1509) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_14
  simp only [← c92487, ← c92488, ← c92489] at e_1509_14
  have e_1509_15 := arithEq_of_rows h (row := 1509) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_15
  simp only [← c92490, ← c92491, k2, ← c92492] at e_1509_15
  have hr := e_1509_15
  simp only [e_1509_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1509 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14696)) := by
  have c92493 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92494 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92495 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92496 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92497 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92498 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1509_16 := arithEq_of_rows h (row := 1509) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_16
  simp only [← c92493, ← c92494, ← c92495] at e_1509_16
  have e_1509_17 := arithEq_of_rows h (row := 1509) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_17
  simp only [← c92496, ← c92497, k2, ← c92498] at e_1509_17
  have hr := e_1509_17
  simp only [e_1509_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1509 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14697)) := by
  have c92499 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92500 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92501 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92502 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92503 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92504 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1509_18 := arithEq_of_rows h (row := 1509) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_18
  simp only [← c92499, ← c92500, ← c92501] at e_1509_18
  have e_1509_19 := arithEq_of_rows h (row := 1509) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1509_19
  simp only [← c92502, ← c92503, k2, ← c92504] at e_1509_19
  have hr := e_1509_19
  simp only [e_1509_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1510 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14698)) := by
  have c92505 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92506 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92507 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92508 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92509 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92510 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1510_0 := arithEq_of_rows h (row := 1510) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_0
  simp only [← c92505, ← c92506, ← c92507] at e_1510_0
  have e_1510_1 := arithEq_of_rows h (row := 1510) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_1
  simp only [← c92508, ← c92509, k2, ← c92510] at e_1510_1
  have hr := e_1510_1
  simp only [e_1510_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1510 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14699)) := by
  have c92511 := (publicBatchWrapper86_copies2890 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92512 := (publicBatchWrapper86_copies2891 a h).1
  have c92513 := (publicBatchWrapper86_copies2891 a h).2.1
  have c92514 := (publicBatchWrapper86_copies2891 a h).2.2.1
  have c92515 := (publicBatchWrapper86_copies2891 a h).2.2.2.1
  have c92516 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1510_2 := arithEq_of_rows h (row := 1510) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_2
  simp only [← c92511, ← c92512, ← c92513] at e_1510_2
  have e_1510_3 := arithEq_of_rows h (row := 1510) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_3
  simp only [← c92514, ← c92515, k2, ← c92516] at e_1510_3
  have hr := e_1510_3
  simp only [e_1510_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1510 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14700)) := by
  have c92517 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.1
  have c92518 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.1
  have c92519 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.1
  have c92520 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.1
  have c92521 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.1
  have c92522 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1510_4 := arithEq_of_rows h (row := 1510) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_4
  simp only [← c92517, ← c92518, ← c92519] at e_1510_4
  have e_1510_5 := arithEq_of_rows h (row := 1510) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_5
  simp only [← c92520, ← c92521, k2, ← c92522] at e_1510_5
  have hr := e_1510_5
  simp only [e_1510_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1510 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14701)) := by
  have c92523 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92524 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92525 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92526 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92527 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92528 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1510_6 := arithEq_of_rows h (row := 1510) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_6
  simp only [← c92523, ← c92524, ← c92525] at e_1510_6
  have e_1510_7 := arithEq_of_rows h (row := 1510) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_7
  simp only [← c92526, ← c92527, k2, ← c92528] at e_1510_7
  have hr := e_1510_7
  simp only [e_1510_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1510 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14702)) := by
  have c92529 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92530 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92531 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92532 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92533 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92534 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1510_8 := arithEq_of_rows h (row := 1510) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_8
  simp only [← c92529, ← c92530, ← c92531] at e_1510_8
  have e_1510_9 := arithEq_of_rows h (row := 1510) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_9
  simp only [← c92532, ← c92533, k2, ← c92534] at e_1510_9
  have hr := e_1510_9
  simp only [e_1510_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1510 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14703)) := by
  have c92535 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92536 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92537 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92538 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92539 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92540 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1510_10 := arithEq_of_rows h (row := 1510) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_10
  simp only [← c92535, ← c92536, ← c92537] at e_1510_10
  have e_1510_11 := arithEq_of_rows h (row := 1510) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_11
  simp only [← c92538, ← c92539, k2, ← c92540] at e_1510_11
  have hr := e_1510_11
  simp only [e_1510_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1510 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14704)) := by
  have c92541 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92542 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92543 := (publicBatchWrapper86_copies2891 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92544 := (publicBatchWrapper86_copies2892 a h).1
  have c92545 := (publicBatchWrapper86_copies2892 a h).2.1
  have c92546 := (publicBatchWrapper86_copies2892 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1510_12 := arithEq_of_rows h (row := 1510) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_12
  simp only [← c92541, ← c92542, ← c92543] at e_1510_12
  have e_1510_13 := arithEq_of_rows h (row := 1510) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_13
  simp only [← c92544, ← c92545, k2, ← c92546] at e_1510_13
  have hr := e_1510_13
  simp only [e_1510_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1510 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14705)) := by
  have c92547 := (publicBatchWrapper86_copies2892 a h).2.2.2.1
  have c92548 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.1
  have c92549 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.1
  have c92550 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.1
  have c92551 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.1
  have c92552 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1510_14 := arithEq_of_rows h (row := 1510) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_14
  simp only [← c92547, ← c92548, ← c92549] at e_1510_14
  have e_1510_15 := arithEq_of_rows h (row := 1510) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_15
  simp only [← c92550, ← c92551, k2, ← c92552] at e_1510_15
  have hr := e_1510_15
  simp only [e_1510_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1510 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14706)) := by
  have c92553 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.1
  have c92554 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92555 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92556 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92557 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92558 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1510_16 := arithEq_of_rows h (row := 1510) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_16
  simp only [← c92553, ← c92554, ← c92555] at e_1510_16
  have e_1510_17 := arithEq_of_rows h (row := 1510) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_17
  simp only [← c92556, ← c92557, k2, ← c92558] at e_1510_17
  have hr := e_1510_17
  simp only [e_1510_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1510 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14707)) := by
  have c92559 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92560 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92561 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92562 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92563 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92564 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1510_18 := arithEq_of_rows h (row := 1510) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_18
  simp only [← c92559, ← c92560, ← c92561] at e_1510_18
  have e_1510_19 := arithEq_of_rows h (row := 1510) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1510_19
  simp only [← c92562, ← c92563, k2, ← c92564] at e_1510_19
  have hr := e_1510_19
  simp only [e_1510_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1511 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14708)) := by
  have c92565 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92566 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92567 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92568 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92569 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92570 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1511_0 := arithEq_of_rows h (row := 1511) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_0
  simp only [← c92565, ← c92566, ← c92567] at e_1511_0
  have e_1511_1 := arithEq_of_rows h (row := 1511) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_1
  simp only [← c92568, ← c92569, k2, ← c92570] at e_1511_1
  have hr := e_1511_1
  simp only [e_1511_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1511 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14709)) := by
  have c92571 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92572 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92573 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92574 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92575 := (publicBatchWrapper86_copies2892 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92576 := (publicBatchWrapper86_copies2893 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1511_2 := arithEq_of_rows h (row := 1511) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_2
  simp only [← c92571, ← c92572, ← c92573] at e_1511_2
  have e_1511_3 := arithEq_of_rows h (row := 1511) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_3
  simp only [← c92574, ← c92575, k2, ← c92576] at e_1511_3
  have hr := e_1511_3
  simp only [e_1511_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1511 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14710)) := by
  have c92577 := (publicBatchWrapper86_copies2893 a h).2.1
  have c92578 := (publicBatchWrapper86_copies2893 a h).2.2.1
  have c92579 := (publicBatchWrapper86_copies2893 a h).2.2.2.1
  have c92580 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.1
  have c92581 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.1
  have c92582 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1511_4 := arithEq_of_rows h (row := 1511) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_4
  simp only [← c92577, ← c92578, ← c92579] at e_1511_4
  have e_1511_5 := arithEq_of_rows h (row := 1511) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_5
  simp only [← c92580, ← c92581, k2, ← c92582] at e_1511_5
  have hr := e_1511_5
  simp only [e_1511_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1511 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14711)) := by
  have c92583 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.1
  have c92584 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.1
  have c92585 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.1
  have c92586 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92587 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92588 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1511_6 := arithEq_of_rows h (row := 1511) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_6
  simp only [← c92583, ← c92584, ← c92585] at e_1511_6
  have e_1511_7 := arithEq_of_rows h (row := 1511) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_7
  simp only [← c92586, ← c92587, k2, ← c92588] at e_1511_7
  have hr := e_1511_7
  simp only [e_1511_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1511 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14712)) := by
  have c92589 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92590 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92591 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92592 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92593 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92594 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1511_8 := arithEq_of_rows h (row := 1511) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_8
  simp only [← c92589, ← c92590, ← c92591] at e_1511_8
  have e_1511_9 := arithEq_of_rows h (row := 1511) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_9
  simp only [← c92592, ← c92593, k2, ← c92594] at e_1511_9
  have hr := e_1511_9
  simp only [e_1511_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1511 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14713)) := by
  have c92595 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92596 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92597 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92598 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92599 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92600 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1511_10 := arithEq_of_rows h (row := 1511) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_10
  simp only [← c92595, ← c92596, ← c92597] at e_1511_10
  have e_1511_11 := arithEq_of_rows h (row := 1511) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_11
  simp only [← c92598, ← c92599, k2, ← c92600] at e_1511_11
  have hr := e_1511_11
  simp only [e_1511_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1511 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14714)) := by
  have c92601 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92602 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92603 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92604 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92605 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92606 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1511_12 := arithEq_of_rows h (row := 1511) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_12
  simp only [← c92601, ← c92602, ← c92603] at e_1511_12
  have e_1511_13 := arithEq_of_rows h (row := 1511) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_13
  simp only [← c92604, ← c92605, k2, ← c92606] at e_1511_13
  have hr := e_1511_13
  simp only [e_1511_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1511 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14715)) := by
  have c92607 := (publicBatchWrapper86_copies2893 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92608 := (publicBatchWrapper86_copies2894 a h).1
  have c92609 := (publicBatchWrapper86_copies2894 a h).2.1
  have c92610 := (publicBatchWrapper86_copies2894 a h).2.2.1
  have c92611 := (publicBatchWrapper86_copies2894 a h).2.2.2.1
  have c92612 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1511_14 := arithEq_of_rows h (row := 1511) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_14
  simp only [← c92607, ← c92608, ← c92609] at e_1511_14
  have e_1511_15 := arithEq_of_rows h (row := 1511) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_15
  simp only [← c92610, ← c92611, k2, ← c92612] at e_1511_15
  have hr := e_1511_15
  simp only [e_1511_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1511 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14716)) := by
  have c92613 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.1
  have c92614 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.1
  have c92615 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.1
  have c92616 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.1
  have c92617 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.1
  have c92618 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1511_16 := arithEq_of_rows h (row := 1511) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_16
  simp only [← c92613, ← c92614, ← c92615] at e_1511_16
  have e_1511_17 := arithEq_of_rows h (row := 1511) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_17
  simp only [← c92616, ← c92617, k2, ← c92618] at e_1511_17
  have hr := e_1511_17
  simp only [e_1511_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1511 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14717)) := by
  have c92619 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92620 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92621 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92622 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92623 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92624 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1511_18 := arithEq_of_rows h (row := 1511) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_18
  simp only [← c92619, ← c92620, ← c92621] at e_1511_18
  have e_1511_19 := arithEq_of_rows h (row := 1511) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1511_19
  simp only [← c92622, ← c92623, k2, ← c92624] at e_1511_19
  have hr := e_1511_19
  simp only [e_1511_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1512 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14718)) := by
  have c92625 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92626 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92627 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92628 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92629 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92630 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1512_0 := arithEq_of_rows h (row := 1512) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_0
  simp only [← c92625, ← c92626, ← c92627] at e_1512_0
  have e_1512_1 := arithEq_of_rows h (row := 1512) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_1
  simp only [← c92628, ← c92629, k2, ← c92630] at e_1512_1
  have hr := e_1512_1
  simp only [e_1512_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1512 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14719)) := by
  have c92631 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92632 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92633 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92634 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92635 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92636 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1512_2 := arithEq_of_rows h (row := 1512) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_2
  simp only [← c92631, ← c92632, ← c92633] at e_1512_2
  have e_1512_3 := arithEq_of_rows h (row := 1512) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_3
  simp only [← c92634, ← c92635, k2, ← c92636] at e_1512_3
  have hr := e_1512_3
  simp only [e_1512_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1512 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14720)) := by
  have c92637 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92638 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92639 := (publicBatchWrapper86_copies2894 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92640 := (publicBatchWrapper86_copies2895 a h).1
  have c92641 := (publicBatchWrapper86_copies2895 a h).2.1
  have c92642 := (publicBatchWrapper86_copies2895 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1512_4 := arithEq_of_rows h (row := 1512) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_4
  simp only [← c92637, ← c92638, ← c92639] at e_1512_4
  have e_1512_5 := arithEq_of_rows h (row := 1512) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_5
  simp only [← c92640, ← c92641, k2, ← c92642] at e_1512_5
  have hr := e_1512_5
  simp only [e_1512_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1512 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14721)) := by
  have c92643 := (publicBatchWrapper86_copies2895 a h).2.2.2.1
  have c92644 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.1
  have c92645 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.1
  have c92646 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.1
  have c92647 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.1
  have c92648 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1512_6 := arithEq_of_rows h (row := 1512) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_6
  simp only [← c92643, ← c92644, ← c92645] at e_1512_6
  have e_1512_7 := arithEq_of_rows h (row := 1512) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_7
  simp only [← c92646, ← c92647, k2, ← c92648] at e_1512_7
  have hr := e_1512_7
  simp only [e_1512_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1512 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14722)) := by
  have c92649 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.1
  have c92650 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92651 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92652 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92653 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92654 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1512_8 := arithEq_of_rows h (row := 1512) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_8
  simp only [← c92649, ← c92650, ← c92651] at e_1512_8
  have e_1512_9 := arithEq_of_rows h (row := 1512) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_9
  simp only [← c92652, ← c92653, k2, ← c92654] at e_1512_9
  have hr := e_1512_9
  simp only [e_1512_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1512 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14723)) := by
  have c92655 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92656 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92657 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92658 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92659 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92660 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1512_10 := arithEq_of_rows h (row := 1512) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_10
  simp only [← c92655, ← c92656, ← c92657] at e_1512_10
  have e_1512_11 := arithEq_of_rows h (row := 1512) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_11
  simp only [← c92658, ← c92659, k2, ← c92660] at e_1512_11
  have hr := e_1512_11
  simp only [e_1512_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1512 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14724)) := by
  have c92661 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92662 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92663 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92664 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92665 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92666 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1512_12 := arithEq_of_rows h (row := 1512) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_12
  simp only [← c92661, ← c92662, ← c92663] at e_1512_12
  have e_1512_13 := arithEq_of_rows h (row := 1512) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_13
  simp only [← c92664, ← c92665, k2, ← c92666] at e_1512_13
  have hr := e_1512_13
  simp only [e_1512_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1512 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14725)) := by
  have c92667 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92668 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92669 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92670 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92671 := (publicBatchWrapper86_copies2895 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92672 := (publicBatchWrapper86_copies2896 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1512_14 := arithEq_of_rows h (row := 1512) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_14
  simp only [← c92667, ← c92668, ← c92669] at e_1512_14
  have e_1512_15 := arithEq_of_rows h (row := 1512) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_15
  simp only [← c92670, ← c92671, k2, ← c92672] at e_1512_15
  have hr := e_1512_15
  simp only [e_1512_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1512 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14726)) := by
  have c92673 := (publicBatchWrapper86_copies2896 a h).2.1
  have c92674 := (publicBatchWrapper86_copies2896 a h).2.2.1
  have c92675 := (publicBatchWrapper86_copies2896 a h).2.2.2.1
  have c92676 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.1
  have c92677 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.1
  have c92678 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1512_16 := arithEq_of_rows h (row := 1512) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_16
  simp only [← c92673, ← c92674, ← c92675] at e_1512_16
  have e_1512_17 := arithEq_of_rows h (row := 1512) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_17
  simp only [← c92676, ← c92677, k2, ← c92678] at e_1512_17
  have hr := e_1512_17
  simp only [e_1512_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1512 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14727)) := by
  have c92679 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.1
  have c92680 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.1
  have c92681 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.1
  have c92682 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92683 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92684 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1512_18 := arithEq_of_rows h (row := 1512) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_18
  simp only [← c92679, ← c92680, ← c92681] at e_1512_18
  have e_1512_19 := arithEq_of_rows h (row := 1512) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1512_19
  simp only [← c92682, ← c92683, k2, ← c92684] at e_1512_19
  have hr := e_1512_19
  simp only [e_1512_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1513 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14728)) := by
  have c92685 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92686 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92687 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92688 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92689 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92690 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1513_0 := arithEq_of_rows h (row := 1513) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_0
  simp only [← c92685, ← c92686, ← c92687] at e_1513_0
  have e_1513_1 := arithEq_of_rows h (row := 1513) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_1
  simp only [← c92688, ← c92689, k2, ← c92690] at e_1513_1
  have hr := e_1513_1
  simp only [e_1513_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1513 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14729)) := by
  have c92691 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92692 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92693 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92694 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92695 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92696 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1513_2 := arithEq_of_rows h (row := 1513) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_2
  simp only [← c92691, ← c92692, ← c92693] at e_1513_2
  have e_1513_3 := arithEq_of_rows h (row := 1513) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_3
  simp only [← c92694, ← c92695, k2, ← c92696] at e_1513_3
  have hr := e_1513_3
  simp only [e_1513_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1513 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14730)) := by
  have c92697 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92698 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92699 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92700 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92701 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92702 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1513_4 := arithEq_of_rows h (row := 1513) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_4
  simp only [← c92697, ← c92698, ← c92699] at e_1513_4
  have e_1513_5 := arithEq_of_rows h (row := 1513) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_5
  simp only [← c92700, ← c92701, k2, ← c92702] at e_1513_5
  have hr := e_1513_5
  simp only [e_1513_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1513 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14731)) := by
  have c92703 := (publicBatchWrapper86_copies2896 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92704 := (publicBatchWrapper86_copies2897 a h).1
  have c92705 := (publicBatchWrapper86_copies2897 a h).2.1
  have c92706 := (publicBatchWrapper86_copies2897 a h).2.2.1
  have c92707 := (publicBatchWrapper86_copies2897 a h).2.2.2.1
  have c92708 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1513_6 := arithEq_of_rows h (row := 1513) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_6
  simp only [← c92703, ← c92704, ← c92705] at e_1513_6
  have e_1513_7 := arithEq_of_rows h (row := 1513) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_7
  simp only [← c92706, ← c92707, k2, ← c92708] at e_1513_7
  have hr := e_1513_7
  simp only [e_1513_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1513 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14732)) := by
  have c92709 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.1
  have c92710 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.1
  have c92711 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.1
  have c92712 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.1
  have c92713 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.1
  have c92714 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1513_8 := arithEq_of_rows h (row := 1513) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_8
  simp only [← c92709, ← c92710, ← c92711] at e_1513_8
  have e_1513_9 := arithEq_of_rows h (row := 1513) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_9
  simp only [← c92712, ← c92713, k2, ← c92714] at e_1513_9
  have hr := e_1513_9
  simp only [e_1513_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1513 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14733)) := by
  have c92715 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92716 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92717 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92718 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92719 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92720 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1513_10 := arithEq_of_rows h (row := 1513) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_10
  simp only [← c92715, ← c92716, ← c92717] at e_1513_10
  have e_1513_11 := arithEq_of_rows h (row := 1513) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_11
  simp only [← c92718, ← c92719, k2, ← c92720] at e_1513_11
  have hr := e_1513_11
  simp only [e_1513_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1513 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14734)) := by
  have c92721 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92722 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92723 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92724 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92725 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92726 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1513_12 := arithEq_of_rows h (row := 1513) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_12
  simp only [← c92721, ← c92722, ← c92723] at e_1513_12
  have e_1513_13 := arithEq_of_rows h (row := 1513) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_13
  simp only [← c92724, ← c92725, k2, ← c92726] at e_1513_13
  have hr := e_1513_13
  simp only [e_1513_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1513 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14735)) := by
  have c92727 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92728 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92729 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92730 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92731 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92732 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1513_14 := arithEq_of_rows h (row := 1513) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_14
  simp only [← c92727, ← c92728, ← c92729] at e_1513_14
  have e_1513_15 := arithEq_of_rows h (row := 1513) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_15
  simp only [← c92730, ← c92731, k2, ← c92732] at e_1513_15
  have hr := e_1513_15
  simp only [e_1513_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1513 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14736)) := by
  have c92733 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92734 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92735 := (publicBatchWrapper86_copies2897 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92736 := (publicBatchWrapper86_copies2898 a h).1
  have c92737 := (publicBatchWrapper86_copies2898 a h).2.1
  have c92738 := (publicBatchWrapper86_copies2898 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1513_16 := arithEq_of_rows h (row := 1513) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_16
  simp only [← c92733, ← c92734, ← c92735] at e_1513_16
  have e_1513_17 := arithEq_of_rows h (row := 1513) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_17
  simp only [← c92736, ← c92737, k2, ← c92738] at e_1513_17
  have hr := e_1513_17
  simp only [e_1513_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1513 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14737)) := by
  have c92739 := (publicBatchWrapper86_copies2898 a h).2.2.2.1
  have c92740 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.1
  have c92741 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.1
  have c92742 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.1
  have c92743 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.1
  have c92744 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1513_18 := arithEq_of_rows h (row := 1513) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_18
  simp only [← c92739, ← c92740, ← c92741] at e_1513_18
  have e_1513_19 := arithEq_of_rows h (row := 1513) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1513_19
  simp only [← c92742, ← c92743, k2, ← c92744] at e_1513_19
  have hr := e_1513_19
  simp only [e_1513_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1514 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14738)) := by
  have c92745 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.1
  have c92746 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92747 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92748 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92749 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92750 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1514_0 := arithEq_of_rows h (row := 1514) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_0
  simp only [← c92745, ← c92746, ← c92747] at e_1514_0
  have e_1514_1 := arithEq_of_rows h (row := 1514) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_1
  simp only [← c92748, ← c92749, k2, ← c92750] at e_1514_1
  have hr := e_1514_1
  simp only [e_1514_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1514 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14739)) := by
  have c92751 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92752 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92753 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92754 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92755 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92756 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1514_2 := arithEq_of_rows h (row := 1514) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_2
  simp only [← c92751, ← c92752, ← c92753] at e_1514_2
  have e_1514_3 := arithEq_of_rows h (row := 1514) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_3
  simp only [← c92754, ← c92755, k2, ← c92756] at e_1514_3
  have hr := e_1514_3
  simp only [e_1514_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1514 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14740)) := by
  have c92757 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92758 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92759 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92760 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92761 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92762 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1514_4 := arithEq_of_rows h (row := 1514) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_4
  simp only [← c92757, ← c92758, ← c92759] at e_1514_4
  have e_1514_5 := arithEq_of_rows h (row := 1514) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_5
  simp only [← c92760, ← c92761, k2, ← c92762] at e_1514_5
  have hr := e_1514_5
  simp only [e_1514_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1514 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14741)) := by
  have c92763 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92764 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92765 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92766 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92767 := (publicBatchWrapper86_copies2898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92768 := (publicBatchWrapper86_copies2899 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1514_6 := arithEq_of_rows h (row := 1514) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_6
  simp only [← c92763, ← c92764, ← c92765] at e_1514_6
  have e_1514_7 := arithEq_of_rows h (row := 1514) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_7
  simp only [← c92766, ← c92767, k2, ← c92768] at e_1514_7
  have hr := e_1514_7
  simp only [e_1514_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1514 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14742)) := by
  have c92769 := (publicBatchWrapper86_copies2899 a h).2.1
  have c92770 := (publicBatchWrapper86_copies2899 a h).2.2.1
  have c92771 := (publicBatchWrapper86_copies2899 a h).2.2.2.1
  have c92772 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.1
  have c92773 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.1
  have c92774 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1514_8 := arithEq_of_rows h (row := 1514) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_8
  simp only [← c92769, ← c92770, ← c92771] at e_1514_8
  have e_1514_9 := arithEq_of_rows h (row := 1514) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_9
  simp only [← c92772, ← c92773, k2, ← c92774] at e_1514_9
  have hr := e_1514_9
  simp only [e_1514_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1514 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14743)) := by
  have c92775 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.1
  have c92776 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.1
  have c92777 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.1
  have c92778 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92779 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92780 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1514_10 := arithEq_of_rows h (row := 1514) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_10
  simp only [← c92775, ← c92776, ← c92777] at e_1514_10
  have e_1514_11 := arithEq_of_rows h (row := 1514) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_11
  simp only [← c92778, ← c92779, k2, ← c92780] at e_1514_11
  have hr := e_1514_11
  simp only [e_1514_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1514 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14744)) := by
  have c92781 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92782 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92783 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92784 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92785 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92786 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1514_12 := arithEq_of_rows h (row := 1514) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_12
  simp only [← c92781, ← c92782, ← c92783] at e_1514_12
  have e_1514_13 := arithEq_of_rows h (row := 1514) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_13
  simp only [← c92784, ← c92785, k2, ← c92786] at e_1514_13
  have hr := e_1514_13
  simp only [e_1514_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1514 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14745)) := by
  have c92787 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92788 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92789 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92790 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92791 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92792 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1514_14 := arithEq_of_rows h (row := 1514) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_14
  simp only [← c92787, ← c92788, ← c92789] at e_1514_14
  have e_1514_15 := arithEq_of_rows h (row := 1514) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_15
  simp only [← c92790, ← c92791, k2, ← c92792] at e_1514_15
  have hr := e_1514_15
  simp only [e_1514_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1514 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14746)) := by
  have c92793 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92794 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92795 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92796 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92797 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92798 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1514_16 := arithEq_of_rows h (row := 1514) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_16
  simp only [← c92793, ← c92794, ← c92795] at e_1514_16
  have e_1514_17 := arithEq_of_rows h (row := 1514) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_17
  simp only [← c92796, ← c92797, k2, ← c92798] at e_1514_17
  have hr := e_1514_17
  simp only [e_1514_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1514 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14747)) := by
  have c92799 := (publicBatchWrapper86_copies2899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92800 := (publicBatchWrapper86_copies2900 a h).1
  have c92801 := (publicBatchWrapper86_copies2900 a h).2.1
  have c92802 := (publicBatchWrapper86_copies2900 a h).2.2.1
  have c92803 := (publicBatchWrapper86_copies2900 a h).2.2.2.1
  have c92804 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1514_18 := arithEq_of_rows h (row := 1514) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_18
  simp only [← c92799, ← c92800, ← c92801] at e_1514_18
  have e_1514_19 := arithEq_of_rows h (row := 1514) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1514_19
  simp only [← c92802, ← c92803, k2, ← c92804] at e_1514_19
  have hr := e_1514_19
  simp only [e_1514_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1515 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14748)) := by
  have c92805 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.1
  have c92806 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.1
  have c92807 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.1
  have c92808 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.1
  have c92809 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.1
  have c92810 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1515_0 := arithEq_of_rows h (row := 1515) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_0
  simp only [← c92805, ← c92806, ← c92807] at e_1515_0
  have e_1515_1 := arithEq_of_rows h (row := 1515) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_1
  simp only [← c92808, ← c92809, k2, ← c92810] at e_1515_1
  have hr := e_1515_1
  simp only [e_1515_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1515 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14749)) := by
  have c92811 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92812 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92813 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92814 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92815 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92816 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1515_2 := arithEq_of_rows h (row := 1515) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_2
  simp only [← c92811, ← c92812, ← c92813] at e_1515_2
  have e_1515_3 := arithEq_of_rows h (row := 1515) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_3
  simp only [← c92814, ← c92815, k2, ← c92816] at e_1515_3
  have hr := e_1515_3
  simp only [e_1515_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1515 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14750)) := by
  have c92817 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92818 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92819 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92820 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92821 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92822 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1515_4 := arithEq_of_rows h (row := 1515) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_4
  simp only [← c92817, ← c92818, ← c92819] at e_1515_4
  have e_1515_5 := arithEq_of_rows h (row := 1515) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_5
  simp only [← c92820, ← c92821, k2, ← c92822] at e_1515_5
  have hr := e_1515_5
  simp only [e_1515_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1515 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14751)) := by
  have c92823 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92824 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92825 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92826 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92827 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92828 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1515_6 := arithEq_of_rows h (row := 1515) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_6
  simp only [← c92823, ← c92824, ← c92825] at e_1515_6
  have e_1515_7 := arithEq_of_rows h (row := 1515) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_7
  simp only [← c92826, ← c92827, k2, ← c92828] at e_1515_7
  have hr := e_1515_7
  simp only [e_1515_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1515 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14752)) := by
  have c92829 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92830 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92831 := (publicBatchWrapper86_copies2900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92832 := (publicBatchWrapper86_copies2901 a h).1
  have c92833 := (publicBatchWrapper86_copies2901 a h).2.1
  have c92834 := (publicBatchWrapper86_copies2901 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1515_8 := arithEq_of_rows h (row := 1515) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_8
  simp only [← c92829, ← c92830, ← c92831] at e_1515_8
  have e_1515_9 := arithEq_of_rows h (row := 1515) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_9
  simp only [← c92832, ← c92833, k2, ← c92834] at e_1515_9
  have hr := e_1515_9
  simp only [e_1515_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1515 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14753)) := by
  have c92835 := (publicBatchWrapper86_copies2901 a h).2.2.2.1
  have c92836 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.1
  have c92837 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.1
  have c92838 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.1
  have c92839 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.1
  have c92840 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1515_10 := arithEq_of_rows h (row := 1515) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_10
  simp only [← c92835, ← c92836, ← c92837] at e_1515_10
  have e_1515_11 := arithEq_of_rows h (row := 1515) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_11
  simp only [← c92838, ← c92839, k2, ← c92840] at e_1515_11
  have hr := e_1515_11
  simp only [e_1515_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1515 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14754)) := by
  have c92841 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.1
  have c92842 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92843 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92844 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92845 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92846 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1515_12 := arithEq_of_rows h (row := 1515) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_12
  simp only [← c92841, ← c92842, ← c92843] at e_1515_12
  have e_1515_13 := arithEq_of_rows h (row := 1515) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_13
  simp only [← c92844, ← c92845, k2, ← c92846] at e_1515_13
  have hr := e_1515_13
  simp only [e_1515_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1515 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14755)) := by
  have c92847 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92848 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92849 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92850 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92851 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92852 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1515_14 := arithEq_of_rows h (row := 1515) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_14
  simp only [← c92847, ← c92848, ← c92849] at e_1515_14
  have e_1515_15 := arithEq_of_rows h (row := 1515) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_15
  simp only [← c92850, ← c92851, k2, ← c92852] at e_1515_15
  have hr := e_1515_15
  simp only [e_1515_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1515 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14756)) := by
  have c92853 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92854 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92855 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92856 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92857 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92858 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1515_16 := arithEq_of_rows h (row := 1515) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_16
  simp only [← c92853, ← c92854, ← c92855] at e_1515_16
  have e_1515_17 := arithEq_of_rows h (row := 1515) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_17
  simp only [← c92856, ← c92857, k2, ← c92858] at e_1515_17
  have hr := e_1515_17
  simp only [e_1515_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1515 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14757)) := by
  have c92859 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92860 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92861 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92862 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92863 := (publicBatchWrapper86_copies2901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92864 := (publicBatchWrapper86_copies2902 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1515_18 := arithEq_of_rows h (row := 1515) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_18
  simp only [← c92859, ← c92860, ← c92861] at e_1515_18
  have e_1515_19 := arithEq_of_rows h (row := 1515) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1515_19
  simp only [← c92862, ← c92863, k2, ← c92864] at e_1515_19
  have hr := e_1515_19
  simp only [e_1515_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1516 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14758)) := by
  have c92865 := (publicBatchWrapper86_copies2902 a h).2.1
  have c92866 := (publicBatchWrapper86_copies2902 a h).2.2.1
  have c92867 := (publicBatchWrapper86_copies2902 a h).2.2.2.1
  have c92868 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.1
  have c92869 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.1
  have c92870 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1516_0 := arithEq_of_rows h (row := 1516) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_0
  simp only [← c92865, ← c92866, ← c92867] at e_1516_0
  have e_1516_1 := arithEq_of_rows h (row := 1516) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_1
  simp only [← c92868, ← c92869, k2, ← c92870] at e_1516_1
  have hr := e_1516_1
  simp only [e_1516_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1516 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14759)) := by
  have c92871 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.1
  have c92872 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.1
  have c92873 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.1
  have c92874 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92875 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92876 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1516_2 := arithEq_of_rows h (row := 1516) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_2
  simp only [← c92871, ← c92872, ← c92873] at e_1516_2
  have e_1516_3 := arithEq_of_rows h (row := 1516) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_3
  simp only [← c92874, ← c92875, k2, ← c92876] at e_1516_3
  have hr := e_1516_3
  simp only [e_1516_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1516 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14760)) := by
  have c92877 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92878 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92879 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92880 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92881 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92882 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1516_4 := arithEq_of_rows h (row := 1516) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_4
  simp only [← c92877, ← c92878, ← c92879] at e_1516_4
  have e_1516_5 := arithEq_of_rows h (row := 1516) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_5
  simp only [← c92880, ← c92881, k2, ← c92882] at e_1516_5
  have hr := e_1516_5
  simp only [e_1516_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1516 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14761)) := by
  have c92883 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92884 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92885 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92886 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92887 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92888 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1516_6 := arithEq_of_rows h (row := 1516) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_6
  simp only [← c92883, ← c92884, ← c92885] at e_1516_6
  have e_1516_7 := arithEq_of_rows h (row := 1516) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_7
  simp only [← c92886, ← c92887, k2, ← c92888] at e_1516_7
  have hr := e_1516_7
  simp only [e_1516_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1516 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14762)) := by
  have c92889 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92890 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92891 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92892 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92893 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92894 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1516_8 := arithEq_of_rows h (row := 1516) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_8
  simp only [← c92889, ← c92890, ← c92891] at e_1516_8
  have e_1516_9 := arithEq_of_rows h (row := 1516) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_9
  simp only [← c92892, ← c92893, k2, ← c92894] at e_1516_9
  have hr := e_1516_9
  simp only [e_1516_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1516 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14763)) := by
  have c92895 := (publicBatchWrapper86_copies2902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92896 := (publicBatchWrapper86_copies2903 a h).1
  have c92897 := (publicBatchWrapper86_copies2903 a h).2.1
  have c92898 := (publicBatchWrapper86_copies2903 a h).2.2.1
  have c92899 := (publicBatchWrapper86_copies2903 a h).2.2.2.1
  have c92900 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1516_10 := arithEq_of_rows h (row := 1516) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_10
  simp only [← c92895, ← c92896, ← c92897] at e_1516_10
  have e_1516_11 := arithEq_of_rows h (row := 1516) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_11
  simp only [← c92898, ← c92899, k2, ← c92900] at e_1516_11
  have hr := e_1516_11
  simp only [e_1516_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1516 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14764)) := by
  have c92901 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.1
  have c92902 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.1
  have c92903 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.1
  have c92904 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.1
  have c92905 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.1
  have c92906 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1516_12 := arithEq_of_rows h (row := 1516) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_12
  simp only [← c92901, ← c92902, ← c92903] at e_1516_12
  have e_1516_13 := arithEq_of_rows h (row := 1516) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_13
  simp only [← c92904, ← c92905, k2, ← c92906] at e_1516_13
  have hr := e_1516_13
  simp only [e_1516_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1516 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14765)) := by
  have c92907 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92908 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92909 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92910 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92911 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92912 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1516_14 := arithEq_of_rows h (row := 1516) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_14
  simp only [← c92907, ← c92908, ← c92909] at e_1516_14
  have e_1516_15 := arithEq_of_rows h (row := 1516) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_15
  simp only [← c92910, ← c92911, k2, ← c92912] at e_1516_15
  have hr := e_1516_15
  simp only [e_1516_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1516 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14766)) := by
  have c92913 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92914 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92915 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92916 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92917 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92918 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1516_16 := arithEq_of_rows h (row := 1516) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_16
  simp only [← c92913, ← c92914, ← c92915] at e_1516_16
  have e_1516_17 := arithEq_of_rows h (row := 1516) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_17
  simp only [← c92916, ← c92917, k2, ← c92918] at e_1516_17
  have hr := e_1516_17
  simp only [e_1516_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1516 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14767)) := by
  have c92919 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92920 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92921 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92922 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92923 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92924 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1516_18 := arithEq_of_rows h (row := 1516) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_18
  simp only [← c92919, ← c92920, ← c92921] at e_1516_18
  have e_1516_19 := arithEq_of_rows h (row := 1516) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1516_19
  simp only [← c92922, ← c92923, k2, ← c92924] at e_1516_19
  have hr := e_1516_19
  simp only [e_1516_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1517 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14768)) := by
  have c92925 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92926 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92927 := (publicBatchWrapper86_copies2903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92928 := (publicBatchWrapper86_copies2904 a h).1
  have c92929 := (publicBatchWrapper86_copies2904 a h).2.1
  have c92930 := (publicBatchWrapper86_copies2904 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1517_0 := arithEq_of_rows h (row := 1517) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_0
  simp only [← c92925, ← c92926, ← c92927] at e_1517_0
  have e_1517_1 := arithEq_of_rows h (row := 1517) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_1
  simp only [← c92928, ← c92929, k2, ← c92930] at e_1517_1
  have hr := e_1517_1
  simp only [e_1517_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1517 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14769)) := by
  have c92931 := (publicBatchWrapper86_copies2904 a h).2.2.2.1
  have c92932 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.1
  have c92933 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.1
  have c92934 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.1
  have c92935 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.1
  have c92936 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1517_2 := arithEq_of_rows h (row := 1517) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_2
  simp only [← c92931, ← c92932, ← c92933] at e_1517_2
  have e_1517_3 := arithEq_of_rows h (row := 1517) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_3
  simp only [← c92934, ← c92935, k2, ← c92936] at e_1517_3
  have hr := e_1517_3
  simp only [e_1517_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1517 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14810)) := by
  have c92937 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.1
  have c92938 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92939 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92940 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92941 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92942 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1517_4 := arithEq_of_rows h (row := 1517) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_4
  simp only [← c92937, ← c92938, ← c92939] at e_1517_4
  have e_1517_5 := arithEq_of_rows h (row := 1517) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_5
  simp only [← c92940, ← c92941, k2, ← c92942] at e_1517_5
  have hr := e_1517_5
  simp only [e_1517_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1517 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14811)) := by
  have c92943 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92944 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92945 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92946 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92947 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92948 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1517_6 := arithEq_of_rows h (row := 1517) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_6
  simp only [← c92943, ← c92944, ← c92945] at e_1517_6
  have e_1517_7 := arithEq_of_rows h (row := 1517) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_7
  simp only [← c92946, ← c92947, k2, ← c92948] at e_1517_7
  have hr := e_1517_7
  simp only [e_1517_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1517 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14812)) := by
  have c92949 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92950 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92951 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92952 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92953 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92954 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1517_8 := arithEq_of_rows h (row := 1517) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_8
  simp only [← c92949, ← c92950, ← c92951] at e_1517_8
  have e_1517_9 := arithEq_of_rows h (row := 1517) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_9
  simp only [← c92952, ← c92953, k2, ← c92954] at e_1517_9
  have hr := e_1517_9
  simp only [e_1517_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1517 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14813)) := by
  have c92955 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92956 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92957 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92958 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92959 := (publicBatchWrapper86_copies2904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92960 := (publicBatchWrapper86_copies2905 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1517_10 := arithEq_of_rows h (row := 1517) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_10
  simp only [← c92955, ← c92956, ← c92957] at e_1517_10
  have e_1517_11 := arithEq_of_rows h (row := 1517) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_11
  simp only [← c92958, ← c92959, k2, ← c92960] at e_1517_11
  have hr := e_1517_11
  simp only [e_1517_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1517 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14814)) := by
  have c92961 := (publicBatchWrapper86_copies2905 a h).2.1
  have c92962 := (publicBatchWrapper86_copies2905 a h).2.2.1
  have c92963 := (publicBatchWrapper86_copies2905 a h).2.2.2.1
  have c92964 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.1
  have c92965 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.1
  have c92966 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1517_12 := arithEq_of_rows h (row := 1517) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_12
  simp only [← c92961, ← c92962, ← c92963] at e_1517_12
  have e_1517_13 := arithEq_of_rows h (row := 1517) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_13
  simp only [← c92964, ← c92965, k2, ← c92966] at e_1517_13
  have hr := e_1517_13
  simp only [e_1517_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1517 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14815)) := by
  have c92967 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.1
  have c92968 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.1
  have c92969 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.1
  have c92970 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92971 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92972 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1517_14 := arithEq_of_rows h (row := 1517) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_14
  simp only [← c92967, ← c92968, ← c92969] at e_1517_14
  have e_1517_15 := arithEq_of_rows h (row := 1517) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_15
  simp only [← c92970, ← c92971, k2, ← c92972] at e_1517_15
  have hr := e_1517_15
  simp only [e_1517_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1517 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14816)) := by
  have c92973 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92974 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92975 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92976 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92977 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92978 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1517_16 := arithEq_of_rows h (row := 1517) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_16
  simp only [← c92973, ← c92974, ← c92975] at e_1517_16
  have e_1517_17 := arithEq_of_rows h (row := 1517) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_17
  simp only [← c92976, ← c92977, k2, ← c92978] at e_1517_17
  have hr := e_1517_17
  simp only [e_1517_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1517 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14817)) := by
  have c92979 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92980 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92981 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92982 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92983 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92984 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1517_18 := arithEq_of_rows h (row := 1517) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_18
  simp only [← c92979, ← c92980, ← c92981] at e_1517_18
  have e_1517_19 := arithEq_of_rows h (row := 1517) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1517_19
  simp only [← c92982, ← c92983, k2, ← c92984] at e_1517_19
  have hr := e_1517_19
  simp only [e_1517_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1518 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14818)) := by
  have c92985 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92986 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92987 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92988 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92989 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92990 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1518_0 := arithEq_of_rows h (row := 1518) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_0
  simp only [← c92985, ← c92986, ← c92987] at e_1518_0
  have e_1518_1 := arithEq_of_rows h (row := 1518) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_1
  simp only [← c92988, ← c92989, k2, ← c92990] at e_1518_1
  have hr := e_1518_1
  simp only [e_1518_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1518 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14819)) := by
  have c92991 := (publicBatchWrapper86_copies2905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92992 := (publicBatchWrapper86_copies2906 a h).1
  have c92993 := (publicBatchWrapper86_copies2906 a h).2.1
  have c92994 := (publicBatchWrapper86_copies2906 a h).2.2.1
  have c92995 := (publicBatchWrapper86_copies2906 a h).2.2.2.1
  have c92996 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1518_2 := arithEq_of_rows h (row := 1518) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_2
  simp only [← c92991, ← c92992, ← c92993] at e_1518_2
  have e_1518_3 := arithEq_of_rows h (row := 1518) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_3
  simp only [← c92994, ← c92995, k2, ← c92996] at e_1518_3
  have hr := e_1518_3
  simp only [e_1518_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1518 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14820)) := by
  have c92997 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.1
  have c92998 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.1
  have c92999 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.1
  have c93000 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.1
  have c93001 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.1
  have c93002 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1518_4 := arithEq_of_rows h (row := 1518) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_4
  simp only [← c92997, ← c92998, ← c92999] at e_1518_4
  have e_1518_5 := arithEq_of_rows h (row := 1518) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_5
  simp only [← c93000, ← c93001, k2, ← c93002] at e_1518_5
  have hr := e_1518_5
  simp only [e_1518_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1518 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14821)) := by
  have c93003 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93004 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93005 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93006 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93007 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93008 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1518_6 := arithEq_of_rows h (row := 1518) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_6
  simp only [← c93003, ← c93004, ← c93005] at e_1518_6
  have e_1518_7 := arithEq_of_rows h (row := 1518) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_7
  simp only [← c93006, ← c93007, k2, ← c93008] at e_1518_7
  have hr := e_1518_7
  simp only [e_1518_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1518 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14822)) := by
  have c93009 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93010 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93011 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93012 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93013 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93014 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1518_8 := arithEq_of_rows h (row := 1518) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_8
  simp only [← c93009, ← c93010, ← c93011] at e_1518_8
  have e_1518_9 := arithEq_of_rows h (row := 1518) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_9
  simp only [← c93012, ← c93013, k2, ← c93014] at e_1518_9
  have hr := e_1518_9
  simp only [e_1518_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1518 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14823)) := by
  have c93015 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93016 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93017 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93018 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93019 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93020 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1518_10 := arithEq_of_rows h (row := 1518) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_10
  simp only [← c93015, ← c93016, ← c93017] at e_1518_10
  have e_1518_11 := arithEq_of_rows h (row := 1518) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_11
  simp only [← c93018, ← c93019, k2, ← c93020] at e_1518_11
  have hr := e_1518_11
  simp only [e_1518_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1518 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14824)) := by
  have c93021 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93022 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93023 := (publicBatchWrapper86_copies2906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93024 := (publicBatchWrapper86_copies2907 a h).1
  have c93025 := (publicBatchWrapper86_copies2907 a h).2.1
  have c93026 := (publicBatchWrapper86_copies2907 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1518_12 := arithEq_of_rows h (row := 1518) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_12
  simp only [← c93021, ← c93022, ← c93023] at e_1518_12
  have e_1518_13 := arithEq_of_rows h (row := 1518) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_13
  simp only [← c93024, ← c93025, k2, ← c93026] at e_1518_13
  have hr := e_1518_13
  simp only [e_1518_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1518 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14825)) := by
  have c93027 := (publicBatchWrapper86_copies2907 a h).2.2.2.1
  have c93028 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.1
  have c93029 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.1
  have c93030 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.1
  have c93031 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.1
  have c93032 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1518_14 := arithEq_of_rows h (row := 1518) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_14
  simp only [← c93027, ← c93028, ← c93029] at e_1518_14
  have e_1518_15 := arithEq_of_rows h (row := 1518) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_15
  simp only [← c93030, ← c93031, k2, ← c93032] at e_1518_15
  have hr := e_1518_15
  simp only [e_1518_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1518 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14826)) := by
  have c93033 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.1
  have c93034 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93035 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93036 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93037 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93038 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1518_16 := arithEq_of_rows h (row := 1518) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_16
  simp only [← c93033, ← c93034, ← c93035] at e_1518_16
  have e_1518_17 := arithEq_of_rows h (row := 1518) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_17
  simp only [← c93036, ← c93037, k2, ← c93038] at e_1518_17
  have hr := e_1518_17
  simp only [e_1518_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14781 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1518 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14827)) := by
  have c93039 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93040 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93041 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93042 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93043 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93044 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1518_18 := arithEq_of_rows h (row := 1518) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_18
  simp only [← c93039, ← c93040, ← c93041] at e_1518_18
  have e_1518_19 := arithEq_of_rows h (row := 1518) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1518_19
  simp only [← c93042, ← c93043, k2, ← c93044] at e_1518_19
  have hr := e_1518_19
  simp only [e_1518_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14782 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1519 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14828)) := by
  have c93045 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93046 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93047 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93048 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93049 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93050 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1519_0 := arithEq_of_rows h (row := 1519) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_0
  simp only [← c93045, ← c93046, ← c93047] at e_1519_0
  have e_1519_1 := arithEq_of_rows h (row := 1519) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_1
  simp only [← c93048, ← c93049, k2, ← c93050] at e_1519_1
  have hr := e_1519_1
  simp only [e_1519_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14783 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1519 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14829)) := by
  have c93051 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93052 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93053 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93054 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93055 := (publicBatchWrapper86_copies2907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93056 := (publicBatchWrapper86_copies2908 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1519_2 := arithEq_of_rows h (row := 1519) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_2
  simp only [← c93051, ← c93052, ← c93053] at e_1519_2
  have e_1519_3 := arithEq_of_rows h (row := 1519) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_3
  simp only [← c93054, ← c93055, k2, ← c93056] at e_1519_3
  have hr := e_1519_3
  simp only [e_1519_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14784 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1519 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14830)) := by
  have c93057 := (publicBatchWrapper86_copies2908 a h).2.1
  have c93058 := (publicBatchWrapper86_copies2908 a h).2.2.1
  have c93059 := (publicBatchWrapper86_copies2908 a h).2.2.2.1
  have c93060 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.1
  have c93061 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.1
  have c93062 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1519_4 := arithEq_of_rows h (row := 1519) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_4
  simp only [← c93057, ← c93058, ← c93059] at e_1519_4
  have e_1519_5 := arithEq_of_rows h (row := 1519) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_5
  simp only [← c93060, ← c93061, k2, ← c93062] at e_1519_5
  have hr := e_1519_5
  simp only [e_1519_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14785 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1519 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14831)) := by
  have c93063 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.1
  have c93064 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.1
  have c93065 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.1
  have c93066 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93067 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93068 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1519_6 := arithEq_of_rows h (row := 1519) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_6
  simp only [← c93063, ← c93064, ← c93065] at e_1519_6
  have e_1519_7 := arithEq_of_rows h (row := 1519) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_7
  simp only [← c93066, ← c93067, k2, ← c93068] at e_1519_7
  have hr := e_1519_7
  simp only [e_1519_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14786 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1519 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14832)) := by
  have c93069 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93070 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93071 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93072 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93073 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93074 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1519_8 := arithEq_of_rows h (row := 1519) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_8
  simp only [← c93069, ← c93070, ← c93071] at e_1519_8
  have e_1519_9 := arithEq_of_rows h (row := 1519) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_9
  simp only [← c93072, ← c93073, k2, ← c93074] at e_1519_9
  have hr := e_1519_9
  simp only [e_1519_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14787 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1519 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14833)) := by
  have c93075 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93076 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93077 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93078 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93079 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93080 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1519_10 := arithEq_of_rows h (row := 1519) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_10
  simp only [← c93075, ← c93076, ← c93077] at e_1519_10
  have e_1519_11 := arithEq_of_rows h (row := 1519) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_11
  simp only [← c93078, ← c93079, k2, ← c93080] at e_1519_11
  have hr := e_1519_11
  simp only [e_1519_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14788 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1519 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14834)) := by
  have c93081 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93082 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93083 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93084 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93085 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93086 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1519_12 := arithEq_of_rows h (row := 1519) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_12
  simp only [← c93081, ← c93082, ← c93083] at e_1519_12
  have e_1519_13 := arithEq_of_rows h (row := 1519) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_13
  simp only [← c93084, ← c93085, k2, ← c93086] at e_1519_13
  have hr := e_1519_13
  simp only [e_1519_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14789 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1519 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14835)) := by
  have c93087 := (publicBatchWrapper86_copies2908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93088 := (publicBatchWrapper86_copies2909 a h).1
  have c93089 := (publicBatchWrapper86_copies2909 a h).2.1
  have c93090 := (publicBatchWrapper86_copies2909 a h).2.2.1
  have c93091 := (publicBatchWrapper86_copies2909 a h).2.2.2.1
  have c93092 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1519_14 := arithEq_of_rows h (row := 1519) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_14
  simp only [← c93087, ← c93088, ← c93089] at e_1519_14
  have e_1519_15 := arithEq_of_rows h (row := 1519) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_15
  simp only [← c93090, ← c93091, k2, ← c93092] at e_1519_15
  have hr := e_1519_15
  simp only [e_1519_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14790 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1519 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14836)) := by
  have c93093 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.1
  have c93094 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.1
  have c93095 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.1
  have c93096 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.1
  have c93097 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.1
  have c93098 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1519_16 := arithEq_of_rows h (row := 1519) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_16
  simp only [← c93093, ← c93094, ← c93095] at e_1519_16
  have e_1519_17 := arithEq_of_rows h (row := 1519) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_17
  simp only [← c93096, ← c93097, k2, ← c93098] at e_1519_17
  have hr := e_1519_17
  simp only [e_1519_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14791 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1519 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14837)) := by
  have c93099 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93100 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93101 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93102 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93103 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93104 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1519_18 := arithEq_of_rows h (row := 1519) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_18
  simp only [← c93099, ← c93100, ← c93101] at e_1519_18
  have e_1519_19 := arithEq_of_rows h (row := 1519) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1519_19
  simp only [← c93102, ← c93103, k2, ← c93104] at e_1519_19
  have hr := e_1519_19
  simp only [e_1519_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14792 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1520 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14838)) := by
  have c93105 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93106 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93107 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93108 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93109 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93110 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1520_0 := arithEq_of_rows h (row := 1520) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_0
  simp only [← c93105, ← c93106, ← c93107] at e_1520_0
  have e_1520_1 := arithEq_of_rows h (row := 1520) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_1
  simp only [← c93108, ← c93109, k2, ← c93110] at e_1520_1
  have hr := e_1520_1
  simp only [e_1520_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14793 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1520 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14839)) := by
  have c93111 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93112 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93113 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93114 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93115 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93116 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1520_2 := arithEq_of_rows h (row := 1520) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_2
  simp only [← c93111, ← c93112, ← c93113] at e_1520_2
  have e_1520_3 := arithEq_of_rows h (row := 1520) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_3
  simp only [← c93114, ← c93115, k2, ← c93116] at e_1520_3
  have hr := e_1520_3
  simp only [e_1520_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14794 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1520 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14840)) := by
  have c93117 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93118 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93119 := (publicBatchWrapper86_copies2909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93120 := (publicBatchWrapper86_copies2910 a h).1
  have c93121 := (publicBatchWrapper86_copies2910 a h).2.1
  have c93122 := (publicBatchWrapper86_copies2910 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1520_4 := arithEq_of_rows h (row := 1520) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_4
  simp only [← c93117, ← c93118, ← c93119] at e_1520_4
  have e_1520_5 := arithEq_of_rows h (row := 1520) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_5
  simp only [← c93120, ← c93121, k2, ← c93122] at e_1520_5
  have hr := e_1520_5
  simp only [e_1520_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14795 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1520 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14841)) := by
  have c93123 := (publicBatchWrapper86_copies2910 a h).2.2.2.1
  have c93124 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.1
  have c93125 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.1
  have c93126 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.1
  have c93127 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.1
  have c93128 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1520_6 := arithEq_of_rows h (row := 1520) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_6
  simp only [← c93123, ← c93124, ← c93125] at e_1520_6
  have e_1520_7 := arithEq_of_rows h (row := 1520) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_7
  simp only [← c93126, ← c93127, k2, ← c93128] at e_1520_7
  have hr := e_1520_7
  simp only [e_1520_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14796 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1520 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14842)) := by
  have c93129 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.1
  have c93130 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93131 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93132 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93133 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93134 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1520_8 := arithEq_of_rows h (row := 1520) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_8
  simp only [← c93129, ← c93130, ← c93131] at e_1520_8
  have e_1520_9 := arithEq_of_rows h (row := 1520) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_9
  simp only [← c93132, ← c93133, k2, ← c93134] at e_1520_9
  have hr := e_1520_9
  simp only [e_1520_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14797 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1520 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14843)) := by
  have c93135 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93136 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93137 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93138 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93139 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93140 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1520_10 := arithEq_of_rows h (row := 1520) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_10
  simp only [← c93135, ← c93136, ← c93137] at e_1520_10
  have e_1520_11 := arithEq_of_rows h (row := 1520) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_11
  simp only [← c93138, ← c93139, k2, ← c93140] at e_1520_11
  have hr := e_1520_11
  simp only [e_1520_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14798 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1520 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14844)) := by
  have c93141 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93142 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93143 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93144 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93145 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93146 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1520_12 := arithEq_of_rows h (row := 1520) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_12
  simp only [← c93141, ← c93142, ← c93143] at e_1520_12
  have e_1520_13 := arithEq_of_rows h (row := 1520) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_13
  simp only [← c93144, ← c93145, k2, ← c93146] at e_1520_13
  have hr := e_1520_13
  simp only [e_1520_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14799 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1520 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14845)) := by
  have c93147 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93148 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93149 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93150 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93151 := (publicBatchWrapper86_copies2910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93152 := (publicBatchWrapper86_copies2911 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1520_14 := arithEq_of_rows h (row := 1520) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_14
  simp only [← c93147, ← c93148, ← c93149] at e_1520_14
  have e_1520_15 := arithEq_of_rows h (row := 1520) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_15
  simp only [← c93150, ← c93151, k2, ← c93152] at e_1520_15
  have hr := e_1520_15
  simp only [e_1520_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1520 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14846)) := by
  have c93153 := (publicBatchWrapper86_copies2911 a h).2.1
  have c93154 := (publicBatchWrapper86_copies2911 a h).2.2.1
  have c93155 := (publicBatchWrapper86_copies2911 a h).2.2.2.1
  have c93156 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.1
  have c93157 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.1
  have c93158 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1520_16 := arithEq_of_rows h (row := 1520) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_16
  simp only [← c93153, ← c93154, ← c93155] at e_1520_16
  have e_1520_17 := arithEq_of_rows h (row := 1520) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_17
  simp only [← c93156, ← c93157, k2, ← c93158] at e_1520_17
  have hr := e_1520_17
  simp only [e_1520_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1520 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14847)) := by
  have c93159 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.1
  have c93160 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.1
  have c93161 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.1
  have c93162 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93163 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93164 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1520_18 := arithEq_of_rows h (row := 1520) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_18
  simp only [← c93159, ← c93160, ← c93161] at e_1520_18
  have e_1520_19 := arithEq_of_rows h (row := 1520) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1520_19
  simp only [← c93162, ← c93163, k2, ← c93164] at e_1520_19
  have hr := e_1520_19
  simp only [e_1520_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1521 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14848)) := by
  have c93165 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93166 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93167 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93168 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93169 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93170 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1521_0 := arithEq_of_rows h (row := 1521) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_0
  simp only [← c93165, ← c93166, ← c93167] at e_1521_0
  have e_1521_1 := arithEq_of_rows h (row := 1521) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_1
  simp only [← c93168, ← c93169, k2, ← c93170] at e_1521_1
  have hr := e_1521_1
  simp only [e_1521_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1521 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14849)) := by
  have c93171 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93172 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93173 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93174 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93175 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93176 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1521_2 := arithEq_of_rows h (row := 1521) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_2
  simp only [← c93171, ← c93172, ← c93173] at e_1521_2
  have e_1521_3 := arithEq_of_rows h (row := 1521) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_3
  simp only [← c93174, ← c93175, k2, ← c93176] at e_1521_3
  have hr := e_1521_3
  simp only [e_1521_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1521 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14850)) := by
  have c93177 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93178 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93179 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93180 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93181 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93182 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1521_4 := arithEq_of_rows h (row := 1521) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_4
  simp only [← c93177, ← c93178, ← c93179] at e_1521_4
  have e_1521_5 := arithEq_of_rows h (row := 1521) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_5
  simp only [← c93180, ← c93181, k2, ← c93182] at e_1521_5
  have hr := e_1521_5
  simp only [e_1521_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1521 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14851)) := by
  have c93183 := (publicBatchWrapper86_copies2911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93184 := (publicBatchWrapper86_copies2912 a h).1
  have c93185 := (publicBatchWrapper86_copies2912 a h).2.1
  have c93186 := (publicBatchWrapper86_copies2912 a h).2.2.1
  have c93187 := (publicBatchWrapper86_copies2912 a h).2.2.2.1
  have c93188 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1521_6 := arithEq_of_rows h (row := 1521) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_6
  simp only [← c93183, ← c93184, ← c93185] at e_1521_6
  have e_1521_7 := arithEq_of_rows h (row := 1521) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_7
  simp only [← c93186, ← c93187, k2, ← c93188] at e_1521_7
  have hr := e_1521_7
  simp only [e_1521_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1521 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14852)) := by
  have c93189 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.1
  have c93190 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.1
  have c93191 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.1
  have c93192 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.1
  have c93193 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.1
  have c93194 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1521_8 := arithEq_of_rows h (row := 1521) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_8
  simp only [← c93189, ← c93190, ← c93191] at e_1521_8
  have e_1521_9 := arithEq_of_rows h (row := 1521) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_9
  simp only [← c93192, ← c93193, k2, ← c93194] at e_1521_9
  have hr := e_1521_9
  simp only [e_1521_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1521 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14853)) := by
  have c93195 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93196 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93197 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93198 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93199 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93200 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1521_10 := arithEq_of_rows h (row := 1521) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_10
  simp only [← c93195, ← c93196, ← c93197] at e_1521_10
  have e_1521_11 := arithEq_of_rows h (row := 1521) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_11
  simp only [← c93198, ← c93199, k2, ← c93200] at e_1521_11
  have hr := e_1521_11
  simp only [e_1521_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1521 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14854)) := by
  have c93201 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93202 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93203 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93204 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93205 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93206 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1521_12 := arithEq_of_rows h (row := 1521) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_12
  simp only [← c93201, ← c93202, ← c93203] at e_1521_12
  have e_1521_13 := arithEq_of_rows h (row := 1521) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_13
  simp only [← c93204, ← c93205, k2, ← c93206] at e_1521_13
  have hr := e_1521_13
  simp only [e_1521_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1521 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14855)) := by
  have c93207 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93208 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93209 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93210 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93211 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93212 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1521_14 := arithEq_of_rows h (row := 1521) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_14
  simp only [← c93207, ← c93208, ← c93209] at e_1521_14
  have e_1521_15 := arithEq_of_rows h (row := 1521) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_15
  simp only [← c93210, ← c93211, k2, ← c93212] at e_1521_15
  have hr := e_1521_15
  simp only [e_1521_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1521 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14856)) := by
  have c93213 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93214 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93215 := (publicBatchWrapper86_copies2912 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93216 := (publicBatchWrapper86_copies2913 a h).1
  have c93217 := (publicBatchWrapper86_copies2913 a h).2.1
  have c93218 := (publicBatchWrapper86_copies2913 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1521_16 := arithEq_of_rows h (row := 1521) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_16
  simp only [← c93213, ← c93214, ← c93215] at e_1521_16
  have e_1521_17 := arithEq_of_rows h (row := 1521) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_17
  simp only [← c93216, ← c93217, k2, ← c93218] at e_1521_17
  have hr := e_1521_17
  simp only [e_1521_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1521 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14857)) := by
  have c93219 := (publicBatchWrapper86_copies2913 a h).2.2.2.1
  have c93220 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.1
  have c93221 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.1
  have c93222 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.1
  have c93223 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.1
  have c93224 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1521_18 := arithEq_of_rows h (row := 1521) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_18
  simp only [← c93219, ← c93220, ← c93221] at e_1521_18
  have e_1521_19 := arithEq_of_rows h (row := 1521) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1521_19
  simp only [← c93222, ← c93223, k2, ← c93224] at e_1521_19
  have hr := e_1521_19
  simp only [e_1521_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1522 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14858)) := by
  have c93225 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.1
  have c93226 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93227 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93228 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93229 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93230 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1522_0 := arithEq_of_rows h (row := 1522) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_0
  simp only [← c93225, ← c93226, ← c93227] at e_1522_0
  have e_1522_1 := arithEq_of_rows h (row := 1522) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_1
  simp only [← c93228, ← c93229, k2, ← c93230] at e_1522_1
  have hr := e_1522_1
  simp only [e_1522_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1522 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14859)) := by
  have c93231 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93232 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93233 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93234 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93235 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93236 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1522_2 := arithEq_of_rows h (row := 1522) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_2
  simp only [← c93231, ← c93232, ← c93233] at e_1522_2
  have e_1522_3 := arithEq_of_rows h (row := 1522) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_3
  simp only [← c93234, ← c93235, k2, ← c93236] at e_1522_3
  have hr := e_1522_3
  simp only [e_1522_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1522 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14860)) := by
  have c93237 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93238 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93239 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93240 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93241 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93242 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1522_4 := arithEq_of_rows h (row := 1522) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_4
  simp only [← c93237, ← c93238, ← c93239] at e_1522_4
  have e_1522_5 := arithEq_of_rows h (row := 1522) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_5
  simp only [← c93240, ← c93241, k2, ← c93242] at e_1522_5
  have hr := e_1522_5
  simp only [e_1522_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1522 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14861)) := by
  have c93243 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93244 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93245 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93246 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93247 := (publicBatchWrapper86_copies2913 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93248 := (publicBatchWrapper86_copies2914 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1522_6 := arithEq_of_rows h (row := 1522) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_6
  simp only [← c93243, ← c93244, ← c93245] at e_1522_6
  have e_1522_7 := arithEq_of_rows h (row := 1522) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_7
  simp only [← c93246, ← c93247, k2, ← c93248] at e_1522_7
  have hr := e_1522_7
  simp only [e_1522_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1522 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14862)) := by
  have c93249 := (publicBatchWrapper86_copies2914 a h).2.1
  have c93250 := (publicBatchWrapper86_copies2914 a h).2.2.1
  have c93251 := (publicBatchWrapper86_copies2914 a h).2.2.2.1
  have c93252 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.1
  have c93253 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.1
  have c93254 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1522_8 := arithEq_of_rows h (row := 1522) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_8
  simp only [← c93249, ← c93250, ← c93251] at e_1522_8
  have e_1522_9 := arithEq_of_rows h (row := 1522) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_9
  simp only [← c93252, ← c93253, k2, ← c93254] at e_1522_9
  have hr := e_1522_9
  simp only [e_1522_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1522 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14863)) := by
  have c93255 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.1
  have c93256 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.1
  have c93257 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.1
  have c93258 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93259 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93260 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1522_10 := arithEq_of_rows h (row := 1522) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_10
  simp only [← c93255, ← c93256, ← c93257] at e_1522_10
  have e_1522_11 := arithEq_of_rows h (row := 1522) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_11
  simp only [← c93258, ← c93259, k2, ← c93260] at e_1522_11
  have hr := e_1522_11
  simp only [e_1522_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1522 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14864)) := by
  have c93261 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93262 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93263 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93264 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93265 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93266 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1522_12 := arithEq_of_rows h (row := 1522) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_12
  simp only [← c93261, ← c93262, ← c93263] at e_1522_12
  have e_1522_13 := arithEq_of_rows h (row := 1522) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_13
  simp only [← c93264, ← c93265, k2, ← c93266] at e_1522_13
  have hr := e_1522_13
  simp only [e_1522_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1522 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14865)) := by
  have c93267 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93268 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93269 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93270 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93271 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93272 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1522_14 := arithEq_of_rows h (row := 1522) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_14
  simp only [← c93267, ← c93268, ← c93269] at e_1522_14
  have e_1522_15 := arithEq_of_rows h (row := 1522) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_15
  simp only [← c93270, ← c93271, k2, ← c93272] at e_1522_15
  have hr := e_1522_15
  simp only [e_1522_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1522 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14866)) := by
  have c93273 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93274 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93275 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93276 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93277 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93278 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1522_16 := arithEq_of_rows h (row := 1522) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_16
  simp only [← c93273, ← c93274, ← c93275] at e_1522_16
  have e_1522_17 := arithEq_of_rows h (row := 1522) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_17
  simp only [← c93276, ← c93277, k2, ← c93278] at e_1522_17
  have hr := e_1522_17
  simp only [e_1522_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1522 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14867)) := by
  have c93279 := (publicBatchWrapper86_copies2914 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93280 := (publicBatchWrapper86_copies2915 a h).1
  have c93281 := (publicBatchWrapper86_copies2915 a h).2.1
  have c93282 := (publicBatchWrapper86_copies2915 a h).2.2.1
  have c93283 := (publicBatchWrapper86_copies2915 a h).2.2.2.1
  have c93284 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1522_18 := arithEq_of_rows h (row := 1522) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_18
  simp only [← c93279, ← c93280, ← c93281] at e_1522_18
  have e_1522_19 := arithEq_of_rows h (row := 1522) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1522_19
  simp only [← c93282, ← c93283, k2, ← c93284] at e_1522_19
  have hr := e_1522_19
  simp only [e_1522_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1523 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14868)) := by
  have c93285 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.1
  have c93286 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.1
  have c93287 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.1
  have c93288 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.1
  have c93289 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.1
  have c93290 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1523_0 := arithEq_of_rows h (row := 1523) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_0
  simp only [← c93285, ← c93286, ← c93287] at e_1523_0
  have e_1523_1 := arithEq_of_rows h (row := 1523) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_1
  simp only [← c93288, ← c93289, k2, ← c93290] at e_1523_1
  have hr := e_1523_1
  simp only [e_1523_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1523 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14869)) := by
  have c93291 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93292 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93293 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93294 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93295 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93296 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1523_2 := arithEq_of_rows h (row := 1523) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_2
  simp only [← c93291, ← c93292, ← c93293] at e_1523_2
  have e_1523_3 := arithEq_of_rows h (row := 1523) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_3
  simp only [← c93294, ← c93295, k2, ← c93296] at e_1523_3
  have hr := e_1523_3
  simp only [e_1523_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1523 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14870)) := by
  have c93297 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93298 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93299 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93300 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93301 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93302 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1523_4 := arithEq_of_rows h (row := 1523) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_4
  simp only [← c93297, ← c93298, ← c93299] at e_1523_4
  have e_1523_5 := arithEq_of_rows h (row := 1523) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_5
  simp only [← c93300, ← c93301, k2, ← c93302] at e_1523_5
  have hr := e_1523_5
  simp only [e_1523_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1523 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14871)) := by
  have c93303 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93304 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93305 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93306 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93307 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93308 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1523_6 := arithEq_of_rows h (row := 1523) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_6
  simp only [← c93303, ← c93304, ← c93305] at e_1523_6
  have e_1523_7 := arithEq_of_rows h (row := 1523) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_7
  simp only [← c93306, ← c93307, k2, ← c93308] at e_1523_7
  have hr := e_1523_7
  simp only [e_1523_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14826 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1523 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14872)) := by
  have c93309 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93310 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93311 := (publicBatchWrapper86_copies2915 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93312 := (publicBatchWrapper86_copies2916 a h).1
  have c93313 := (publicBatchWrapper86_copies2916 a h).2.1
  have c93314 := (publicBatchWrapper86_copies2916 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1523_8 := arithEq_of_rows h (row := 1523) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_8
  simp only [← c93309, ← c93310, ← c93311] at e_1523_8
  have e_1523_9 := arithEq_of_rows h (row := 1523) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_9
  simp only [← c93312, ← c93313, k2, ← c93314] at e_1523_9
  have hr := e_1523_9
  simp only [e_1523_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14827 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1523 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14873)) := by
  have c93315 := (publicBatchWrapper86_copies2916 a h).2.2.2.1
  have c93316 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.1
  have c93317 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.1
  have c93318 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.1
  have c93319 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.1
  have c93320 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1523_10 := arithEq_of_rows h (row := 1523) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_10
  simp only [← c93315, ← c93316, ← c93317] at e_1523_10
  have e_1523_11 := arithEq_of_rows h (row := 1523) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_11
  simp only [← c93318, ← c93319, k2, ← c93320] at e_1523_11
  have hr := e_1523_11
  simp only [e_1523_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14828 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1523 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14874)) := by
  have c93321 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.1
  have c93322 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93323 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93324 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93325 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93326 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1523_12 := arithEq_of_rows h (row := 1523) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_12
  simp only [← c93321, ← c93322, ← c93323] at e_1523_12
  have e_1523_13 := arithEq_of_rows h (row := 1523) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_13
  simp only [← c93324, ← c93325, k2, ← c93326] at e_1523_13
  have hr := e_1523_13
  simp only [e_1523_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14829 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1523 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14875)) := by
  have c93327 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93328 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93329 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93330 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93331 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93332 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1523_14 := arithEq_of_rows h (row := 1523) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_14
  simp only [← c93327, ← c93328, ← c93329] at e_1523_14
  have e_1523_15 := arithEq_of_rows h (row := 1523) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_15
  simp only [← c93330, ← c93331, k2, ← c93332] at e_1523_15
  have hr := e_1523_15
  simp only [e_1523_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14830 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1523 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14876)) := by
  have c93333 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93334 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93335 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93336 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93337 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93338 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1523_16 := arithEq_of_rows h (row := 1523) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_16
  simp only [← c93333, ← c93334, ← c93335] at e_1523_16
  have e_1523_17 := arithEq_of_rows h (row := 1523) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_17
  simp only [← c93336, ← c93337, k2, ← c93338] at e_1523_17
  have hr := e_1523_17
  simp only [e_1523_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14831 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1523 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14877)) := by
  have c93339 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93340 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93341 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93342 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93343 := (publicBatchWrapper86_copies2916 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93344 := (publicBatchWrapper86_copies2917 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1523_18 := arithEq_of_rows h (row := 1523) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_18
  simp only [← c93339, ← c93340, ← c93341] at e_1523_18
  have e_1523_19 := arithEq_of_rows h (row := 1523) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1523_19
  simp only [← c93342, ← c93343, k2, ← c93344] at e_1523_19
  have hr := e_1523_19
  simp only [e_1523_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14832 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1524 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14878)) := by
  have c93345 := (publicBatchWrapper86_copies2917 a h).2.1
  have c93346 := (publicBatchWrapper86_copies2917 a h).2.2.1
  have c93347 := (publicBatchWrapper86_copies2917 a h).2.2.2.1
  have c93348 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.1
  have c93349 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.1
  have c93350 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1524_0 := arithEq_of_rows h (row := 1524) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_0
  simp only [← c93345, ← c93346, ← c93347] at e_1524_0
  have e_1524_1 := arithEq_of_rows h (row := 1524) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_1
  simp only [← c93348, ← c93349, k2, ← c93350] at e_1524_1
  have hr := e_1524_1
  simp only [e_1524_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14833 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1524 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14879)) := by
  have c93351 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.1
  have c93352 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.1
  have c93353 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.1
  have c93354 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93355 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93356 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1524_2 := arithEq_of_rows h (row := 1524) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_2
  simp only [← c93351, ← c93352, ← c93353] at e_1524_2
  have e_1524_3 := arithEq_of_rows h (row := 1524) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_3
  simp only [← c93354, ← c93355, k2, ← c93356] at e_1524_3
  have hr := e_1524_3
  simp only [e_1524_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14834 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1524 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14880)) := by
  have c93357 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93358 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93359 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93360 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93361 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93362 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1524_4 := arithEq_of_rows h (row := 1524) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_4
  simp only [← c93357, ← c93358, ← c93359] at e_1524_4
  have e_1524_5 := arithEq_of_rows h (row := 1524) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_5
  simp only [← c93360, ← c93361, k2, ← c93362] at e_1524_5
  have hr := e_1524_5
  simp only [e_1524_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14835 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1524 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14881)) := by
  have c93363 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93364 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93365 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93366 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93367 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93368 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1524_6 := arithEq_of_rows h (row := 1524) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_6
  simp only [← c93363, ← c93364, ← c93365] at e_1524_6
  have e_1524_7 := arithEq_of_rows h (row := 1524) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_7
  simp only [← c93366, ← c93367, k2, ← c93368] at e_1524_7
  have hr := e_1524_7
  simp only [e_1524_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14836 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1524 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14882)) := by
  have c93369 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93370 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93371 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93372 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93373 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93374 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1524_8 := arithEq_of_rows h (row := 1524) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_8
  simp only [← c93369, ← c93370, ← c93371] at e_1524_8
  have e_1524_9 := arithEq_of_rows h (row := 1524) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_9
  simp only [← c93372, ← c93373, k2, ← c93374] at e_1524_9
  have hr := e_1524_9
  simp only [e_1524_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14837 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1524 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14883)) := by
  have c93375 := (publicBatchWrapper86_copies2917 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93376 := (publicBatchWrapper86_copies2918 a h).1
  have c93377 := (publicBatchWrapper86_copies2918 a h).2.1
  have c93378 := (publicBatchWrapper86_copies2918 a h).2.2.1
  have c93379 := (publicBatchWrapper86_copies2918 a h).2.2.2.1
  have c93380 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1524_10 := arithEq_of_rows h (row := 1524) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_10
  simp only [← c93375, ← c93376, ← c93377] at e_1524_10
  have e_1524_11 := arithEq_of_rows h (row := 1524) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_11
  simp only [← c93378, ← c93379, k2, ← c93380] at e_1524_11
  have hr := e_1524_11
  simp only [e_1524_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14838 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1524 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14884)) := by
  have c93381 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.1
  have c93382 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.1
  have c93383 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.1
  have c93384 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.1
  have c93385 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.1
  have c93386 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1524_12 := arithEq_of_rows h (row := 1524) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_12
  simp only [← c93381, ← c93382, ← c93383] at e_1524_12
  have e_1524_13 := arithEq_of_rows h (row := 1524) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_13
  simp only [← c93384, ← c93385, k2, ← c93386] at e_1524_13
  have hr := e_1524_13
  simp only [e_1524_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14839 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1524 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14885)) := by
  have c93387 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93388 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93389 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93390 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93391 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93392 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1524_14 := arithEq_of_rows h (row := 1524) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_14
  simp only [← c93387, ← c93388, ← c93389] at e_1524_14
  have e_1524_15 := arithEq_of_rows h (row := 1524) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_15
  simp only [← c93390, ← c93391, k2, ← c93392] at e_1524_15
  have hr := e_1524_15
  simp only [e_1524_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14840 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1524 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14886)) := by
  have c93393 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93394 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93395 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93396 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93397 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93398 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1524_16 := arithEq_of_rows h (row := 1524) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_16
  simp only [← c93393, ← c93394, ← c93395] at e_1524_16
  have e_1524_17 := arithEq_of_rows h (row := 1524) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_17
  simp only [← c93396, ← c93397, k2, ← c93398] at e_1524_17
  have hr := e_1524_17
  simp only [e_1524_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14841 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1524 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14887)) := by
  have c93399 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93400 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93401 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93402 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93403 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93404 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1524_18 := arithEq_of_rows h (row := 1524) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_18
  simp only [← c93399, ← c93400, ← c93401] at e_1524_18
  have e_1524_19 := arithEq_of_rows h (row := 1524) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1524_19
  simp only [← c93402, ← c93403, k2, ← c93404] at e_1524_19
  have hr := e_1524_19
  simp only [e_1524_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14842 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1525 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14888)) := by
  have c93405 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93406 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93407 := (publicBatchWrapper86_copies2918 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93408 := (publicBatchWrapper86_copies2919 a h).1
  have c93409 := (publicBatchWrapper86_copies2919 a h).2.1
  have c93410 := (publicBatchWrapper86_copies2919 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1525_0 := arithEq_of_rows h (row := 1525) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_0
  simp only [← c93405, ← c93406, ← c93407] at e_1525_0
  have e_1525_1 := arithEq_of_rows h (row := 1525) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_1
  simp only [← c93408, ← c93409, k2, ← c93410] at e_1525_1
  have hr := e_1525_1
  simp only [e_1525_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14843 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1525 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14889)) := by
  have c93411 := (publicBatchWrapper86_copies2919 a h).2.2.2.1
  have c93412 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.1
  have c93413 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.1
  have c93414 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.1
  have c93415 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.1
  have c93416 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1525_2 := arithEq_of_rows h (row := 1525) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_2
  simp only [← c93411, ← c93412, ← c93413] at e_1525_2
  have e_1525_3 := arithEq_of_rows h (row := 1525) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_3
  simp only [← c93414, ← c93415, k2, ← c93416] at e_1525_3
  have hr := e_1525_3
  simp only [e_1525_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14844 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1525 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14890)) := by
  have c93417 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.1
  have c93418 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93419 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93420 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93421 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93422 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1525_4 := arithEq_of_rows h (row := 1525) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_4
  simp only [← c93417, ← c93418, ← c93419] at e_1525_4
  have e_1525_5 := arithEq_of_rows h (row := 1525) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_5
  simp only [← c93420, ← c93421, k2, ← c93422] at e_1525_5
  have hr := e_1525_5
  simp only [e_1525_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14845 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1525 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14891)) := by
  have c93423 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93424 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93425 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93426 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93427 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93428 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1525_6 := arithEq_of_rows h (row := 1525) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_6
  simp only [← c93423, ← c93424, ← c93425] at e_1525_6
  have e_1525_7 := arithEq_of_rows h (row := 1525) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_7
  simp only [← c93426, ← c93427, k2, ← c93428] at e_1525_7
  have hr := e_1525_7
  simp only [e_1525_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14846 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1525 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14892)) := by
  have c93429 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93430 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93431 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93432 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93433 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93434 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1525_8 := arithEq_of_rows h (row := 1525) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_8
  simp only [← c93429, ← c93430, ← c93431] at e_1525_8
  have e_1525_9 := arithEq_of_rows h (row := 1525) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_9
  simp only [← c93432, ← c93433, k2, ← c93434] at e_1525_9
  have hr := e_1525_9
  simp only [e_1525_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14847 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1525 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14893)) := by
  have c93435 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93436 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93437 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93438 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93439 := (publicBatchWrapper86_copies2919 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93440 := (publicBatchWrapper86_copies2920 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1525_10 := arithEq_of_rows h (row := 1525) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_10
  simp only [← c93435, ← c93436, ← c93437] at e_1525_10
  have e_1525_11 := arithEq_of_rows h (row := 1525) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_11
  simp only [← c93438, ← c93439, k2, ← c93440] at e_1525_11
  have hr := e_1525_11
  simp only [e_1525_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14848 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1525 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14894)) := by
  have c93441 := (publicBatchWrapper86_copies2920 a h).2.1
  have c93442 := (publicBatchWrapper86_copies2920 a h).2.2.1
  have c93443 := (publicBatchWrapper86_copies2920 a h).2.2.2.1
  have c93444 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.1
  have c93445 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.1
  have c93446 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1525_12 := arithEq_of_rows h (row := 1525) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_12
  simp only [← c93441, ← c93442, ← c93443] at e_1525_12
  have e_1525_13 := arithEq_of_rows h (row := 1525) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_13
  simp only [← c93444, ← c93445, k2, ← c93446] at e_1525_13
  have hr := e_1525_13
  simp only [e_1525_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14849 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1525 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14895)) := by
  have c93447 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.1
  have c93448 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.1
  have c93449 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.1
  have c93450 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93451 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93452 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1525_14 := arithEq_of_rows h (row := 1525) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_14
  simp only [← c93447, ← c93448, ← c93449] at e_1525_14
  have e_1525_15 := arithEq_of_rows h (row := 1525) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_15
  simp only [← c93450, ← c93451, k2, ← c93452] at e_1525_15
  have hr := e_1525_15
  simp only [e_1525_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14850 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1525 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14896)) := by
  have c93453 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93454 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93455 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93456 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93457 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93458 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1525_16 := arithEq_of_rows h (row := 1525) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_16
  simp only [← c93453, ← c93454, ← c93455] at e_1525_16
  have e_1525_17 := arithEq_of_rows h (row := 1525) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_17
  simp only [← c93456, ← c93457, k2, ← c93458] at e_1525_17
  have hr := e_1525_17
  simp only [e_1525_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14851 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1525 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14897)) := by
  have c93459 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93460 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93461 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93462 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93463 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93464 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1525_18 := arithEq_of_rows h (row := 1525) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_18
  simp only [← c93459, ← c93460, ← c93461] at e_1525_18
  have e_1525_19 := arithEq_of_rows h (row := 1525) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1525_19
  simp only [← c93462, ← c93463, k2, ← c93464] at e_1525_19
  have hr := e_1525_19
  simp only [e_1525_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14852 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1526 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14898)) := by
  have c93465 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93466 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93467 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93468 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93469 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93470 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1526_0 := arithEq_of_rows h (row := 1526) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_0
  simp only [← c93465, ← c93466, ← c93467] at e_1526_0
  have e_1526_1 := arithEq_of_rows h (row := 1526) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_1
  simp only [← c93468, ← c93469, k2, ← c93470] at e_1526_1
  have hr := e_1526_1
  simp only [e_1526_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14853 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1526 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14899)) := by
  have c93471 := (publicBatchWrapper86_copies2920 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93472 := (publicBatchWrapper86_copies2921 a h).1
  have c93473 := (publicBatchWrapper86_copies2921 a h).2.1
  have c93474 := (publicBatchWrapper86_copies2921 a h).2.2.1
  have c93475 := (publicBatchWrapper86_copies2921 a h).2.2.2.1
  have c93476 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1526_2 := arithEq_of_rows h (row := 1526) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_2
  simp only [← c93471, ← c93472, ← c93473] at e_1526_2
  have e_1526_3 := arithEq_of_rows h (row := 1526) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_3
  simp only [← c93474, ← c93475, k2, ← c93476] at e_1526_3
  have hr := e_1526_3
  simp only [e_1526_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14854 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1526 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14900)) := by
  have c93477 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.1
  have c93478 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.1
  have c93479 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.1
  have c93480 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.1
  have c93481 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.1
  have c93482 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1526_4 := arithEq_of_rows h (row := 1526) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_4
  simp only [← c93477, ← c93478, ← c93479] at e_1526_4
  have e_1526_5 := arithEq_of_rows h (row := 1526) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_5
  simp only [← c93480, ← c93481, k2, ← c93482] at e_1526_5
  have hr := e_1526_5
  simp only [e_1526_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14855 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1526 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14901)) := by
  have c93483 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93484 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93485 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93486 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93487 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93488 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1526_6 := arithEq_of_rows h (row := 1526) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_6
  simp only [← c93483, ← c93484, ← c93485] at e_1526_6
  have e_1526_7 := arithEq_of_rows h (row := 1526) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_7
  simp only [← c93486, ← c93487, k2, ← c93488] at e_1526_7
  have hr := e_1526_7
  simp only [e_1526_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14856 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1526 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14902)) := by
  have c93489 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93490 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93491 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93492 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93493 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93494 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1526_8 := arithEq_of_rows h (row := 1526) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_8
  simp only [← c93489, ← c93490, ← c93491] at e_1526_8
  have e_1526_9 := arithEq_of_rows h (row := 1526) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_9
  simp only [← c93492, ← c93493, k2, ← c93494] at e_1526_9
  have hr := e_1526_9
  simp only [e_1526_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14857 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1526 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14903)) := by
  have c93495 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93496 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93497 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93498 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93499 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93500 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1526_10 := arithEq_of_rows h (row := 1526) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_10
  simp only [← c93495, ← c93496, ← c93497] at e_1526_10
  have e_1526_11 := arithEq_of_rows h (row := 1526) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_11
  simp only [← c93498, ← c93499, k2, ← c93500] at e_1526_11
  have hr := e_1526_11
  simp only [e_1526_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14858 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1526 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14904)) := by
  have c93501 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93502 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93503 := (publicBatchWrapper86_copies2921 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93504 := (publicBatchWrapper86_copies2922 a h).1
  have c93505 := (publicBatchWrapper86_copies2922 a h).2.1
  have c93506 := (publicBatchWrapper86_copies2922 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1526_12 := arithEq_of_rows h (row := 1526) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_12
  simp only [← c93501, ← c93502, ← c93503] at e_1526_12
  have e_1526_13 := arithEq_of_rows h (row := 1526) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_13
  simp only [← c93504, ← c93505, k2, ← c93506] at e_1526_13
  have hr := e_1526_13
  simp only [e_1526_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14859 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1526 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14905)) := by
  have c93507 := (publicBatchWrapper86_copies2922 a h).2.2.2.1
  have c93508 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.1
  have c93509 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.1
  have c93510 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.1
  have c93511 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.1
  have c93512 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1526_14 := arithEq_of_rows h (row := 1526) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_14
  simp only [← c93507, ← c93508, ← c93509] at e_1526_14
  have e_1526_15 := arithEq_of_rows h (row := 1526) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_15
  simp only [← c93510, ← c93511, k2, ← c93512] at e_1526_15
  have hr := e_1526_15
  simp only [e_1526_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14860 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1526 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14906)) := by
  have c93513 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.1
  have c93514 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93515 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93516 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93517 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93518 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1526_16 := arithEq_of_rows h (row := 1526) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_16
  simp only [← c93513, ← c93514, ← c93515] at e_1526_16
  have e_1526_17 := arithEq_of_rows h (row := 1526) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_17
  simp only [← c93516, ← c93517, k2, ← c93518] at e_1526_17
  have hr := e_1526_17
  simp only [e_1526_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14861 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1526 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14907)) := by
  have c93519 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93520 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93521 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93522 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93523 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93524 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1526_18 := arithEq_of_rows h (row := 1526) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_18
  simp only [← c93519, ← c93520, ← c93521] at e_1526_18
  have e_1526_19 := arithEq_of_rows h (row := 1526) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1526_19
  simp only [← c93522, ← c93523, k2, ← c93524] at e_1526_19
  have hr := e_1526_19
  simp only [e_1526_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14862 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1527 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14908)) := by
  have c93525 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93526 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93527 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93528 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93529 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93530 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1527_0 := arithEq_of_rows h (row := 1527) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_0
  simp only [← c93525, ← c93526, ← c93527] at e_1527_0
  have e_1527_1 := arithEq_of_rows h (row := 1527) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_1
  simp only [← c93528, ← c93529, k2, ← c93530] at e_1527_1
  have hr := e_1527_1
  simp only [e_1527_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14863 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1527 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14909)) := by
  have c93531 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93532 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93533 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93534 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93535 := (publicBatchWrapper86_copies2922 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93536 := (publicBatchWrapper86_copies2923 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1527_2 := arithEq_of_rows h (row := 1527) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_2
  simp only [← c93531, ← c93532, ← c93533] at e_1527_2
  have e_1527_3 := arithEq_of_rows h (row := 1527) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_3
  simp only [← c93534, ← c93535, k2, ← c93536] at e_1527_3
  have hr := e_1527_3
  simp only [e_1527_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14864 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1527 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14910)) := by
  have c93537 := (publicBatchWrapper86_copies2923 a h).2.1
  have c93538 := (publicBatchWrapper86_copies2923 a h).2.2.1
  have c93539 := (publicBatchWrapper86_copies2923 a h).2.2.2.1
  have c93540 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.1
  have c93541 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.1
  have c93542 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1527_4 := arithEq_of_rows h (row := 1527) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_4
  simp only [← c93537, ← c93538, ← c93539] at e_1527_4
  have e_1527_5 := arithEq_of_rows h (row := 1527) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_5
  simp only [← c93540, ← c93541, k2, ← c93542] at e_1527_5
  have hr := e_1527_5
  simp only [e_1527_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14865 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1527 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14911)) := by
  have c93543 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.1
  have c93544 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.1
  have c93545 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.1
  have c93546 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93547 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93548 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1527_6 := arithEq_of_rows h (row := 1527) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_6
  simp only [← c93543, ← c93544, ← c93545] at e_1527_6
  have e_1527_7 := arithEq_of_rows h (row := 1527) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_7
  simp only [← c93546, ← c93547, k2, ← c93548] at e_1527_7
  have hr := e_1527_7
  simp only [e_1527_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14866 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1527 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14912)) := by
  have c93549 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93550 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93551 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93552 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93553 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93554 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1527_8 := arithEq_of_rows h (row := 1527) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_8
  simp only [← c93549, ← c93550, ← c93551] at e_1527_8
  have e_1527_9 := arithEq_of_rows h (row := 1527) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_9
  simp only [← c93552, ← c93553, k2, ← c93554] at e_1527_9
  have hr := e_1527_9
  simp only [e_1527_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14867 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1527 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14913)) := by
  have c93555 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93556 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93557 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93558 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93559 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93560 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1527_10 := arithEq_of_rows h (row := 1527) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_10
  simp only [← c93555, ← c93556, ← c93557] at e_1527_10
  have e_1527_11 := arithEq_of_rows h (row := 1527) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_11
  simp only [← c93558, ← c93559, k2, ← c93560] at e_1527_11
  have hr := e_1527_11
  simp only [e_1527_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14868 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1527 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14914)) := by
  have c93561 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93562 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93563 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93564 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93565 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93566 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1527_12 := arithEq_of_rows h (row := 1527) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_12
  simp only [← c93561, ← c93562, ← c93563] at e_1527_12
  have e_1527_13 := arithEq_of_rows h (row := 1527) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_13
  simp only [← c93564, ← c93565, k2, ← c93566] at e_1527_13
  have hr := e_1527_13
  simp only [e_1527_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14869 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1527 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14915)) := by
  have c93567 := (publicBatchWrapper86_copies2923 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93568 := (publicBatchWrapper86_copies2924 a h).1
  have c93569 := (publicBatchWrapper86_copies2924 a h).2.1
  have c93570 := (publicBatchWrapper86_copies2924 a h).2.2.1
  have c93571 := (publicBatchWrapper86_copies2924 a h).2.2.2.1
  have c93572 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1527_14 := arithEq_of_rows h (row := 1527) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_14
  simp only [← c93567, ← c93568, ← c93569] at e_1527_14
  have e_1527_15 := arithEq_of_rows h (row := 1527) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_15
  simp only [← c93570, ← c93571, k2, ← c93572] at e_1527_15
  have hr := e_1527_15
  simp only [e_1527_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1527 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14916)) := by
  have c93573 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.1
  have c93574 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.1
  have c93575 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.1
  have c93576 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.1
  have c93577 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.1
  have c93578 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1527_16 := arithEq_of_rows h (row := 1527) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_16
  simp only [← c93573, ← c93574, ← c93575] at e_1527_16
  have e_1527_17 := arithEq_of_rows h (row := 1527) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_17
  simp only [← c93576, ← c93577, k2, ← c93578] at e_1527_17
  have hr := e_1527_17
  simp only [e_1527_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1527 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14917)) := by
  have c93579 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93580 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93581 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93582 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93583 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93584 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1527_18 := arithEq_of_rows h (row := 1527) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_18
  simp only [← c93579, ← c93580, ← c93581] at e_1527_18
  have e_1527_19 := arithEq_of_rows h (row := 1527) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1527_19
  simp only [← c93582, ← c93583, k2, ← c93584] at e_1527_19
  have hr := e_1527_19
  simp only [e_1527_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1528 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14918)) := by
  have c93585 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93586 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93587 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93588 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93589 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93590 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1528_0 := arithEq_of_rows h (row := 1528) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_0
  simp only [← c93585, ← c93586, ← c93587] at e_1528_0
  have e_1528_1 := arithEq_of_rows h (row := 1528) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_1
  simp only [← c93588, ← c93589, k2, ← c93590] at e_1528_1
  have hr := e_1528_1
  simp only [e_1528_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1528 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14919)) := by
  have c93591 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93592 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93593 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93594 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93595 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93596 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1528_2 := arithEq_of_rows h (row := 1528) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_2
  simp only [← c93591, ← c93592, ← c93593] at e_1528_2
  have e_1528_3 := arithEq_of_rows h (row := 1528) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_3
  simp only [← c93594, ← c93595, k2, ← c93596] at e_1528_3
  have hr := e_1528_3
  simp only [e_1528_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1528 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14920)) := by
  have c93597 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93598 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93599 := (publicBatchWrapper86_copies2924 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93600 := (publicBatchWrapper86_copies2925 a h).1
  have c93601 := (publicBatchWrapper86_copies2925 a h).2.1
  have c93602 := (publicBatchWrapper86_copies2925 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1528_4 := arithEq_of_rows h (row := 1528) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_4
  simp only [← c93597, ← c93598, ← c93599] at e_1528_4
  have e_1528_5 := arithEq_of_rows h (row := 1528) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_5
  simp only [← c93600, ← c93601, k2, ← c93602] at e_1528_5
  have hr := e_1528_5
  simp only [e_1528_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1528 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14921)) := by
  have c93603 := (publicBatchWrapper86_copies2925 a h).2.2.2.1
  have c93604 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.1
  have c93605 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.1
  have c93606 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.1
  have c93607 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.1
  have c93608 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1528_6 := arithEq_of_rows h (row := 1528) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_6
  simp only [← c93603, ← c93604, ← c93605] at e_1528_6
  have e_1528_7 := arithEq_of_rows h (row := 1528) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_7
  simp only [← c93606, ← c93607, k2, ← c93608] at e_1528_7
  have hr := e_1528_7
  simp only [e_1528_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1528 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14922)) := by
  have c93609 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.1
  have c93610 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93611 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93612 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93613 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93614 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1528_8 := arithEq_of_rows h (row := 1528) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_8
  simp only [← c93609, ← c93610, ← c93611] at e_1528_8
  have e_1528_9 := arithEq_of_rows h (row := 1528) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_9
  simp only [← c93612, ← c93613, k2, ← c93614] at e_1528_9
  have hr := e_1528_9
  simp only [e_1528_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14877 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1528 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14923)) := by
  have c93615 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93616 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93617 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93618 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93619 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93620 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1528_10 := arithEq_of_rows h (row := 1528) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_10
  simp only [← c93615, ← c93616, ← c93617] at e_1528_10
  have e_1528_11 := arithEq_of_rows h (row := 1528) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_11
  simp only [← c93618, ← c93619, k2, ← c93620] at e_1528_11
  have hr := e_1528_11
  simp only [e_1528_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14878 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1528 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14924)) := by
  have c93621 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93622 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93623 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93624 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93625 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93626 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1528_12 := arithEq_of_rows h (row := 1528) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_12
  simp only [← c93621, ← c93622, ← c93623] at e_1528_12
  have e_1528_13 := arithEq_of_rows h (row := 1528) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_13
  simp only [← c93624, ← c93625, k2, ← c93626] at e_1528_13
  have hr := e_1528_13
  simp only [e_1528_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14879 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1528 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14925)) := by
  have c93627 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93628 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93629 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93630 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93631 := (publicBatchWrapper86_copies2925 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93632 := (publicBatchWrapper86_copies2926 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1528_14 := arithEq_of_rows h (row := 1528) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_14
  simp only [← c93627, ← c93628, ← c93629] at e_1528_14
  have e_1528_15 := arithEq_of_rows h (row := 1528) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_15
  simp only [← c93630, ← c93631, k2, ← c93632] at e_1528_15
  have hr := e_1528_15
  simp only [e_1528_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14880 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1528 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14926)) := by
  have c93633 := (publicBatchWrapper86_copies2926 a h).2.1
  have c93634 := (publicBatchWrapper86_copies2926 a h).2.2.1
  have c93635 := (publicBatchWrapper86_copies2926 a h).2.2.2.1
  have c93636 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.1
  have c93637 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.1
  have c93638 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1528_16 := arithEq_of_rows h (row := 1528) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_16
  simp only [← c93633, ← c93634, ← c93635] at e_1528_16
  have e_1528_17 := arithEq_of_rows h (row := 1528) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_17
  simp only [← c93636, ← c93637, k2, ← c93638] at e_1528_17
  have hr := e_1528_17
  simp only [e_1528_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14881 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1528 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14927)) := by
  have c93639 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.1
  have c93640 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.1
  have c93641 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.1
  have c93642 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93643 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93644 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1528_18 := arithEq_of_rows h (row := 1528) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_18
  simp only [← c93639, ← c93640, ← c93641] at e_1528_18
  have e_1528_19 := arithEq_of_rows h (row := 1528) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1528_19
  simp only [← c93642, ← c93643, k2, ← c93644] at e_1528_19
  have hr := e_1528_19
  simp only [e_1528_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14882 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1529 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14928)) := by
  have c93645 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93646 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93647 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93648 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93649 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93650 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1529_0 := arithEq_of_rows h (row := 1529) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_0
  simp only [← c93645, ← c93646, ← c93647] at e_1529_0
  have e_1529_1 := arithEq_of_rows h (row := 1529) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_1
  simp only [← c93648, ← c93649, k2, ← c93650] at e_1529_1
  have hr := e_1529_1
  simp only [e_1529_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14883 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1529 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14929)) := by
  have c93651 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93652 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93653 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93654 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93655 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93656 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1529_2 := arithEq_of_rows h (row := 1529) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_2
  simp only [← c93651, ← c93652, ← c93653] at e_1529_2
  have e_1529_3 := arithEq_of_rows h (row := 1529) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_3
  simp only [← c93654, ← c93655, k2, ← c93656] at e_1529_3
  have hr := e_1529_3
  simp only [e_1529_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14884 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1529 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14930)) := by
  have c93657 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93658 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93659 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93660 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93661 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93662 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1529_4 := arithEq_of_rows h (row := 1529) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_4
  simp only [← c93657, ← c93658, ← c93659] at e_1529_4
  have e_1529_5 := arithEq_of_rows h (row := 1529) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_5
  simp only [← c93660, ← c93661, k2, ← c93662] at e_1529_5
  have hr := e_1529_5
  simp only [e_1529_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14885 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1529 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14931)) := by
  have c93663 := (publicBatchWrapper86_copies2926 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93664 := (publicBatchWrapper86_copies2927 a h).1
  have c93665 := (publicBatchWrapper86_copies2927 a h).2.1
  have c93666 := (publicBatchWrapper86_copies2927 a h).2.2.1
  have c93667 := (publicBatchWrapper86_copies2927 a h).2.2.2.1
  have c93668 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1529_6 := arithEq_of_rows h (row := 1529) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_6
  simp only [← c93663, ← c93664, ← c93665] at e_1529_6
  have e_1529_7 := arithEq_of_rows h (row := 1529) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_7
  simp only [← c93666, ← c93667, k2, ← c93668] at e_1529_7
  have hr := e_1529_7
  simp only [e_1529_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14886 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1529 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14932)) := by
  have c93669 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.1
  have c93670 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.1
  have c93671 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.1
  have c93672 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.1
  have c93673 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.1
  have c93674 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1529_8 := arithEq_of_rows h (row := 1529) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_8
  simp only [← c93669, ← c93670, ← c93671] at e_1529_8
  have e_1529_9 := arithEq_of_rows h (row := 1529) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_9
  simp only [← c93672, ← c93673, k2, ← c93674] at e_1529_9
  have hr := e_1529_9
  simp only [e_1529_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14887 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1529 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14933)) := by
  have c93675 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93676 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93677 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93678 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93679 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93680 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1529_10 := arithEq_of_rows h (row := 1529) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_10
  simp only [← c93675, ← c93676, ← c93677] at e_1529_10
  have e_1529_11 := arithEq_of_rows h (row := 1529) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_11
  simp only [← c93678, ← c93679, k2, ← c93680] at e_1529_11
  have hr := e_1529_11
  simp only [e_1529_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14888 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1529 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14934)) := by
  have c93681 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93682 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93683 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93684 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93685 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93686 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1529_12 := arithEq_of_rows h (row := 1529) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_12
  simp only [← c93681, ← c93682, ← c93683] at e_1529_12
  have e_1529_13 := arithEq_of_rows h (row := 1529) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_13
  simp only [← c93684, ← c93685, k2, ← c93686] at e_1529_13
  have hr := e_1529_13
  simp only [e_1529_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14889 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1529 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14935)) := by
  have c93687 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93688 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93689 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93690 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93691 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93692 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1529_14 := arithEq_of_rows h (row := 1529) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_14
  simp only [← c93687, ← c93688, ← c93689] at e_1529_14
  have e_1529_15 := arithEq_of_rows h (row := 1529) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_15
  simp only [← c93690, ← c93691, k2, ← c93692] at e_1529_15
  have hr := e_1529_15
  simp only [e_1529_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14890 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1529 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14936)) := by
  have c93693 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93694 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93695 := (publicBatchWrapper86_copies2927 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93696 := (publicBatchWrapper86_copies2928 a h).1
  have c93697 := (publicBatchWrapper86_copies2928 a h).2.1
  have c93698 := (publicBatchWrapper86_copies2928 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1529_16 := arithEq_of_rows h (row := 1529) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_16
  simp only [← c93693, ← c93694, ← c93695] at e_1529_16
  have e_1529_17 := arithEq_of_rows h (row := 1529) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_17
  simp only [← c93696, ← c93697, k2, ← c93698] at e_1529_17
  have hr := e_1529_17
  simp only [e_1529_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14891 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1529 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14937)) := by
  have c93699 := (publicBatchWrapper86_copies2928 a h).2.2.2.1
  have c93700 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.1
  have c93701 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.1
  have c93702 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.1
  have c93703 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.1
  have c93704 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1529_18 := arithEq_of_rows h (row := 1529) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_18
  simp only [← c93699, ← c93700, ← c93701] at e_1529_18
  have e_1529_19 := arithEq_of_rows h (row := 1529) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1529_19
  simp only [← c93702, ← c93703, k2, ← c93704] at e_1529_19
  have hr := e_1529_19
  simp only [e_1529_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14892 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1530 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14938)) := by
  have c93705 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.1
  have c93706 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93707 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93708 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93709 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93710 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1530_0 := arithEq_of_rows h (row := 1530) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_0
  simp only [← c93705, ← c93706, ← c93707] at e_1530_0
  have e_1530_1 := arithEq_of_rows h (row := 1530) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_1
  simp only [← c93708, ← c93709, k2, ← c93710] at e_1530_1
  have hr := e_1530_1
  simp only [e_1530_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14893 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1530 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14939)) := by
  have c93711 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93712 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93713 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93714 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93715 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93716 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1530_2 := arithEq_of_rows h (row := 1530) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_2
  simp only [← c93711, ← c93712, ← c93713] at e_1530_2
  have e_1530_3 := arithEq_of_rows h (row := 1530) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_3
  simp only [← c93714, ← c93715, k2, ← c93716] at e_1530_3
  have hr := e_1530_3
  simp only [e_1530_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14894 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1530 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14940)) := by
  have c93717 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93718 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93719 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93720 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93721 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93722 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1530_4 := arithEq_of_rows h (row := 1530) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_4
  simp only [← c93717, ← c93718, ← c93719] at e_1530_4
  have e_1530_5 := arithEq_of_rows h (row := 1530) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_5
  simp only [← c93720, ← c93721, k2, ← c93722] at e_1530_5
  have hr := e_1530_5
  simp only [e_1530_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14895 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1530 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14941)) := by
  have c93723 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93724 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93725 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93726 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93727 := (publicBatchWrapper86_copies2928 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93728 := (publicBatchWrapper86_copies2929 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1530_6 := arithEq_of_rows h (row := 1530) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_6
  simp only [← c93723, ← c93724, ← c93725] at e_1530_6
  have e_1530_7 := arithEq_of_rows h (row := 1530) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_7
  simp only [← c93726, ← c93727, k2, ← c93728] at e_1530_7
  have hr := e_1530_7
  simp only [e_1530_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14896 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1530 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14942)) := by
  have c93729 := (publicBatchWrapper86_copies2929 a h).2.1
  have c93730 := (publicBatchWrapper86_copies2929 a h).2.2.1
  have c93731 := (publicBatchWrapper86_copies2929 a h).2.2.2.1
  have c93732 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.1
  have c93733 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.1
  have c93734 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1530_8 := arithEq_of_rows h (row := 1530) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_8
  simp only [← c93729, ← c93730, ← c93731] at e_1530_8
  have e_1530_9 := arithEq_of_rows h (row := 1530) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_9
  simp only [← c93732, ← c93733, k2, ← c93734] at e_1530_9
  have hr := e_1530_9
  simp only [e_1530_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14897 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1530 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14943)) := by
  have c93735 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.1
  have c93736 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.1
  have c93737 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.1
  have c93738 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93739 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93740 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1530_10 := arithEq_of_rows h (row := 1530) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_10
  simp only [← c93735, ← c93736, ← c93737] at e_1530_10
  have e_1530_11 := arithEq_of_rows h (row := 1530) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_11
  simp only [← c93738, ← c93739, k2, ← c93740] at e_1530_11
  have hr := e_1530_11
  simp only [e_1530_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14898 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1530 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14944)) := by
  have c93741 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93742 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93743 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93744 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93745 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93746 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1530_12 := arithEq_of_rows h (row := 1530) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_12
  simp only [← c93741, ← c93742, ← c93743] at e_1530_12
  have e_1530_13 := arithEq_of_rows h (row := 1530) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_13
  simp only [← c93744, ← c93745, k2, ← c93746] at e_1530_13
  have hr := e_1530_13
  simp only [e_1530_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14899 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1530 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14945)) := by
  have c93747 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93748 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93749 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93750 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93751 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93752 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1530_14 := arithEq_of_rows h (row := 1530) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_14
  simp only [← c93747, ← c93748, ← c93749] at e_1530_14
  have e_1530_15 := arithEq_of_rows h (row := 1530) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_15
  simp only [← c93750, ← c93751, k2, ← c93752] at e_1530_15
  have hr := e_1530_15
  simp only [e_1530_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14900 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1530 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14946)) := by
  have c93753 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93754 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93755 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93756 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93757 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93758 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1530_16 := arithEq_of_rows h (row := 1530) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_16
  simp only [← c93753, ← c93754, ← c93755] at e_1530_16
  have e_1530_17 := arithEq_of_rows h (row := 1530) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_17
  simp only [← c93756, ← c93757, k2, ← c93758] at e_1530_17
  have hr := e_1530_17
  simp only [e_1530_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14901 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1530 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14947)) := by
  have c93759 := (publicBatchWrapper86_copies2929 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93760 := (publicBatchWrapper86_copies2930 a h).1
  have c93761 := (publicBatchWrapper86_copies2930 a h).2.1
  have c93762 := (publicBatchWrapper86_copies2930 a h).2.2.1
  have c93763 := (publicBatchWrapper86_copies2930 a h).2.2.2.1
  have c93764 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1530_18 := arithEq_of_rows h (row := 1530) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_18
  simp only [← c93759, ← c93760, ← c93761] at e_1530_18
  have e_1530_19 := arithEq_of_rows h (row := 1530) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1530_19
  simp only [← c93762, ← c93763, k2, ← c93764] at e_1530_19
  have hr := e_1530_19
  simp only [e_1530_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14902 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1531 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14948)) := by
  have c93765 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.1
  have c93766 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.1
  have c93767 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.1
  have c93768 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.1
  have c93769 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.1
  have c93770 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1531_0 := arithEq_of_rows h (row := 1531) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_0
  simp only [← c93765, ← c93766, ← c93767] at e_1531_0
  have e_1531_1 := arithEq_of_rows h (row := 1531) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_1
  simp only [← c93768, ← c93769, k2, ← c93770] at e_1531_1
  have hr := e_1531_1
  simp only [e_1531_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14903 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1531 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14949)) := by
  have c93771 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93772 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93773 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93774 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93775 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93776 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1531_2 := arithEq_of_rows h (row := 1531) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_2
  simp only [← c93771, ← c93772, ← c93773] at e_1531_2
  have e_1531_3 := arithEq_of_rows h (row := 1531) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_3
  simp only [← c93774, ← c93775, k2, ← c93776] at e_1531_3
  have hr := e_1531_3
  simp only [e_1531_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14904 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1531 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14950)) := by
  have c93777 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93778 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93779 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93780 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93781 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93782 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1531_4 := arithEq_of_rows h (row := 1531) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_4
  simp only [← c93777, ← c93778, ← c93779] at e_1531_4
  have e_1531_5 := arithEq_of_rows h (row := 1531) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_5
  simp only [← c93780, ← c93781, k2, ← c93782] at e_1531_5
  have hr := e_1531_5
  simp only [e_1531_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14905 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1531 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14951)) := by
  have c93783 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93784 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93785 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93786 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93787 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93788 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1531_6 := arithEq_of_rows h (row := 1531) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_6
  simp only [← c93783, ← c93784, ← c93785] at e_1531_6
  have e_1531_7 := arithEq_of_rows h (row := 1531) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_7
  simp only [← c93786, ← c93787, k2, ← c93788] at e_1531_7
  have hr := e_1531_7
  simp only [e_1531_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14906 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1531 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14952)) := by
  have c93789 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93790 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93791 := (publicBatchWrapper86_copies2930 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93792 := (publicBatchWrapper86_copies2931 a h).1
  have c93793 := (publicBatchWrapper86_copies2931 a h).2.1
  have c93794 := (publicBatchWrapper86_copies2931 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1531_8 := arithEq_of_rows h (row := 1531) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_8
  simp only [← c93789, ← c93790, ← c93791] at e_1531_8
  have e_1531_9 := arithEq_of_rows h (row := 1531) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_9
  simp only [← c93792, ← c93793, k2, ← c93794] at e_1531_9
  have hr := e_1531_9
  simp only [e_1531_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14907 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1531 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14953)) := by
  have c93795 := (publicBatchWrapper86_copies2931 a h).2.2.2.1
  have c93796 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.1
  have c93797 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.1
  have c93798 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.1
  have c93799 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.1
  have c93800 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1531_10 := arithEq_of_rows h (row := 1531) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_10
  simp only [← c93795, ← c93796, ← c93797] at e_1531_10
  have e_1531_11 := arithEq_of_rows h (row := 1531) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_11
  simp only [← c93798, ← c93799, k2, ← c93800] at e_1531_11
  have hr := e_1531_11
  simp only [e_1531_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14908 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1531 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14954)) := by
  have c93801 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.1
  have c93802 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93803 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93804 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93805 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93806 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1531_12 := arithEq_of_rows h (row := 1531) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_12
  simp only [← c93801, ← c93802, ← c93803] at e_1531_12
  have e_1531_13 := arithEq_of_rows h (row := 1531) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_13
  simp only [← c93804, ← c93805, k2, ← c93806] at e_1531_13
  have hr := e_1531_13
  simp only [e_1531_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14909 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1531 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14955)) := by
  have c93807 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93808 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93809 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93810 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93811 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93812 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1531_14 := arithEq_of_rows h (row := 1531) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_14
  simp only [← c93807, ← c93808, ← c93809] at e_1531_14
  have e_1531_15 := arithEq_of_rows h (row := 1531) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_15
  simp only [← c93810, ← c93811, k2, ← c93812] at e_1531_15
  have hr := e_1531_15
  simp only [e_1531_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14910 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1531 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14956)) := by
  have c93813 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93814 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93815 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93816 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93817 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93818 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1531_16 := arithEq_of_rows h (row := 1531) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_16
  simp only [← c93813, ← c93814, ← c93815] at e_1531_16
  have e_1531_17 := arithEq_of_rows h (row := 1531) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_17
  simp only [← c93816, ← c93817, k2, ← c93818] at e_1531_17
  have hr := e_1531_17
  simp only [e_1531_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14911 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1531 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14957)) := by
  have c93819 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93820 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93821 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93822 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93823 := (publicBatchWrapper86_copies2931 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93824 := (publicBatchWrapper86_copies2932 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1531_18 := arithEq_of_rows h (row := 1531) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_18
  simp only [← c93819, ← c93820, ← c93821] at e_1531_18
  have e_1531_19 := arithEq_of_rows h (row := 1531) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1531_19
  simp only [← c93822, ← c93823, k2, ← c93824] at e_1531_19
  have hr := e_1531_19
  simp only [e_1531_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14912 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1532 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14958)) := by
  have c93825 := (publicBatchWrapper86_copies2932 a h).2.1
  have c93826 := (publicBatchWrapper86_copies2932 a h).2.2.1
  have c93827 := (publicBatchWrapper86_copies2932 a h).2.2.2.1
  have c93828 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.1
  have c93829 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.1
  have c93830 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1532_0 := arithEq_of_rows h (row := 1532) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_0
  simp only [← c93825, ← c93826, ← c93827] at e_1532_0
  have e_1532_1 := arithEq_of_rows h (row := 1532) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_1
  simp only [← c93828, ← c93829, k2, ← c93830] at e_1532_1
  have hr := e_1532_1
  simp only [e_1532_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14913 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1532 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14959)) := by
  have c93831 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.1
  have c93832 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.1
  have c93833 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.1
  have c93834 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93835 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93836 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1532_2 := arithEq_of_rows h (row := 1532) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_2
  simp only [← c93831, ← c93832, ← c93833] at e_1532_2
  have e_1532_3 := arithEq_of_rows h (row := 1532) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_3
  simp only [← c93834, ← c93835, k2, ← c93836] at e_1532_3
  have hr := e_1532_3
  simp only [e_1532_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14914 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1532 23) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14960)) := by
  have c93837 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93838 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93839 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93840 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93841 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93842 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1532_4 := arithEq_of_rows h (row := 1532) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_4
  simp only [← c93837, ← c93838, ← c93839] at e_1532_4
  have e_1532_5 := arithEq_of_rows h (row := 1532) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_5
  simp only [← c93840, ← c93841, k2, ← c93842] at e_1532_5
  have hr := e_1532_5
  simp only [e_1532_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14915 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1532 31) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14961)) := by
  have c93843 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93844 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93845 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93846 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93847 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93848 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1532_6 := arithEq_of_rows h (row := 1532) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_6
  simp only [← c93843, ← c93844, ← c93845] at e_1532_6
  have e_1532_7 := arithEq_of_rows h (row := 1532) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_7
  simp only [← c93846, ← c93847, k2, ← c93848] at e_1532_7
  have hr := e_1532_7
  simp only [e_1532_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14916 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1532 39) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14962)) := by
  have c93849 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93850 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93851 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93852 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93853 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93854 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1532_8 := arithEq_of_rows h (row := 1532) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_8
  simp only [← c93849, ← c93850, ← c93851] at e_1532_8
  have e_1532_9 := arithEq_of_rows h (row := 1532) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_9
  simp only [← c93852, ← c93853, k2, ← c93854] at e_1532_9
  have hr := e_1532_9
  simp only [e_1532_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14917 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1532 47) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14963)) := by
  have c93855 := (publicBatchWrapper86_copies2932 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93856 := (publicBatchWrapper86_copies2933 a h).1
  have c93857 := (publicBatchWrapper86_copies2933 a h).2.1
  have c93858 := (publicBatchWrapper86_copies2933 a h).2.2.1
  have c93859 := (publicBatchWrapper86_copies2933 a h).2.2.2.1
  have c93860 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1532_10 := arithEq_of_rows h (row := 1532) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_10
  simp only [← c93855, ← c93856, ← c93857] at e_1532_10
  have e_1532_11 := arithEq_of_rows h (row := 1532) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_11
  simp only [← c93858, ← c93859, k2, ← c93860] at e_1532_11
  have hr := e_1532_11
  simp only [e_1532_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14918 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1532 55) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14964)) := by
  have c93861 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.1
  have c93862 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.1
  have c93863 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.1
  have c93864 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.1
  have c93865 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.1
  have c93866 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1532_12 := arithEq_of_rows h (row := 1532) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_12
  simp only [← c93861, ← c93862, ← c93863] at e_1532_12
  have e_1532_13 := arithEq_of_rows h (row := 1532) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_13
  simp only [← c93864, ← c93865, k2, ← c93866] at e_1532_13
  have hr := e_1532_13
  simp only [e_1532_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14919 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1532 63) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14965)) := by
  have c93867 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93868 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93869 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93870 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93871 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93872 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1532_14 := arithEq_of_rows h (row := 1532) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_14
  simp only [← c93867, ← c93868, ← c93869] at e_1532_14
  have e_1532_15 := arithEq_of_rows h (row := 1532) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_15
  simp only [← c93870, ← c93871, k2, ← c93872] at e_1532_15
  have hr := e_1532_15
  simp only [e_1532_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14920 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1532 71) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14966)) := by
  have c93873 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93874 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93875 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93876 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93877 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93878 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1532_16 := arithEq_of_rows h (row := 1532) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_16
  simp only [← c93873, ← c93874, ← c93875] at e_1532_16
  have e_1532_17 := arithEq_of_rows h (row := 1532) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_17
  simp only [← c93876, ← c93877, k2, ← c93878] at e_1532_17
  have hr := e_1532_17
  simp only [e_1532_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14921 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1532 79) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14967)) := by
  have c93879 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93880 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93881 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93882 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93883 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93884 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1532_18 := arithEq_of_rows h (row := 1532) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_18
  simp only [← c93879, ← c93880, ← c93881] at e_1532_18
  have e_1532_19 := arithEq_of_rows h (row := 1532) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1532_19
  simp only [← c93882, ← c93883, k2, ← c93884] at e_1532_19
  have hr := e_1532_19
  simp only [e_1532_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14922 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1533 7) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14968)) := by
  have c93885 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93886 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93887 := (publicBatchWrapper86_copies2933 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93888 := (publicBatchWrapper86_copies2934 a h).1
  have c93889 := (publicBatchWrapper86_copies2934 a h).2.1
  have c93890 := (publicBatchWrapper86_copies2934 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1533_0 := arithEq_of_rows h (row := 1533) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_0
  simp only [← c93885, ← c93886, ← c93887] at e_1533_0
  have e_1533_1 := arithEq_of_rows h (row := 1533) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_1
  simp only [← c93888, ← c93889, k2, ← c93890] at e_1533_1
  have hr := e_1533_1
  simp only [e_1533_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14923 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1533 15) = bselect (a (.wire 73 3)) (a (.virt 200)) (a (.virt 14969)) := by
  have c93891 := (publicBatchWrapper86_copies2934 a h).2.2.2.1
  have c93892 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.1
  have c93893 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.1
  have c93894 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.1
  have c93895 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.1
  have c93896 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1533_2 := arithEq_of_rows h (row := 1533) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_2
  simp only [← c93891, ← c93892, ← c93893] at e_1533_2
  have e_1533_3 := arithEq_of_rows h (row := 1533) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_3
  simp only [← c93894, ← c93895, k2, ← c93896] at e_1533_3
  have hr := e_1533_3
  simp only [e_1533_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14924 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1533 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15010)) := by
  have c93897 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.1
  have c93898 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93899 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93900 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93901 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93902 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1533_4 := arithEq_of_rows h (row := 1533) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_4
  simp only [← c93897, ← c93898, ← c93899] at e_1533_4
  have e_1533_5 := arithEq_of_rows h (row := 1533) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_5
  simp only [← c93900, ← c93901, k2, ← c93902] at e_1533_5
  have hr := e_1533_5
  simp only [e_1533_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14925 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1533 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15011)) := by
  have c93903 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93904 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93905 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93906 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93907 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93908 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1533_6 := arithEq_of_rows h (row := 1533) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_6
  simp only [← c93903, ← c93904, ← c93905] at e_1533_6
  have e_1533_7 := arithEq_of_rows h (row := 1533) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_7
  simp only [← c93906, ← c93907, k2, ← c93908] at e_1533_7
  have hr := e_1533_7
  simp only [e_1533_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14926 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1533 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15012)) := by
  have c93909 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93910 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93911 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93912 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93913 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93914 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1533_8 := arithEq_of_rows h (row := 1533) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_8
  simp only [← c93909, ← c93910, ← c93911] at e_1533_8
  have e_1533_9 := arithEq_of_rows h (row := 1533) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_9
  simp only [← c93912, ← c93913, k2, ← c93914] at e_1533_9
  have hr := e_1533_9
  simp only [e_1533_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14927 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1533 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15013)) := by
  have c93915 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93916 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93917 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93918 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93919 := (publicBatchWrapper86_copies2934 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93920 := (publicBatchWrapper86_copies2935 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1533_10 := arithEq_of_rows h (row := 1533) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_10
  simp only [← c93915, ← c93916, ← c93917] at e_1533_10
  have e_1533_11 := arithEq_of_rows h (row := 1533) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_11
  simp only [← c93918, ← c93919, k2, ← c93920] at e_1533_11
  have hr := e_1533_11
  simp only [e_1533_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14928 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1533 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15014)) := by
  have c93921 := (publicBatchWrapper86_copies2935 a h).2.1
  have c93922 := (publicBatchWrapper86_copies2935 a h).2.2.1
  have c93923 := (publicBatchWrapper86_copies2935 a h).2.2.2.1
  have c93924 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.1
  have c93925 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.1
  have c93926 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1533_12 := arithEq_of_rows h (row := 1533) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_12
  simp only [← c93921, ← c93922, ← c93923] at e_1533_12
  have e_1533_13 := arithEq_of_rows h (row := 1533) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_13
  simp only [← c93924, ← c93925, k2, ← c93926] at e_1533_13
  have hr := e_1533_13
  simp only [e_1533_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14929 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1533 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15015)) := by
  have c93927 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.1
  have c93928 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.1
  have c93929 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.1
  have c93930 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93931 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93932 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1533_14 := arithEq_of_rows h (row := 1533) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_14
  simp only [← c93927, ← c93928, ← c93929] at e_1533_14
  have e_1533_15 := arithEq_of_rows h (row := 1533) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_15
  simp only [← c93930, ← c93931, k2, ← c93932] at e_1533_15
  have hr := e_1533_15
  simp only [e_1533_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14930 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1533 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15016)) := by
  have c93933 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93934 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93935 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93936 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93937 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93938 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1533_16 := arithEq_of_rows h (row := 1533) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_16
  simp only [← c93933, ← c93934, ← c93935] at e_1533_16
  have e_1533_17 := arithEq_of_rows h (row := 1533) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_17
  simp only [← c93936, ← c93937, k2, ← c93938] at e_1533_17
  have hr := e_1533_17
  simp only [e_1533_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14931 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1533 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15017)) := by
  have c93939 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93940 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93941 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93942 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93943 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93944 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1533_18 := arithEq_of_rows h (row := 1533) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_18
  simp only [← c93939, ← c93940, ← c93941] at e_1533_18
  have e_1533_19 := arithEq_of_rows h (row := 1533) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1533_19
  simp only [← c93942, ← c93943, k2, ← c93944] at e_1533_19
  have hr := e_1533_19
  simp only [e_1533_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14932 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1534 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15018)) := by
  have c93945 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93946 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93947 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93948 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93949 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93950 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1534_0 := arithEq_of_rows h (row := 1534) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_0
  simp only [← c93945, ← c93946, ← c93947] at e_1534_0
  have e_1534_1 := arithEq_of_rows h (row := 1534) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_1
  simp only [← c93948, ← c93949, k2, ← c93950] at e_1534_1
  have hr := e_1534_1
  simp only [e_1534_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14933 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1534 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15019)) := by
  have c93951 := (publicBatchWrapper86_copies2935 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93952 := (publicBatchWrapper86_copies2936 a h).1
  have c93953 := (publicBatchWrapper86_copies2936 a h).2.1
  have c93954 := (publicBatchWrapper86_copies2936 a h).2.2.1
  have c93955 := (publicBatchWrapper86_copies2936 a h).2.2.2.1
  have c93956 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1534_2 := arithEq_of_rows h (row := 1534) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_2
  simp only [← c93951, ← c93952, ← c93953] at e_1534_2
  have e_1534_3 := arithEq_of_rows h (row := 1534) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_3
  simp only [← c93954, ← c93955, k2, ← c93956] at e_1534_3
  have hr := e_1534_3
  simp only [e_1534_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14934 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1534 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15020)) := by
  have c93957 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.1
  have c93958 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.1
  have c93959 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.1
  have c93960 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.1
  have c93961 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.1
  have c93962 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1534_4 := arithEq_of_rows h (row := 1534) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_4
  simp only [← c93957, ← c93958, ← c93959] at e_1534_4
  have e_1534_5 := arithEq_of_rows h (row := 1534) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_5
  simp only [← c93960, ← c93961, k2, ← c93962] at e_1534_5
  have hr := e_1534_5
  simp only [e_1534_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14935 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1534 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15021)) := by
  have c93963 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93964 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93965 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93966 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93967 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93968 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1534_6 := arithEq_of_rows h (row := 1534) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_6
  simp only [← c93963, ← c93964, ← c93965] at e_1534_6
  have e_1534_7 := arithEq_of_rows h (row := 1534) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_7
  simp only [← c93966, ← c93967, k2, ← c93968] at e_1534_7
  have hr := e_1534_7
  simp only [e_1534_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14936 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1534 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15022)) := by
  have c93969 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93970 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93971 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93972 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93973 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93974 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1534_8 := arithEq_of_rows h (row := 1534) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_8
  simp only [← c93969, ← c93970, ← c93971] at e_1534_8
  have e_1534_9 := arithEq_of_rows h (row := 1534) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_9
  simp only [← c93972, ← c93973, k2, ← c93974] at e_1534_9
  have hr := e_1534_9
  simp only [e_1534_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14937 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1534 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15023)) := by
  have c93975 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93976 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93977 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93978 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93979 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93980 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1534_10 := arithEq_of_rows h (row := 1534) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_10
  simp only [← c93975, ← c93976, ← c93977] at e_1534_10
  have e_1534_11 := arithEq_of_rows h (row := 1534) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_11
  simp only [← c93978, ← c93979, k2, ← c93980] at e_1534_11
  have hr := e_1534_11
  simp only [e_1534_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14938 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1534 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15024)) := by
  have c93981 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93982 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93983 := (publicBatchWrapper86_copies2936 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c93984 := (publicBatchWrapper86_copies2937 a h).1
  have c93985 := (publicBatchWrapper86_copies2937 a h).2.1
  have c93986 := (publicBatchWrapper86_copies2937 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1534_12 := arithEq_of_rows h (row := 1534) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_12
  simp only [← c93981, ← c93982, ← c93983] at e_1534_12
  have e_1534_13 := arithEq_of_rows h (row := 1534) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_13
  simp only [← c93984, ← c93985, k2, ← c93986] at e_1534_13
  have hr := e_1534_13
  simp only [e_1534_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14939 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1534 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15025)) := by
  have c93987 := (publicBatchWrapper86_copies2937 a h).2.2.2.1
  have c93988 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.1
  have c93989 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.1
  have c93990 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.1
  have c93991 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.1
  have c93992 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1534_14 := arithEq_of_rows h (row := 1534) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_14
  simp only [← c93987, ← c93988, ← c93989] at e_1534_14
  have e_1534_15 := arithEq_of_rows h (row := 1534) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_15
  simp only [← c93990, ← c93991, k2, ← c93992] at e_1534_15
  have hr := e_1534_15
  simp only [e_1534_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14940 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1534 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15026)) := by
  have c93993 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.1
  have c93994 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.1
  have c93995 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c93996 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93997 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c93998 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1534_16 := arithEq_of_rows h (row := 1534) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_16
  simp only [← c93993, ← c93994, ← c93995] at e_1534_16
  have e_1534_17 := arithEq_of_rows h (row := 1534) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_17
  simp only [← c93996, ← c93997, k2, ← c93998] at e_1534_17
  have hr := e_1534_17
  simp only [e_1534_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14941 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1534 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15027)) := by
  have c93999 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94000 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94001 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94002 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94003 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94004 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1534_18 := arithEq_of_rows h (row := 1534) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_18
  simp only [← c93999, ← c94000, ← c94001] at e_1534_18
  have e_1534_19 := arithEq_of_rows h (row := 1534) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1534_19
  simp only [← c94002, ← c94003, k2, ← c94004] at e_1534_19
  have hr := e_1534_19
  simp only [e_1534_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14942 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1535 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15028)) := by
  have c94005 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94006 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94007 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94008 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94009 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94010 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1535_0 := arithEq_of_rows h (row := 1535) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_0
  simp only [← c94005, ← c94006, ← c94007] at e_1535_0
  have e_1535_1 := arithEq_of_rows h (row := 1535) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_1
  simp only [← c94008, ← c94009, k2, ← c94010] at e_1535_1
  have hr := e_1535_1
  simp only [e_1535_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14943 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1535 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15029)) := by
  have c94011 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94012 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94013 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94014 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94015 := (publicBatchWrapper86_copies2937 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94016 := (publicBatchWrapper86_copies2938 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1535_2 := arithEq_of_rows h (row := 1535) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_2
  simp only [← c94011, ← c94012, ← c94013] at e_1535_2
  have e_1535_3 := arithEq_of_rows h (row := 1535) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_3
  simp only [← c94014, ← c94015, k2, ← c94016] at e_1535_3
  have hr := e_1535_3
  simp only [e_1535_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14944 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1535 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15030)) := by
  have c94017 := (publicBatchWrapper86_copies2938 a h).2.1
  have c94018 := (publicBatchWrapper86_copies2938 a h).2.2.1
  have c94019 := (publicBatchWrapper86_copies2938 a h).2.2.2.1
  have c94020 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.1
  have c94021 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.1
  have c94022 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1535_4 := arithEq_of_rows h (row := 1535) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_4
  simp only [← c94017, ← c94018, ← c94019] at e_1535_4
  have e_1535_5 := arithEq_of_rows h (row := 1535) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_5
  simp only [← c94020, ← c94021, k2, ← c94022] at e_1535_5
  have hr := e_1535_5
  simp only [e_1535_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14945 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1535 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15031)) := by
  have c94023 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.1
  have c94024 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.1
  have c94025 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.1
  have c94026 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94027 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94028 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1535_6 := arithEq_of_rows h (row := 1535) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_6
  simp only [← c94023, ← c94024, ← c94025] at e_1535_6
  have e_1535_7 := arithEq_of_rows h (row := 1535) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_7
  simp only [← c94026, ← c94027, k2, ← c94028] at e_1535_7
  have hr := e_1535_7
  simp only [e_1535_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14946 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1535 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15032)) := by
  have c94029 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94030 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94031 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94032 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94033 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94034 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1535_8 := arithEq_of_rows h (row := 1535) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_8
  simp only [← c94029, ← c94030, ← c94031] at e_1535_8
  have e_1535_9 := arithEq_of_rows h (row := 1535) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_9
  simp only [← c94032, ← c94033, k2, ← c94034] at e_1535_9
  have hr := e_1535_9
  simp only [e_1535_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14947 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1535 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15033)) := by
  have c94035 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94036 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94037 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94038 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94039 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94040 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1535_10 := arithEq_of_rows h (row := 1535) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_10
  simp only [← c94035, ← c94036, ← c94037] at e_1535_10
  have e_1535_11 := arithEq_of_rows h (row := 1535) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_11
  simp only [← c94038, ← c94039, k2, ← c94040] at e_1535_11
  have hr := e_1535_11
  simp only [e_1535_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14948 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1535 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15034)) := by
  have c94041 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94042 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94043 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94044 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94045 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94046 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1535_12 := arithEq_of_rows h (row := 1535) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_12
  simp only [← c94041, ← c94042, ← c94043] at e_1535_12
  have e_1535_13 := arithEq_of_rows h (row := 1535) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_13
  simp only [← c94044, ← c94045, k2, ← c94046] at e_1535_13
  have hr := e_1535_13
  simp only [e_1535_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14949 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1535 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15035)) := by
  have c94047 := (publicBatchWrapper86_copies2938 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94048 := (publicBatchWrapper86_copies2939 a h).1
  have c94049 := (publicBatchWrapper86_copies2939 a h).2.1
  have c94050 := (publicBatchWrapper86_copies2939 a h).2.2.1
  have c94051 := (publicBatchWrapper86_copies2939 a h).2.2.2.1
  have c94052 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1535_14 := arithEq_of_rows h (row := 1535) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_14
  simp only [← c94047, ← c94048, ← c94049] at e_1535_14
  have e_1535_15 := arithEq_of_rows h (row := 1535) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_15
  simp only [← c94050, ← c94051, k2, ← c94052] at e_1535_15
  have hr := e_1535_15
  simp only [e_1535_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14950 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1535 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15036)) := by
  have c94053 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.1
  have c94054 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.1
  have c94055 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.1
  have c94056 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.1
  have c94057 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.1
  have c94058 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1535_16 := arithEq_of_rows h (row := 1535) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_16
  simp only [← c94053, ← c94054, ← c94055] at e_1535_16
  have e_1535_17 := arithEq_of_rows h (row := 1535) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_17
  simp only [← c94056, ← c94057, k2, ← c94058] at e_1535_17
  have hr := e_1535_17
  simp only [e_1535_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14951 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1535 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15037)) := by
  have c94059 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94060 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94061 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94062 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94063 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94064 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1535_18 := arithEq_of_rows h (row := 1535) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_18
  simp only [← c94059, ← c94060, ← c94061] at e_1535_18
  have e_1535_19 := arithEq_of_rows h (row := 1535) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1535_19
  simp only [← c94062, ← c94063, k2, ← c94064] at e_1535_19
  have hr := e_1535_19
  simp only [e_1535_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14952 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1536 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15038)) := by
  have c94065 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94066 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94067 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94068 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94069 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94070 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1536_0 := arithEq_of_rows h (row := 1536) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_0
  simp only [← c94065, ← c94066, ← c94067] at e_1536_0
  have e_1536_1 := arithEq_of_rows h (row := 1536) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_1
  simp only [← c94068, ← c94069, k2, ← c94070] at e_1536_1
  have hr := e_1536_1
  simp only [e_1536_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1536 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15039)) := by
  have c94071 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94072 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94073 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94074 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94075 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94076 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1536_2 := arithEq_of_rows h (row := 1536) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_2
  simp only [← c94071, ← c94072, ← c94073] at e_1536_2
  have e_1536_3 := arithEq_of_rows h (row := 1536) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_3
  simp only [← c94074, ← c94075, k2, ← c94076] at e_1536_3
  have hr := e_1536_3
  simp only [e_1536_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14954 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1536 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15040)) := by
  have c94077 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94078 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94079 := (publicBatchWrapper86_copies2939 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94080 := (publicBatchWrapper86_copies2940 a h).1
  have c94081 := (publicBatchWrapper86_copies2940 a h).2.1
  have c94082 := (publicBatchWrapper86_copies2940 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1536_4 := arithEq_of_rows h (row := 1536) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_4
  simp only [← c94077, ← c94078, ← c94079] at e_1536_4
  have e_1536_5 := arithEq_of_rows h (row := 1536) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_5
  simp only [← c94080, ← c94081, k2, ← c94082] at e_1536_5
  have hr := e_1536_5
  simp only [e_1536_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14955 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1536 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15041)) := by
  have c94083 := (publicBatchWrapper86_copies2940 a h).2.2.2.1
  have c94084 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.1
  have c94085 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.1
  have c94086 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.1
  have c94087 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.1
  have c94088 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1536_6 := arithEq_of_rows h (row := 1536) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_6
  simp only [← c94083, ← c94084, ← c94085] at e_1536_6
  have e_1536_7 := arithEq_of_rows h (row := 1536) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_7
  simp only [← c94086, ← c94087, k2, ← c94088] at e_1536_7
  have hr := e_1536_7
  simp only [e_1536_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14956 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1536 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15042)) := by
  have c94089 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.1
  have c94090 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94091 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94092 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94093 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94094 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1536_8 := arithEq_of_rows h (row := 1536) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_8
  simp only [← c94089, ← c94090, ← c94091] at e_1536_8
  have e_1536_9 := arithEq_of_rows h (row := 1536) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_9
  simp only [← c94092, ← c94093, k2, ← c94094] at e_1536_9
  have hr := e_1536_9
  simp only [e_1536_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14957 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1536 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15043)) := by
  have c94095 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94096 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94097 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94098 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94099 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94100 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1536_10 := arithEq_of_rows h (row := 1536) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_10
  simp only [← c94095, ← c94096, ← c94097] at e_1536_10
  have e_1536_11 := arithEq_of_rows h (row := 1536) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_11
  simp only [← c94098, ← c94099, k2, ← c94100] at e_1536_11
  have hr := e_1536_11
  simp only [e_1536_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14958 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1536 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15044)) := by
  have c94101 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94102 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94103 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94104 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94105 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94106 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1536_12 := arithEq_of_rows h (row := 1536) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_12
  simp only [← c94101, ← c94102, ← c94103] at e_1536_12
  have e_1536_13 := arithEq_of_rows h (row := 1536) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_13
  simp only [← c94104, ← c94105, k2, ← c94106] at e_1536_13
  have hr := e_1536_13
  simp only [e_1536_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14959 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1536 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15045)) := by
  have c94107 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94108 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94109 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94110 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94111 := (publicBatchWrapper86_copies2940 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94112 := (publicBatchWrapper86_copies2941 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1536_14 := arithEq_of_rows h (row := 1536) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_14
  simp only [← c94107, ← c94108, ← c94109] at e_1536_14
  have e_1536_15 := arithEq_of_rows h (row := 1536) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_15
  simp only [← c94110, ← c94111, k2, ← c94112] at e_1536_15
  have hr := e_1536_15
  simp only [e_1536_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14960 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1536 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15046)) := by
  have c94113 := (publicBatchWrapper86_copies2941 a h).2.1
  have c94114 := (publicBatchWrapper86_copies2941 a h).2.2.1
  have c94115 := (publicBatchWrapper86_copies2941 a h).2.2.2.1
  have c94116 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.1
  have c94117 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.1
  have c94118 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1536_16 := arithEq_of_rows h (row := 1536) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_16
  simp only [← c94113, ← c94114, ← c94115] at e_1536_16
  have e_1536_17 := arithEq_of_rows h (row := 1536) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_17
  simp only [← c94116, ← c94117, k2, ← c94118] at e_1536_17
  have hr := e_1536_17
  simp only [e_1536_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14961 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1536 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15047)) := by
  have c94119 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.1
  have c94120 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.1
  have c94121 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.1
  have c94122 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94123 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94124 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1536_18 := arithEq_of_rows h (row := 1536) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_18
  simp only [← c94119, ← c94120, ← c94121] at e_1536_18
  have e_1536_19 := arithEq_of_rows h (row := 1536) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1536_19
  simp only [← c94122, ← c94123, k2, ← c94124] at e_1536_19
  have hr := e_1536_19
  simp only [e_1536_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14962 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1537 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15048)) := by
  have c94125 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94126 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94127 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94128 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94129 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94130 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1537_0 := arithEq_of_rows h (row := 1537) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_0
  simp only [← c94125, ← c94126, ← c94127] at e_1537_0
  have e_1537_1 := arithEq_of_rows h (row := 1537) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_1
  simp only [← c94128, ← c94129, k2, ← c94130] at e_1537_1
  have hr := e_1537_1
  simp only [e_1537_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14963 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1537 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15049)) := by
  have c94131 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94132 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94133 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94134 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94135 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94136 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1537_2 := arithEq_of_rows h (row := 1537) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_2
  simp only [← c94131, ← c94132, ← c94133] at e_1537_2
  have e_1537_3 := arithEq_of_rows h (row := 1537) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_3
  simp only [← c94134, ← c94135, k2, ← c94136] at e_1537_3
  have hr := e_1537_3
  simp only [e_1537_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14964 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1537 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15050)) := by
  have c94137 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94138 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94139 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94140 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94141 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94142 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1537_4 := arithEq_of_rows h (row := 1537) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_4
  simp only [← c94137, ← c94138, ← c94139] at e_1537_4
  have e_1537_5 := arithEq_of_rows h (row := 1537) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_5
  simp only [← c94140, ← c94141, k2, ← c94142] at e_1537_5
  have hr := e_1537_5
  simp only [e_1537_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14965 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1537 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15051)) := by
  have c94143 := (publicBatchWrapper86_copies2941 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94144 := (publicBatchWrapper86_copies2942 a h).1
  have c94145 := (publicBatchWrapper86_copies2942 a h).2.1
  have c94146 := (publicBatchWrapper86_copies2942 a h).2.2.1
  have c94147 := (publicBatchWrapper86_copies2942 a h).2.2.2.1
  have c94148 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1537_6 := arithEq_of_rows h (row := 1537) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_6
  simp only [← c94143, ← c94144, ← c94145] at e_1537_6
  have e_1537_7 := arithEq_of_rows h (row := 1537) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_7
  simp only [← c94146, ← c94147, k2, ← c94148] at e_1537_7
  have hr := e_1537_7
  simp only [e_1537_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14966 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1537 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15052)) := by
  have c94149 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.1
  have c94150 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.1
  have c94151 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.1
  have c94152 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.1
  have c94153 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.1
  have c94154 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1537_8 := arithEq_of_rows h (row := 1537) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_8
  simp only [← c94149, ← c94150, ← c94151] at e_1537_8
  have e_1537_9 := arithEq_of_rows h (row := 1537) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_9
  simp only [← c94152, ← c94153, k2, ← c94154] at e_1537_9
  have hr := e_1537_9
  simp only [e_1537_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14967 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1537 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15053)) := by
  have c94155 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94156 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94157 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94158 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94159 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94160 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1537_10 := arithEq_of_rows h (row := 1537) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_10
  simp only [← c94155, ← c94156, ← c94157] at e_1537_10
  have e_1537_11 := arithEq_of_rows h (row := 1537) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_11
  simp only [← c94158, ← c94159, k2, ← c94160] at e_1537_11
  have hr := e_1537_11
  simp only [e_1537_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14968 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1537 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15054)) := by
  have c94161 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94162 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94163 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94164 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94165 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94166 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1537_12 := arithEq_of_rows h (row := 1537) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_12
  simp only [← c94161, ← c94162, ← c94163] at e_1537_12
  have e_1537_13 := arithEq_of_rows h (row := 1537) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_13
  simp only [← c94164, ← c94165, k2, ← c94166] at e_1537_13
  have hr := e_1537_13
  simp only [e_1537_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14969 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1537 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15055)) := by
  have c94167 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94168 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94169 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94170 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94171 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94172 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1537_14 := arithEq_of_rows h (row := 1537) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_14
  simp only [← c94167, ← c94168, ← c94169] at e_1537_14
  have e_1537_15 := arithEq_of_rows h (row := 1537) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_15
  simp only [← c94170, ← c94171, k2, ← c94172] at e_1537_15
  have hr := e_1537_15
  simp only [e_1537_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14970 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1537 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15056)) := by
  have c94173 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94174 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94175 := (publicBatchWrapper86_copies2942 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94176 := (publicBatchWrapper86_copies2943 a h).1
  have c94177 := (publicBatchWrapper86_copies2943 a h).2.1
  have c94178 := (publicBatchWrapper86_copies2943 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1537_16 := arithEq_of_rows h (row := 1537) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_16
  simp only [← c94173, ← c94174, ← c94175] at e_1537_16
  have e_1537_17 := arithEq_of_rows h (row := 1537) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_17
  simp only [← c94176, ← c94177, k2, ← c94178] at e_1537_17
  have hr := e_1537_17
  simp only [e_1537_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14971 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1537 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15057)) := by
  have c94179 := (publicBatchWrapper86_copies2943 a h).2.2.2.1
  have c94180 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.1
  have c94181 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.1
  have c94182 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.1
  have c94183 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.1
  have c94184 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1537_18 := arithEq_of_rows h (row := 1537) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_18
  simp only [← c94179, ← c94180, ← c94181] at e_1537_18
  have e_1537_19 := arithEq_of_rows h (row := 1537) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1537_19
  simp only [← c94182, ← c94183, k2, ← c94184] at e_1537_19
  have hr := e_1537_19
  simp only [e_1537_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14972 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1538 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15058)) := by
  have c94185 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.1
  have c94186 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94187 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94188 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94189 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94190 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1538_0 := arithEq_of_rows h (row := 1538) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_0
  simp only [← c94185, ← c94186, ← c94187] at e_1538_0
  have e_1538_1 := arithEq_of_rows h (row := 1538) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_1
  simp only [← c94188, ← c94189, k2, ← c94190] at e_1538_1
  have hr := e_1538_1
  simp only [e_1538_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14973 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1538 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15059)) := by
  have c94191 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94192 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94193 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94194 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94195 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94196 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1538_2 := arithEq_of_rows h (row := 1538) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_2
  simp only [← c94191, ← c94192, ← c94193] at e_1538_2
  have e_1538_3 := arithEq_of_rows h (row := 1538) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_3
  simp only [← c94194, ← c94195, k2, ← c94196] at e_1538_3
  have hr := e_1538_3
  simp only [e_1538_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1538 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15060)) := by
  have c94197 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94198 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94199 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94200 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94201 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94202 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1538_4 := arithEq_of_rows h (row := 1538) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_4
  simp only [← c94197, ← c94198, ← c94199] at e_1538_4
  have e_1538_5 := arithEq_of_rows h (row := 1538) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_5
  simp only [← c94200, ← c94201, k2, ← c94202] at e_1538_5
  have hr := e_1538_5
  simp only [e_1538_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1538 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15061)) := by
  have c94203 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94204 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94205 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94206 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94207 := (publicBatchWrapper86_copies2943 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94208 := (publicBatchWrapper86_copies2944 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1538_6 := arithEq_of_rows h (row := 1538) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_6
  simp only [← c94203, ← c94204, ← c94205] at e_1538_6
  have e_1538_7 := arithEq_of_rows h (row := 1538) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_7
  simp only [← c94206, ← c94207, k2, ← c94208] at e_1538_7
  have hr := e_1538_7
  simp only [e_1538_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1538 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15062)) := by
  have c94209 := (publicBatchWrapper86_copies2944 a h).2.1
  have c94210 := (publicBatchWrapper86_copies2944 a h).2.2.1
  have c94211 := (publicBatchWrapper86_copies2944 a h).2.2.2.1
  have c94212 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.1
  have c94213 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.1
  have c94214 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1538_8 := arithEq_of_rows h (row := 1538) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_8
  simp only [← c94209, ← c94210, ← c94211] at e_1538_8
  have e_1538_9 := arithEq_of_rows h (row := 1538) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_9
  simp only [← c94212, ← c94213, k2, ← c94214] at e_1538_9
  have hr := e_1538_9
  simp only [e_1538_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1538 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15063)) := by
  have c94215 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.1
  have c94216 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.1
  have c94217 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.1
  have c94218 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94219 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94220 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1538_10 := arithEq_of_rows h (row := 1538) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_10
  simp only [← c94215, ← c94216, ← c94217] at e_1538_10
  have e_1538_11 := arithEq_of_rows h (row := 1538) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_11
  simp only [← c94218, ← c94219, k2, ← c94220] at e_1538_11
  have hr := e_1538_11
  simp only [e_1538_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1538 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15064)) := by
  have c94221 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94222 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94223 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94224 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94225 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94226 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1538_12 := arithEq_of_rows h (row := 1538) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_12
  simp only [← c94221, ← c94222, ← c94223] at e_1538_12
  have e_1538_13 := arithEq_of_rows h (row := 1538) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_13
  simp only [← c94224, ← c94225, k2, ← c94226] at e_1538_13
  have hr := e_1538_13
  simp only [e_1538_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1538 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15065)) := by
  have c94227 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94228 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94229 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94230 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94231 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94232 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1538_14 := arithEq_of_rows h (row := 1538) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_14
  simp only [← c94227, ← c94228, ← c94229] at e_1538_14
  have e_1538_15 := arithEq_of_rows h (row := 1538) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_15
  simp only [← c94230, ← c94231, k2, ← c94232] at e_1538_15
  have hr := e_1538_15
  simp only [e_1538_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1538 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15066)) := by
  have c94233 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94234 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94235 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94236 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94237 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94238 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1538_16 := arithEq_of_rows h (row := 1538) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_16
  simp only [← c94233, ← c94234, ← c94235] at e_1538_16
  have e_1538_17 := arithEq_of_rows h (row := 1538) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_17
  simp only [← c94236, ← c94237, k2, ← c94238] at e_1538_17
  have hr := e_1538_17
  simp only [e_1538_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1538 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15067)) := by
  have c94239 := (publicBatchWrapper86_copies2944 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94240 := (publicBatchWrapper86_copies2945 a h).1
  have c94241 := (publicBatchWrapper86_copies2945 a h).2.1
  have c94242 := (publicBatchWrapper86_copies2945 a h).2.2.1
  have c94243 := (publicBatchWrapper86_copies2945 a h).2.2.2.1
  have c94244 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1538_18 := arithEq_of_rows h (row := 1538) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_18
  simp only [← c94239, ← c94240, ← c94241] at e_1538_18
  have e_1538_19 := arithEq_of_rows h (row := 1538) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1538_19
  simp only [← c94242, ← c94243, k2, ← c94244] at e_1538_19
  have hr := e_1538_19
  simp only [e_1538_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1539 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15068)) := by
  have c94245 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.1
  have c94246 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.1
  have c94247 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.1
  have c94248 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.1
  have c94249 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.1
  have c94250 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1539_0 := arithEq_of_rows h (row := 1539) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_0
  simp only [← c94245, ← c94246, ← c94247] at e_1539_0
  have e_1539_1 := arithEq_of_rows h (row := 1539) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_1
  simp only [← c94248, ← c94249, k2, ← c94250] at e_1539_1
  have hr := e_1539_1
  simp only [e_1539_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1539 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15069)) := by
  have c94251 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94252 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94253 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94254 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94255 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94256 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1539_2 := arithEq_of_rows h (row := 1539) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_2
  simp only [← c94251, ← c94252, ← c94253] at e_1539_2
  have e_1539_3 := arithEq_of_rows h (row := 1539) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_3
  simp only [← c94254, ← c94255, k2, ← c94256] at e_1539_3
  have hr := e_1539_3
  simp only [e_1539_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1539 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15070)) := by
  have c94257 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94258 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94259 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94260 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94261 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94262 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1539_4 := arithEq_of_rows h (row := 1539) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_4
  simp only [← c94257, ← c94258, ← c94259] at e_1539_4
  have e_1539_5 := arithEq_of_rows h (row := 1539) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_5
  simp only [← c94260, ← c94261, k2, ← c94262] at e_1539_5
  have hr := e_1539_5
  simp only [e_1539_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1539 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15071)) := by
  have c94263 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94264 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94265 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94266 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94267 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94268 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1539_6 := arithEq_of_rows h (row := 1539) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_6
  simp only [← c94263, ← c94264, ← c94265] at e_1539_6
  have e_1539_7 := arithEq_of_rows h (row := 1539) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_7
  simp only [← c94266, ← c94267, k2, ← c94268] at e_1539_7
  have hr := e_1539_7
  simp only [e_1539_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1539 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15072)) := by
  have c94269 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94270 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94271 := (publicBatchWrapper86_copies2945 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94272 := (publicBatchWrapper86_copies2946 a h).1
  have c94273 := (publicBatchWrapper86_copies2946 a h).2.1
  have c94274 := (publicBatchWrapper86_copies2946 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1539_8 := arithEq_of_rows h (row := 1539) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_8
  simp only [← c94269, ← c94270, ← c94271] at e_1539_8
  have e_1539_9 := arithEq_of_rows h (row := 1539) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_9
  simp only [← c94272, ← c94273, k2, ← c94274] at e_1539_9
  have hr := e_1539_9
  simp only [e_1539_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1539 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15073)) := by
  have c94275 := (publicBatchWrapper86_copies2946 a h).2.2.2.1
  have c94276 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.1
  have c94277 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.1
  have c94278 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.1
  have c94279 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.1
  have c94280 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1539_10 := arithEq_of_rows h (row := 1539) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_10
  simp only [← c94275, ← c94276, ← c94277] at e_1539_10
  have e_1539_11 := arithEq_of_rows h (row := 1539) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_11
  simp only [← c94278, ← c94279, k2, ← c94280] at e_1539_11
  have hr := e_1539_11
  simp only [e_1539_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1539 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15074)) := by
  have c94281 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.1
  have c94282 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94283 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94284 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94285 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94286 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1539_12 := arithEq_of_rows h (row := 1539) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_12
  simp only [← c94281, ← c94282, ← c94283] at e_1539_12
  have e_1539_13 := arithEq_of_rows h (row := 1539) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_13
  simp only [← c94284, ← c94285, k2, ← c94286] at e_1539_13
  have hr := e_1539_13
  simp only [e_1539_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1539 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15075)) := by
  have c94287 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94288 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94289 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94290 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94291 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94292 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1539_14 := arithEq_of_rows h (row := 1539) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_14
  simp only [← c94287, ← c94288, ← c94289] at e_1539_14
  have e_1539_15 := arithEq_of_rows h (row := 1539) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_15
  simp only [← c94290, ← c94291, k2, ← c94292] at e_1539_15
  have hr := e_1539_15
  simp only [e_1539_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1539 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15076)) := by
  have c94293 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94294 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94295 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94296 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94297 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94298 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1539_16 := arithEq_of_rows h (row := 1539) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_16
  simp only [← c94293, ← c94294, ← c94295] at e_1539_16
  have e_1539_17 := arithEq_of_rows h (row := 1539) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_17
  simp only [← c94296, ← c94297, k2, ← c94298] at e_1539_17
  have hr := e_1539_17
  simp only [e_1539_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1539 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15077)) := by
  have c94299 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94300 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94301 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94302 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94303 := (publicBatchWrapper86_copies2946 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94304 := (publicBatchWrapper86_copies2947 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1539_18 := arithEq_of_rows h (row := 1539) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_18
  simp only [← c94299, ← c94300, ← c94301] at e_1539_18
  have e_1539_19 := arithEq_of_rows h (row := 1539) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1539_19
  simp only [← c94302, ← c94303, k2, ← c94304] at e_1539_19
  have hr := e_1539_19
  simp only [e_1539_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1540 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15078)) := by
  have c94305 := (publicBatchWrapper86_copies2947 a h).2.1
  have c94306 := (publicBatchWrapper86_copies2947 a h).2.2.1
  have c94307 := (publicBatchWrapper86_copies2947 a h).2.2.2.1
  have c94308 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.1
  have c94309 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.1
  have c94310 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1540_0 := arithEq_of_rows h (row := 1540) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_0
  simp only [← c94305, ← c94306, ← c94307] at e_1540_0
  have e_1540_1 := arithEq_of_rows h (row := 1540) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_1
  simp only [← c94308, ← c94309, k2, ← c94310] at e_1540_1
  have hr := e_1540_1
  simp only [e_1540_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1540 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15079)) := by
  have c94311 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.1
  have c94312 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.1
  have c94313 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.1
  have c94314 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94315 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94316 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1540_2 := arithEq_of_rows h (row := 1540) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_2
  simp only [← c94311, ← c94312, ← c94313] at e_1540_2
  have e_1540_3 := arithEq_of_rows h (row := 1540) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_3
  simp only [← c94314, ← c94315, k2, ← c94316] at e_1540_3
  have hr := e_1540_3
  simp only [e_1540_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1540 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15080)) := by
  have c94317 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94318 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94319 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94320 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94321 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94322 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1540_4 := arithEq_of_rows h (row := 1540) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_4
  simp only [← c94317, ← c94318, ← c94319] at e_1540_4
  have e_1540_5 := arithEq_of_rows h (row := 1540) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_5
  simp only [← c94320, ← c94321, k2, ← c94322] at e_1540_5
  have hr := e_1540_5
  simp only [e_1540_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1540 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15081)) := by
  have c94323 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94324 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94325 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94326 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94327 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94328 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1540_6 := arithEq_of_rows h (row := 1540) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_6
  simp only [← c94323, ← c94324, ← c94325] at e_1540_6
  have e_1540_7 := arithEq_of_rows h (row := 1540) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_7
  simp only [← c94326, ← c94327, k2, ← c94328] at e_1540_7
  have hr := e_1540_7
  simp only [e_1540_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1540 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15082)) := by
  have c94329 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94330 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94331 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94332 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94333 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94334 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1540_8 := arithEq_of_rows h (row := 1540) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_8
  simp only [← c94329, ← c94330, ← c94331] at e_1540_8
  have e_1540_9 := arithEq_of_rows h (row := 1540) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_9
  simp only [← c94332, ← c94333, k2, ← c94334] at e_1540_9
  have hr := e_1540_9
  simp only [e_1540_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1540 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15083)) := by
  have c94335 := (publicBatchWrapper86_copies2947 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94336 := (publicBatchWrapper86_copies2948 a h).1
  have c94337 := (publicBatchWrapper86_copies2948 a h).2.1
  have c94338 := (publicBatchWrapper86_copies2948 a h).2.2.1
  have c94339 := (publicBatchWrapper86_copies2948 a h).2.2.2.1
  have c94340 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1540_10 := arithEq_of_rows h (row := 1540) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_10
  simp only [← c94335, ← c94336, ← c94337] at e_1540_10
  have e_1540_11 := arithEq_of_rows h (row := 1540) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_11
  simp only [← c94338, ← c94339, k2, ← c94340] at e_1540_11
  have hr := e_1540_11
  simp only [e_1540_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1540 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15084)) := by
  have c94341 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.1
  have c94342 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.1
  have c94343 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.1
  have c94344 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.1
  have c94345 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.1
  have c94346 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1540_12 := arithEq_of_rows h (row := 1540) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_12
  simp only [← c94341, ← c94342, ← c94343] at e_1540_12
  have e_1540_13 := arithEq_of_rows h (row := 1540) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_13
  simp only [← c94344, ← c94345, k2, ← c94346] at e_1540_13
  have hr := e_1540_13
  simp only [e_1540_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1540 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15085)) := by
  have c94347 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94348 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94349 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94350 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94351 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94352 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1540_14 := arithEq_of_rows h (row := 1540) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_14
  simp only [← c94347, ← c94348, ← c94349] at e_1540_14
  have e_1540_15 := arithEq_of_rows h (row := 1540) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_15
  simp only [← c94350, ← c94351, k2, ← c94352] at e_1540_15
  have hr := e_1540_15
  simp only [e_1540_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1540 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15086)) := by
  have c94353 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94354 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94355 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94356 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94357 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94358 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1540_16 := arithEq_of_rows h (row := 1540) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_16
  simp only [← c94353, ← c94354, ← c94355] at e_1540_16
  have e_1540_17 := arithEq_of_rows h (row := 1540) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_17
  simp only [← c94356, ← c94357, k2, ← c94358] at e_1540_17
  have hr := e_1540_17
  simp only [e_1540_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1540 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15087)) := by
  have c94359 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94360 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94361 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94362 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94363 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94364 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1540_18 := arithEq_of_rows h (row := 1540) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_18
  simp only [← c94359, ← c94360, ← c94361] at e_1540_18
  have e_1540_19 := arithEq_of_rows h (row := 1540) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1540_19
  simp only [← c94362, ← c94363, k2, ← c94364] at e_1540_19
  have hr := e_1540_19
  simp only [e_1540_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1541 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15088)) := by
  have c94365 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94366 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94367 := (publicBatchWrapper86_copies2948 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94368 := (publicBatchWrapper86_copies2949 a h).1
  have c94369 := (publicBatchWrapper86_copies2949 a h).2.1
  have c94370 := (publicBatchWrapper86_copies2949 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1541_0 := arithEq_of_rows h (row := 1541) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_0
  simp only [← c94365, ← c94366, ← c94367] at e_1541_0
  have e_1541_1 := arithEq_of_rows h (row := 1541) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_1
  simp only [← c94368, ← c94369, k2, ← c94370] at e_1541_1
  have hr := e_1541_1
  simp only [e_1541_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1541 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15089)) := by
  have c94371 := (publicBatchWrapper86_copies2949 a h).2.2.2.1
  have c94372 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.1
  have c94373 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.1
  have c94374 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.1
  have c94375 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.1
  have c94376 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1541_2 := arithEq_of_rows h (row := 1541) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_2
  simp only [← c94371, ← c94372, ← c94373] at e_1541_2
  have e_1541_3 := arithEq_of_rows h (row := 1541) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_3
  simp only [← c94374, ← c94375, k2, ← c94376] at e_1541_3
  have hr := e_1541_3
  simp only [e_1541_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1541 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15090)) := by
  have c94377 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.1
  have c94378 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94379 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94380 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94381 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94382 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1541_4 := arithEq_of_rows h (row := 1541) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_4
  simp only [← c94377, ← c94378, ← c94379] at e_1541_4
  have e_1541_5 := arithEq_of_rows h (row := 1541) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_5
  simp only [← c94380, ← c94381, k2, ← c94382] at e_1541_5
  have hr := e_1541_5
  simp only [e_1541_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1541 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15091)) := by
  have c94383 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94384 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94385 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94386 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94387 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94388 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1541_6 := arithEq_of_rows h (row := 1541) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_6
  simp only [← c94383, ← c94384, ← c94385] at e_1541_6
  have e_1541_7 := arithEq_of_rows h (row := 1541) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_7
  simp only [← c94386, ← c94387, k2, ← c94388] at e_1541_7
  have hr := e_1541_7
  simp only [e_1541_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1541 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15092)) := by
  have c94389 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94390 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94391 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94392 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94393 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94394 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1541_8 := arithEq_of_rows h (row := 1541) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_8
  simp only [← c94389, ← c94390, ← c94391] at e_1541_8
  have e_1541_9 := arithEq_of_rows h (row := 1541) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_9
  simp only [← c94392, ← c94393, k2, ← c94394] at e_1541_9
  have hr := e_1541_9
  simp only [e_1541_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1541 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15093)) := by
  have c94395 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94396 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94397 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94398 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94399 := (publicBatchWrapper86_copies2949 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94400 := (publicBatchWrapper86_copies2950 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1541_10 := arithEq_of_rows h (row := 1541) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_10
  simp only [← c94395, ← c94396, ← c94397] at e_1541_10
  have e_1541_11 := arithEq_of_rows h (row := 1541) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_11
  simp only [← c94398, ← c94399, k2, ← c94400] at e_1541_11
  have hr := e_1541_11
  simp only [e_1541_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1541 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15094)) := by
  have c94401 := (publicBatchWrapper86_copies2950 a h).2.1
  have c94402 := (publicBatchWrapper86_copies2950 a h).2.2.1
  have c94403 := (publicBatchWrapper86_copies2950 a h).2.2.2.1
  have c94404 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.1
  have c94405 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.1
  have c94406 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1541_12 := arithEq_of_rows h (row := 1541) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_12
  simp only [← c94401, ← c94402, ← c94403] at e_1541_12
  have e_1541_13 := arithEq_of_rows h (row := 1541) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_13
  simp only [← c94404, ← c94405, k2, ← c94406] at e_1541_13
  have hr := e_1541_13
  simp only [e_1541_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1541 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15095)) := by
  have c94407 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.1
  have c94408 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.1
  have c94409 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.1
  have c94410 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94411 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94412 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1541_14 := arithEq_of_rows h (row := 1541) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_14
  simp only [← c94407, ← c94408, ← c94409] at e_1541_14
  have e_1541_15 := arithEq_of_rows h (row := 1541) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_15
  simp only [← c94410, ← c94411, k2, ← c94412] at e_1541_15
  have hr := e_1541_15
  simp only [e_1541_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1541 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15096)) := by
  have c94413 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94414 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94415 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94416 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94417 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94418 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1541_16 := arithEq_of_rows h (row := 1541) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_16
  simp only [← c94413, ← c94414, ← c94415] at e_1541_16
  have e_1541_17 := arithEq_of_rows h (row := 1541) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_17
  simp only [← c94416, ← c94417, k2, ← c94418] at e_1541_17
  have hr := e_1541_17
  simp only [e_1541_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1541 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15097)) := by
  have c94419 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94420 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94421 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94422 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94423 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94424 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1541_18 := arithEq_of_rows h (row := 1541) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_18
  simp only [← c94419, ← c94420, ← c94421] at e_1541_18
  have e_1541_19 := arithEq_of_rows h (row := 1541) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1541_19
  simp only [← c94422, ← c94423, k2, ← c94424] at e_1541_19
  have hr := e_1541_19
  simp only [e_1541_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1542 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15098)) := by
  have c94425 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94426 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94427 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94428 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94429 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94430 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1542_0 := arithEq_of_rows h (row := 1542) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_0
  simp only [← c94425, ← c94426, ← c94427] at e_1542_0
  have e_1542_1 := arithEq_of_rows h (row := 1542) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_1
  simp only [← c94428, ← c94429, k2, ← c94430] at e_1542_1
  have hr := e_1542_1
  simp only [e_1542_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1542 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15099)) := by
  have c94431 := (publicBatchWrapper86_copies2950 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94432 := (publicBatchWrapper86_copies2951 a h).1
  have c94433 := (publicBatchWrapper86_copies2951 a h).2.1
  have c94434 := (publicBatchWrapper86_copies2951 a h).2.2.1
  have c94435 := (publicBatchWrapper86_copies2951 a h).2.2.2.1
  have c94436 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1542_2 := arithEq_of_rows h (row := 1542) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_2
  simp only [← c94431, ← c94432, ← c94433] at e_1542_2
  have e_1542_3 := arithEq_of_rows h (row := 1542) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_3
  simp only [← c94434, ← c94435, k2, ← c94436] at e_1542_3
  have hr := e_1542_3
  simp only [e_1542_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1542 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15100)) := by
  have c94437 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.1
  have c94438 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.1
  have c94439 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.1
  have c94440 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.1
  have c94441 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.1
  have c94442 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1542_4 := arithEq_of_rows h (row := 1542) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_4
  simp only [← c94437, ← c94438, ← c94439] at e_1542_4
  have e_1542_5 := arithEq_of_rows h (row := 1542) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_5
  simp only [← c94440, ← c94441, k2, ← c94442] at e_1542_5
  have hr := e_1542_5
  simp only [e_1542_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1542 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15101)) := by
  have c94443 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94444 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94445 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94446 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94447 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94448 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1542_6 := arithEq_of_rows h (row := 1542) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_6
  simp only [← c94443, ← c94444, ← c94445] at e_1542_6
  have e_1542_7 := arithEq_of_rows h (row := 1542) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_7
  simp only [← c94446, ← c94447, k2, ← c94448] at e_1542_7
  have hr := e_1542_7
  simp only [e_1542_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1542 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15102)) := by
  have c94449 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94450 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94451 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94452 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94453 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94454 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1542_8 := arithEq_of_rows h (row := 1542) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_8
  simp only [← c94449, ← c94450, ← c94451] at e_1542_8
  have e_1542_9 := arithEq_of_rows h (row := 1542) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_9
  simp only [← c94452, ← c94453, k2, ← c94454] at e_1542_9
  have hr := e_1542_9
  simp only [e_1542_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1542 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15103)) := by
  have c94455 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94456 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94457 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94458 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94459 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94460 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1542_10 := arithEq_of_rows h (row := 1542) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_10
  simp only [← c94455, ← c94456, ← c94457] at e_1542_10
  have e_1542_11 := arithEq_of_rows h (row := 1542) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_11
  simp only [← c94458, ← c94459, k2, ← c94460] at e_1542_11
  have hr := e_1542_11
  simp only [e_1542_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1542 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15104)) := by
  have c94461 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94462 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94463 := (publicBatchWrapper86_copies2951 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94464 := (publicBatchWrapper86_copies2952 a h).1
  have c94465 := (publicBatchWrapper86_copies2952 a h).2.1
  have c94466 := (publicBatchWrapper86_copies2952 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1542_12 := arithEq_of_rows h (row := 1542) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_12
  simp only [← c94461, ← c94462, ← c94463] at e_1542_12
  have e_1542_13 := arithEq_of_rows h (row := 1542) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_13
  simp only [← c94464, ← c94465, k2, ← c94466] at e_1542_13
  have hr := e_1542_13
  simp only [e_1542_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1542 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15105)) := by
  have c94467 := (publicBatchWrapper86_copies2952 a h).2.2.2.1
  have c94468 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.1
  have c94469 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.1
  have c94470 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.1
  have c94471 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.1
  have c94472 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1542_14 := arithEq_of_rows h (row := 1542) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_14
  simp only [← c94467, ← c94468, ← c94469] at e_1542_14
  have e_1542_15 := arithEq_of_rows h (row := 1542) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_15
  simp only [← c94470, ← c94471, k2, ← c94472] at e_1542_15
  have hr := e_1542_15
  simp only [e_1542_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1542 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15106)) := by
  have c94473 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.1
  have c94474 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94475 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94476 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94477 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94478 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1542_16 := arithEq_of_rows h (row := 1542) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_16
  simp only [← c94473, ← c94474, ← c94475] at e_1542_16
  have e_1542_17 := arithEq_of_rows h (row := 1542) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_17
  simp only [← c94476, ← c94477, k2, ← c94478] at e_1542_17
  have hr := e_1542_17
  simp only [e_1542_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1542 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15107)) := by
  have c94479 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94480 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94481 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94482 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94483 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94484 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1542_18 := arithEq_of_rows h (row := 1542) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_18
  simp only [← c94479, ← c94480, ← c94481] at e_1542_18
  have e_1542_19 := arithEq_of_rows h (row := 1542) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1542_19
  simp only [← c94482, ← c94483, k2, ← c94484] at e_1542_19
  have hr := e_1542_19
  simp only [e_1542_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1543 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15108)) := by
  have c94485 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94486 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94487 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94488 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94489 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94490 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1543_0 := arithEq_of_rows h (row := 1543) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_0
  simp only [← c94485, ← c94486, ← c94487] at e_1543_0
  have e_1543_1 := arithEq_of_rows h (row := 1543) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_1
  simp only [← c94488, ← c94489, k2, ← c94490] at e_1543_1
  have hr := e_1543_1
  simp only [e_1543_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1543 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15109)) := by
  have c94491 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94492 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94493 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94494 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94495 := (publicBatchWrapper86_copies2952 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94496 := (publicBatchWrapper86_copies2953 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1543_2 := arithEq_of_rows h (row := 1543) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_2
  simp only [← c94491, ← c94492, ← c94493] at e_1543_2
  have e_1543_3 := arithEq_of_rows h (row := 1543) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_3
  simp only [← c94494, ← c94495, k2, ← c94496] at e_1543_3
  have hr := e_1543_3
  simp only [e_1543_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1543 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15110)) := by
  have c94497 := (publicBatchWrapper86_copies2953 a h).2.1
  have c94498 := (publicBatchWrapper86_copies2953 a h).2.2.1
  have c94499 := (publicBatchWrapper86_copies2953 a h).2.2.2.1
  have c94500 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.1
  have c94501 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.1
  have c94502 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1543_4 := arithEq_of_rows h (row := 1543) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_4
  simp only [← c94497, ← c94498, ← c94499] at e_1543_4
  have e_1543_5 := arithEq_of_rows h (row := 1543) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_5
  simp only [← c94500, ← c94501, k2, ← c94502] at e_1543_5
  have hr := e_1543_5
  simp only [e_1543_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1543 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15111)) := by
  have c94503 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.1
  have c94504 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.1
  have c94505 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.1
  have c94506 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94507 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94508 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1543_6 := arithEq_of_rows h (row := 1543) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_6
  simp only [← c94503, ← c94504, ← c94505] at e_1543_6
  have e_1543_7 := arithEq_of_rows h (row := 1543) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_7
  simp only [← c94506, ← c94507, k2, ← c94508] at e_1543_7
  have hr := e_1543_7
  simp only [e_1543_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1543 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15112)) := by
  have c94509 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94510 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94511 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94512 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94513 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94514 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1543_8 := arithEq_of_rows h (row := 1543) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_8
  simp only [← c94509, ← c94510, ← c94511] at e_1543_8
  have e_1543_9 := arithEq_of_rows h (row := 1543) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_9
  simp only [← c94512, ← c94513, k2, ← c94514] at e_1543_9
  have hr := e_1543_9
  simp only [e_1543_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1543 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15113)) := by
  have c94515 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94516 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94517 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94518 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94519 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94520 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1543_10 := arithEq_of_rows h (row := 1543) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_10
  simp only [← c94515, ← c94516, ← c94517] at e_1543_10
  have e_1543_11 := arithEq_of_rows h (row := 1543) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_11
  simp only [← c94518, ← c94519, k2, ← c94520] at e_1543_11
  have hr := e_1543_11
  simp only [e_1543_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1543 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15114)) := by
  have c94521 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94522 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94523 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94524 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94525 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94526 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1543_12 := arithEq_of_rows h (row := 1543) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_12
  simp only [← c94521, ← c94522, ← c94523] at e_1543_12
  have e_1543_13 := arithEq_of_rows h (row := 1543) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_13
  simp only [← c94524, ← c94525, k2, ← c94526] at e_1543_13
  have hr := e_1543_13
  simp only [e_1543_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1543 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15115)) := by
  have c94527 := (publicBatchWrapper86_copies2953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94528 := (publicBatchWrapper86_copies2954 a h).1
  have c94529 := (publicBatchWrapper86_copies2954 a h).2.1
  have c94530 := (publicBatchWrapper86_copies2954 a h).2.2.1
  have c94531 := (publicBatchWrapper86_copies2954 a h).2.2.2.1
  have c94532 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1543_14 := arithEq_of_rows h (row := 1543) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_14
  simp only [← c94527, ← c94528, ← c94529] at e_1543_14
  have e_1543_15 := arithEq_of_rows h (row := 1543) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_15
  simp only [← c94530, ← c94531, k2, ← c94532] at e_1543_15
  have hr := e_1543_15
  simp only [e_1543_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1543 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15116)) := by
  have c94533 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.1
  have c94534 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.1
  have c94535 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.1
  have c94536 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.1
  have c94537 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.1
  have c94538 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1543_16 := arithEq_of_rows h (row := 1543) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_16
  simp only [← c94533, ← c94534, ← c94535] at e_1543_16
  have e_1543_17 := arithEq_of_rows h (row := 1543) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_17
  simp only [← c94536, ← c94537, k2, ← c94538] at e_1543_17
  have hr := e_1543_17
  simp only [e_1543_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1543 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15117)) := by
  have c94539 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94540 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94541 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94542 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94543 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94544 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1543_18 := arithEq_of_rows h (row := 1543) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_18
  simp only [← c94539, ← c94540, ← c94541] at e_1543_18
  have e_1543_19 := arithEq_of_rows h (row := 1543) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1543_19
  simp only [← c94542, ← c94543, k2, ← c94544] at e_1543_19
  have hr := e_1543_19
  simp only [e_1543_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1544 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15118)) := by
  have c94545 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94546 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94547 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94548 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94549 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94550 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1544_0 := arithEq_of_rows h (row := 1544) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_0
  simp only [← c94545, ← c94546, ← c94547] at e_1544_0
  have e_1544_1 := arithEq_of_rows h (row := 1544) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_1
  simp only [← c94548, ← c94549, k2, ← c94550] at e_1544_1
  have hr := e_1544_1
  simp only [e_1544_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1544 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15119)) := by
  have c94551 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94552 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94553 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94554 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94555 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94556 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1544_2 := arithEq_of_rows h (row := 1544) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_2
  simp only [← c94551, ← c94552, ← c94553] at e_1544_2
  have e_1544_3 := arithEq_of_rows h (row := 1544) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_3
  simp only [← c94554, ← c94555, k2, ← c94556] at e_1544_3
  have hr := e_1544_3
  simp only [e_1544_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1544 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15120)) := by
  have c94557 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94558 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94559 := (publicBatchWrapper86_copies2954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94560 := (publicBatchWrapper86_copies2955 a h).1
  have c94561 := (publicBatchWrapper86_copies2955 a h).2.1
  have c94562 := (publicBatchWrapper86_copies2955 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1544_4 := arithEq_of_rows h (row := 1544) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_4
  simp only [← c94557, ← c94558, ← c94559] at e_1544_4
  have e_1544_5 := arithEq_of_rows h (row := 1544) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_5
  simp only [← c94560, ← c94561, k2, ← c94562] at e_1544_5
  have hr := e_1544_5
  simp only [e_1544_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1544 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15121)) := by
  have c94563 := (publicBatchWrapper86_copies2955 a h).2.2.2.1
  have c94564 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.1
  have c94565 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.1
  have c94566 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.1
  have c94567 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.1
  have c94568 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1544_6 := arithEq_of_rows h (row := 1544) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_6
  simp only [← c94563, ← c94564, ← c94565] at e_1544_6
  have e_1544_7 := arithEq_of_rows h (row := 1544) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_7
  simp only [← c94566, ← c94567, k2, ← c94568] at e_1544_7
  have hr := e_1544_7
  simp only [e_1544_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1544 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15122)) := by
  have c94569 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.1
  have c94570 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94571 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94572 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94573 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94574 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1544_8 := arithEq_of_rows h (row := 1544) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_8
  simp only [← c94569, ← c94570, ← c94571] at e_1544_8
  have e_1544_9 := arithEq_of_rows h (row := 1544) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_9
  simp only [← c94572, ← c94573, k2, ← c94574] at e_1544_9
  have hr := e_1544_9
  simp only [e_1544_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1544 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15123)) := by
  have c94575 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94576 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94577 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94578 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94579 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94580 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1544_10 := arithEq_of_rows h (row := 1544) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_10
  simp only [← c94575, ← c94576, ← c94577] at e_1544_10
  have e_1544_11 := arithEq_of_rows h (row := 1544) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_11
  simp only [← c94578, ← c94579, k2, ← c94580] at e_1544_11
  have hr := e_1544_11
  simp only [e_1544_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1544 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15124)) := by
  have c94581 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94582 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94583 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94584 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94585 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94586 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1544_12 := arithEq_of_rows h (row := 1544) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_12
  simp only [← c94581, ← c94582, ← c94583] at e_1544_12
  have e_1544_13 := arithEq_of_rows h (row := 1544) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_13
  simp only [← c94584, ← c94585, k2, ← c94586] at e_1544_13
  have hr := e_1544_13
  simp only [e_1544_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1544 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15125)) := by
  have c94587 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94588 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94589 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94590 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94591 := (publicBatchWrapper86_copies2955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94592 := (publicBatchWrapper86_copies2956 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1544_14 := arithEq_of_rows h (row := 1544) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_14
  simp only [← c94587, ← c94588, ← c94589] at e_1544_14
  have e_1544_15 := arithEq_of_rows h (row := 1544) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_15
  simp only [← c94590, ← c94591, k2, ← c94592] at e_1544_15
  have hr := e_1544_15
  simp only [e_1544_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1544 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15126)) := by
  have c94593 := (publicBatchWrapper86_copies2956 a h).2.1
  have c94594 := (publicBatchWrapper86_copies2956 a h).2.2.1
  have c94595 := (publicBatchWrapper86_copies2956 a h).2.2.2.1
  have c94596 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.1
  have c94597 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.1
  have c94598 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1544_16 := arithEq_of_rows h (row := 1544) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_16
  simp only [← c94593, ← c94594, ← c94595] at e_1544_16
  have e_1544_17 := arithEq_of_rows h (row := 1544) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_17
  simp only [← c94596, ← c94597, k2, ← c94598] at e_1544_17
  have hr := e_1544_17
  simp only [e_1544_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1544 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15127)) := by
  have c94599 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.1
  have c94600 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.1
  have c94601 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.1
  have c94602 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94603 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94604 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1544_18 := arithEq_of_rows h (row := 1544) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_18
  simp only [← c94599, ← c94600, ← c94601] at e_1544_18
  have e_1544_19 := arithEq_of_rows h (row := 1544) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1544_19
  simp only [← c94602, ← c94603, k2, ← c94604] at e_1544_19
  have hr := e_1544_19
  simp only [e_1544_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1545 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15128)) := by
  have c94605 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94606 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94607 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94608 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94609 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94610 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1545_0 := arithEq_of_rows h (row := 1545) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_0
  simp only [← c94605, ← c94606, ← c94607] at e_1545_0
  have e_1545_1 := arithEq_of_rows h (row := 1545) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_1
  simp only [← c94608, ← c94609, k2, ← c94610] at e_1545_1
  have hr := e_1545_1
  simp only [e_1545_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1545 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15129)) := by
  have c94611 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94612 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94613 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94614 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94615 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94616 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1545_2 := arithEq_of_rows h (row := 1545) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_2
  simp only [← c94611, ← c94612, ← c94613] at e_1545_2
  have e_1545_3 := arithEq_of_rows h (row := 1545) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_3
  simp only [← c94614, ← c94615, k2, ← c94616] at e_1545_3
  have hr := e_1545_3
  simp only [e_1545_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1545 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15130)) := by
  have c94617 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94618 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94619 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94620 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94621 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94622 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1545_4 := arithEq_of_rows h (row := 1545) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_4
  simp only [← c94617, ← c94618, ← c94619] at e_1545_4
  have e_1545_5 := arithEq_of_rows h (row := 1545) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_5
  simp only [← c94620, ← c94621, k2, ← c94622] at e_1545_5
  have hr := e_1545_5
  simp only [e_1545_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1545 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15131)) := by
  have c94623 := (publicBatchWrapper86_copies2956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94624 := (publicBatchWrapper86_copies2957 a h).1
  have c94625 := (publicBatchWrapper86_copies2957 a h).2.1
  have c94626 := (publicBatchWrapper86_copies2957 a h).2.2.1
  have c94627 := (publicBatchWrapper86_copies2957 a h).2.2.2.1
  have c94628 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1545_6 := arithEq_of_rows h (row := 1545) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_6
  simp only [← c94623, ← c94624, ← c94625] at e_1545_6
  have e_1545_7 := arithEq_of_rows h (row := 1545) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_7
  simp only [← c94626, ← c94627, k2, ← c94628] at e_1545_7
  have hr := e_1545_7
  simp only [e_1545_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1545 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15132)) := by
  have c94629 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.1
  have c94630 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.1
  have c94631 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.1
  have c94632 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.1
  have c94633 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.1
  have c94634 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1545_8 := arithEq_of_rows h (row := 1545) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_8
  simp only [← c94629, ← c94630, ← c94631] at e_1545_8
  have e_1545_9 := arithEq_of_rows h (row := 1545) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_9
  simp only [← c94632, ← c94633, k2, ← c94634] at e_1545_9
  have hr := e_1545_9
  simp only [e_1545_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1545 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15133)) := by
  have c94635 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94636 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94637 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94638 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94639 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94640 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1545_10 := arithEq_of_rows h (row := 1545) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_10
  simp only [← c94635, ← c94636, ← c94637] at e_1545_10
  have e_1545_11 := arithEq_of_rows h (row := 1545) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_11
  simp only [← c94638, ← c94639, k2, ← c94640] at e_1545_11
  have hr := e_1545_11
  simp only [e_1545_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1545 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15134)) := by
  have c94641 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94642 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94643 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94644 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94645 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94646 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1545_12 := arithEq_of_rows h (row := 1545) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_12
  simp only [← c94641, ← c94642, ← c94643] at e_1545_12
  have e_1545_13 := arithEq_of_rows h (row := 1545) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_13
  simp only [← c94644, ← c94645, k2, ← c94646] at e_1545_13
  have hr := e_1545_13
  simp only [e_1545_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1545 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15135)) := by
  have c94647 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94648 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94649 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94650 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94651 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94652 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1545_14 := arithEq_of_rows h (row := 1545) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_14
  simp only [← c94647, ← c94648, ← c94649] at e_1545_14
  have e_1545_15 := arithEq_of_rows h (row := 1545) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_15
  simp only [← c94650, ← c94651, k2, ← c94652] at e_1545_15
  have hr := e_1545_15
  simp only [e_1545_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1545 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15136)) := by
  have c94653 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94654 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94655 := (publicBatchWrapper86_copies2957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94656 := (publicBatchWrapper86_copies2958 a h).1
  have c94657 := (publicBatchWrapper86_copies2958 a h).2.1
  have c94658 := (publicBatchWrapper86_copies2958 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1545_16 := arithEq_of_rows h (row := 1545) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_16
  simp only [← c94653, ← c94654, ← c94655] at e_1545_16
  have e_1545_17 := arithEq_of_rows h (row := 1545) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_17
  simp only [← c94656, ← c94657, k2, ← c94658] at e_1545_17
  have hr := e_1545_17
  simp only [e_1545_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1545 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15137)) := by
  have c94659 := (publicBatchWrapper86_copies2958 a h).2.2.2.1
  have c94660 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.1
  have c94661 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.1
  have c94662 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.1
  have c94663 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.1
  have c94664 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1545_18 := arithEq_of_rows h (row := 1545) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_18
  simp only [← c94659, ← c94660, ← c94661] at e_1545_18
  have e_1545_19 := arithEq_of_rows h (row := 1545) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1545_19
  simp only [← c94662, ← c94663, k2, ← c94664] at e_1545_19
  have hr := e_1545_19
  simp only [e_1545_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1546 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15138)) := by
  have c94665 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.1
  have c94666 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94667 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94668 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94669 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94670 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1546_0 := arithEq_of_rows h (row := 1546) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_0
  simp only [← c94665, ← c94666, ← c94667] at e_1546_0
  have e_1546_1 := arithEq_of_rows h (row := 1546) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_1
  simp only [← c94668, ← c94669, k2, ← c94670] at e_1546_1
  have hr := e_1546_1
  simp only [e_1546_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1546 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15139)) := by
  have c94671 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94672 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94673 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94674 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94675 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94676 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1546_2 := arithEq_of_rows h (row := 1546) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_2
  simp only [← c94671, ← c94672, ← c94673] at e_1546_2
  have e_1546_3 := arithEq_of_rows h (row := 1546) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_3
  simp only [← c94674, ← c94675, k2, ← c94676] at e_1546_3
  have hr := e_1546_3
  simp only [e_1546_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1546 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15140)) := by
  have c94677 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94678 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94679 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94680 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94681 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94682 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1546_4 := arithEq_of_rows h (row := 1546) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_4
  simp only [← c94677, ← c94678, ← c94679] at e_1546_4
  have e_1546_5 := arithEq_of_rows h (row := 1546) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_5
  simp only [← c94680, ← c94681, k2, ← c94682] at e_1546_5
  have hr := e_1546_5
  simp only [e_1546_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1546 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15141)) := by
  have c94683 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94684 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94685 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94686 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94687 := (publicBatchWrapper86_copies2958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94688 := (publicBatchWrapper86_copies2959 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1546_6 := arithEq_of_rows h (row := 1546) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_6
  simp only [← c94683, ← c94684, ← c94685] at e_1546_6
  have e_1546_7 := arithEq_of_rows h (row := 1546) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_7
  simp only [← c94686, ← c94687, k2, ← c94688] at e_1546_7
  have hr := e_1546_7
  simp only [e_1546_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1546 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15142)) := by
  have c94689 := (publicBatchWrapper86_copies2959 a h).2.1
  have c94690 := (publicBatchWrapper86_copies2959 a h).2.2.1
  have c94691 := (publicBatchWrapper86_copies2959 a h).2.2.2.1
  have c94692 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.1
  have c94693 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.1
  have c94694 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1546_8 := arithEq_of_rows h (row := 1546) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_8
  simp only [← c94689, ← c94690, ← c94691] at e_1546_8
  have e_1546_9 := arithEq_of_rows h (row := 1546) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_9
  simp only [← c94692, ← c94693, k2, ← c94694] at e_1546_9
  have hr := e_1546_9
  simp only [e_1546_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1546 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15143)) := by
  have c94695 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.1
  have c94696 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.1
  have c94697 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.1
  have c94698 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94699 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94700 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1546_10 := arithEq_of_rows h (row := 1546) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_10
  simp only [← c94695, ← c94696, ← c94697] at e_1546_10
  have e_1546_11 := arithEq_of_rows h (row := 1546) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_11
  simp only [← c94698, ← c94699, k2, ← c94700] at e_1546_11
  have hr := e_1546_11
  simp only [e_1546_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1546 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15144)) := by
  have c94701 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94702 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94703 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94704 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94705 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94706 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1546_12 := arithEq_of_rows h (row := 1546) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_12
  simp only [← c94701, ← c94702, ← c94703] at e_1546_12
  have e_1546_13 := arithEq_of_rows h (row := 1546) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_13
  simp only [← c94704, ← c94705, k2, ← c94706] at e_1546_13
  have hr := e_1546_13
  simp only [e_1546_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1546 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15145)) := by
  have c94707 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94708 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94709 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94710 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94711 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94712 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1546_14 := arithEq_of_rows h (row := 1546) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_14
  simp only [← c94707, ← c94708, ← c94709] at e_1546_14
  have e_1546_15 := arithEq_of_rows h (row := 1546) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_15
  simp only [← c94710, ← c94711, k2, ← c94712] at e_1546_15
  have hr := e_1546_15
  simp only [e_1546_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1546 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15146)) := by
  have c94713 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94714 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94715 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94716 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94717 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94718 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1546_16 := arithEq_of_rows h (row := 1546) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_16
  simp only [← c94713, ← c94714, ← c94715] at e_1546_16
  have e_1546_17 := arithEq_of_rows h (row := 1546) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_17
  simp only [← c94716, ← c94717, k2, ← c94718] at e_1546_17
  have hr := e_1546_17
  simp only [e_1546_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1546 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15147)) := by
  have c94719 := (publicBatchWrapper86_copies2959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94720 := (publicBatchWrapper86_copies2960 a h).1
  have c94721 := (publicBatchWrapper86_copies2960 a h).2.1
  have c94722 := (publicBatchWrapper86_copies2960 a h).2.2.1
  have c94723 := (publicBatchWrapper86_copies2960 a h).2.2.2.1
  have c94724 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1546_18 := arithEq_of_rows h (row := 1546) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_18
  simp only [← c94719, ← c94720, ← c94721] at e_1546_18
  have e_1546_19 := arithEq_of_rows h (row := 1546) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1546_19
  simp only [← c94722, ← c94723, k2, ← c94724] at e_1546_19
  have hr := e_1546_19
  simp only [e_1546_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1547 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15148)) := by
  have c94725 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.1
  have c94726 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.1
  have c94727 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.1
  have c94728 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.1
  have c94729 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.1
  have c94730 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1547_0 := arithEq_of_rows h (row := 1547) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_0
  simp only [← c94725, ← c94726, ← c94727] at e_1547_0
  have e_1547_1 := arithEq_of_rows h (row := 1547) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_1
  simp only [← c94728, ← c94729, k2, ← c94730] at e_1547_1
  have hr := e_1547_1
  simp only [e_1547_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1547 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15149)) := by
  have c94731 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94732 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94733 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94734 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94735 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94736 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1547_2 := arithEq_of_rows h (row := 1547) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_2
  simp only [← c94731, ← c94732, ← c94733] at e_1547_2
  have e_1547_3 := arithEq_of_rows h (row := 1547) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_3
  simp only [← c94734, ← c94735, k2, ← c94736] at e_1547_3
  have hr := e_1547_3
  simp only [e_1547_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1547 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15150)) := by
  have c94737 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94738 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94739 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94740 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94741 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94742 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1547_4 := arithEq_of_rows h (row := 1547) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_4
  simp only [← c94737, ← c94738, ← c94739] at e_1547_4
  have e_1547_5 := arithEq_of_rows h (row := 1547) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_5
  simp only [← c94740, ← c94741, k2, ← c94742] at e_1547_5
  have hr := e_1547_5
  simp only [e_1547_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1547 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15151)) := by
  have c94743 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94744 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94745 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94746 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94747 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94748 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1547_6 := arithEq_of_rows h (row := 1547) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_6
  simp only [← c94743, ← c94744, ← c94745] at e_1547_6
  have e_1547_7 := arithEq_of_rows h (row := 1547) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_7
  simp only [← c94746, ← c94747, k2, ← c94748] at e_1547_7
  have hr := e_1547_7
  simp only [e_1547_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1547 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15152)) := by
  have c94749 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94750 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94751 := (publicBatchWrapper86_copies2960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94752 := (publicBatchWrapper86_copies2961 a h).1
  have c94753 := (publicBatchWrapper86_copies2961 a h).2.1
  have c94754 := (publicBatchWrapper86_copies2961 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1547_8 := arithEq_of_rows h (row := 1547) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_8
  simp only [← c94749, ← c94750, ← c94751] at e_1547_8
  have e_1547_9 := arithEq_of_rows h (row := 1547) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_9
  simp only [← c94752, ← c94753, k2, ← c94754] at e_1547_9
  have hr := e_1547_9
  simp only [e_1547_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1547 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15153)) := by
  have c94755 := (publicBatchWrapper86_copies2961 a h).2.2.2.1
  have c94756 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.1
  have c94757 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.1
  have c94758 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.1
  have c94759 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.1
  have c94760 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1547_10 := arithEq_of_rows h (row := 1547) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_10
  simp only [← c94755, ← c94756, ← c94757] at e_1547_10
  have e_1547_11 := arithEq_of_rows h (row := 1547) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_11
  simp only [← c94758, ← c94759, k2, ← c94760] at e_1547_11
  have hr := e_1547_11
  simp only [e_1547_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1547 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15154)) := by
  have c94761 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.1
  have c94762 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94763 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94764 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94765 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94766 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1547_12 := arithEq_of_rows h (row := 1547) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_12
  simp only [← c94761, ← c94762, ← c94763] at e_1547_12
  have e_1547_13 := arithEq_of_rows h (row := 1547) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_13
  simp only [← c94764, ← c94765, k2, ← c94766] at e_1547_13
  have hr := e_1547_13
  simp only [e_1547_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1547 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15155)) := by
  have c94767 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94768 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94769 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94770 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94771 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94772 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1547_14 := arithEq_of_rows h (row := 1547) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_14
  simp only [← c94767, ← c94768, ← c94769] at e_1547_14
  have e_1547_15 := arithEq_of_rows h (row := 1547) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_15
  simp only [← c94770, ← c94771, k2, ← c94772] at e_1547_15
  have hr := e_1547_15
  simp only [e_1547_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1547 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15156)) := by
  have c94773 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94774 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94775 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94776 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94777 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94778 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1547_16 := arithEq_of_rows h (row := 1547) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_16
  simp only [← c94773, ← c94774, ← c94775] at e_1547_16
  have e_1547_17 := arithEq_of_rows h (row := 1547) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_17
  simp only [← c94776, ← c94777, k2, ← c94778] at e_1547_17
  have hr := e_1547_17
  simp only [e_1547_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1547 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15157)) := by
  have c94779 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94780 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94781 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94782 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94783 := (publicBatchWrapper86_copies2961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94784 := (publicBatchWrapper86_copies2962 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1547_18 := arithEq_of_rows h (row := 1547) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_18
  simp only [← c94779, ← c94780, ← c94781] at e_1547_18
  have e_1547_19 := arithEq_of_rows h (row := 1547) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1547_19
  simp only [← c94782, ← c94783, k2, ← c94784] at e_1547_19
  have hr := e_1547_19
  simp only [e_1547_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1548 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15158)) := by
  have c94785 := (publicBatchWrapper86_copies2962 a h).2.1
  have c94786 := (publicBatchWrapper86_copies2962 a h).2.2.1
  have c94787 := (publicBatchWrapper86_copies2962 a h).2.2.2.1
  have c94788 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.1
  have c94789 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.1
  have c94790 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1548_0 := arithEq_of_rows h (row := 1548) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_0
  simp only [← c94785, ← c94786, ← c94787] at e_1548_0
  have e_1548_1 := arithEq_of_rows h (row := 1548) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_1
  simp only [← c94788, ← c94789, k2, ← c94790] at e_1548_1
  have hr := e_1548_1
  simp only [e_1548_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1548 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15159)) := by
  have c94791 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.1
  have c94792 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.1
  have c94793 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.1
  have c94794 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94795 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94796 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1548_2 := arithEq_of_rows h (row := 1548) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_2
  simp only [← c94791, ← c94792, ← c94793] at e_1548_2
  have e_1548_3 := arithEq_of_rows h (row := 1548) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_3
  simp only [← c94794, ← c94795, k2, ← c94796] at e_1548_3
  have hr := e_1548_3
  simp only [e_1548_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1548 23) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15160)) := by
  have c94797 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94798 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94799 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94800 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94801 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94802 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1548_4 := arithEq_of_rows h (row := 1548) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_4
  simp only [← c94797, ← c94798, ← c94799] at e_1548_4
  have e_1548_5 := arithEq_of_rows h (row := 1548) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_5
  simp only [← c94800, ← c94801, k2, ← c94802] at e_1548_5
  have hr := e_1548_5
  simp only [e_1548_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1548 31) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15161)) := by
  have c94803 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94804 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94805 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94806 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94807 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94808 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1548_6 := arithEq_of_rows h (row := 1548) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_6
  simp only [← c94803, ← c94804, ← c94805] at e_1548_6
  have e_1548_7 := arithEq_of_rows h (row := 1548) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_7
  simp only [← c94806, ← c94807, k2, ← c94808] at e_1548_7
  have hr := e_1548_7
  simp only [e_1548_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1548 39) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15162)) := by
  have c94809 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94810 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94811 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94812 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94813 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94814 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1548_8 := arithEq_of_rows h (row := 1548) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_8
  simp only [← c94809, ← c94810, ← c94811] at e_1548_8
  have e_1548_9 := arithEq_of_rows h (row := 1548) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_9
  simp only [← c94812, ← c94813, k2, ← c94814] at e_1548_9
  have hr := e_1548_9
  simp only [e_1548_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1548 47) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15163)) := by
  have c94815 := (publicBatchWrapper86_copies2962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94816 := (publicBatchWrapper86_copies2963 a h).1
  have c94817 := (publicBatchWrapper86_copies2963 a h).2.1
  have c94818 := (publicBatchWrapper86_copies2963 a h).2.2.1
  have c94819 := (publicBatchWrapper86_copies2963 a h).2.2.2.1
  have c94820 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1548_10 := arithEq_of_rows h (row := 1548) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_10
  simp only [← c94815, ← c94816, ← c94817] at e_1548_10
  have e_1548_11 := arithEq_of_rows h (row := 1548) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_11
  simp only [← c94818, ← c94819, k2, ← c94820] at e_1548_11
  have hr := e_1548_11
  simp only [e_1548_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1548 55) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15164)) := by
  have c94821 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.1
  have c94822 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.1
  have c94823 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.1
  have c94824 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.1
  have c94825 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.1
  have c94826 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1548_12 := arithEq_of_rows h (row := 1548) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_12
  simp only [← c94821, ← c94822, ← c94823] at e_1548_12
  have e_1548_13 := arithEq_of_rows h (row := 1548) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_13
  simp only [← c94824, ← c94825, k2, ← c94826] at e_1548_13
  have hr := e_1548_13
  simp only [e_1548_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1548 63) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15165)) := by
  have c94827 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94828 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94829 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94830 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94831 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94832 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1548_14 := arithEq_of_rows h (row := 1548) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_14
  simp only [← c94827, ← c94828, ← c94829] at e_1548_14
  have e_1548_15 := arithEq_of_rows h (row := 1548) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_15
  simp only [← c94830, ← c94831, k2, ← c94832] at e_1548_15
  have hr := e_1548_15
  simp only [e_1548_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1548 71) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15166)) := by
  have c94833 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94834 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94835 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94836 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94837 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94838 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1548_16 := arithEq_of_rows h (row := 1548) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_16
  simp only [← c94833, ← c94834, ← c94835] at e_1548_16
  have e_1548_17 := arithEq_of_rows h (row := 1548) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_17
  simp only [← c94836, ← c94837, k2, ← c94838] at e_1548_17
  have hr := e_1548_17
  simp only [e_1548_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1548 79) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15167)) := by
  have c94839 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94840 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94841 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94842 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94843 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94844 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1548_18 := arithEq_of_rows h (row := 1548) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_18
  simp only [← c94839, ← c94840, ← c94841] at e_1548_18
  have e_1548_19 := arithEq_of_rows h (row := 1548) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1548_19
  simp only [← c94842, ← c94843, k2, ← c94844] at e_1548_19
  have hr := e_1548_19
  simp only [e_1548_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1549 7) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15168)) := by
  have c94845 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94846 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94847 := (publicBatchWrapper86_copies2963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94848 := (publicBatchWrapper86_copies2964 a h).1
  have c94849 := (publicBatchWrapper86_copies2964 a h).2.1
  have c94850 := (publicBatchWrapper86_copies2964 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1549_0 := arithEq_of_rows h (row := 1549) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_0
  simp only [← c94845, ← c94846, ← c94847] at e_1549_0
  have e_1549_1 := arithEq_of_rows h (row := 1549) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_1
  simp only [← c94848, ← c94849, k2, ← c94850] at e_1549_1
  have hr := e_1549_1
  simp only [e_1549_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1549 15) = bselect (a (.wire 73 47)) (a (.virt 200)) (a (.virt 15169)) := by
  have c94851 := (publicBatchWrapper86_copies2964 a h).2.2.2.1
  have c94852 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.1
  have c94853 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.1
  have c94854 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.1
  have c94855 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.1
  have c94856 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1549_2 := arithEq_of_rows h (row := 1549) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_2
  simp only [← c94851, ← c94852, ← c94853] at e_1549_2
  have e_1549_3 := arithEq_of_rows h (row := 1549) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_3
  simp only [← c94854, ← c94855, k2, ← c94856] at e_1549_3
  have hr := e_1549_3
  simp only [e_1549_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1549 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15210)) := by
  have c94857 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.1
  have c94858 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94859 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94860 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94861 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94862 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1549_4 := arithEq_of_rows h (row := 1549) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_4
  simp only [← c94857, ← c94858, ← c94859] at e_1549_4
  have e_1549_5 := arithEq_of_rows h (row := 1549) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_5
  simp only [← c94860, ← c94861, k2, ← c94862] at e_1549_5
  have hr := e_1549_5
  simp only [e_1549_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1549 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15211)) := by
  have c94863 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94864 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94865 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94866 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94867 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94868 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1549_6 := arithEq_of_rows h (row := 1549) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_6
  simp only [← c94863, ← c94864, ← c94865] at e_1549_6
  have e_1549_7 := arithEq_of_rows h (row := 1549) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_7
  simp only [← c94866, ← c94867, k2, ← c94868] at e_1549_7
  have hr := e_1549_7
  simp only [e_1549_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1549 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15212)) := by
  have c94869 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94870 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94871 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94872 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94873 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94874 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1549_8 := arithEq_of_rows h (row := 1549) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_8
  simp only [← c94869, ← c94870, ← c94871] at e_1549_8
  have e_1549_9 := arithEq_of_rows h (row := 1549) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_9
  simp only [← c94872, ← c94873, k2, ← c94874] at e_1549_9
  have hr := e_1549_9
  simp only [e_1549_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1549 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15213)) := by
  have c94875 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94876 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94877 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94878 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94879 := (publicBatchWrapper86_copies2964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94880 := (publicBatchWrapper86_copies2965 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1549_10 := arithEq_of_rows h (row := 1549) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_10
  simp only [← c94875, ← c94876, ← c94877] at e_1549_10
  have e_1549_11 := arithEq_of_rows h (row := 1549) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_11
  simp only [← c94878, ← c94879, k2, ← c94880] at e_1549_11
  have hr := e_1549_11
  simp only [e_1549_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1549 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15214)) := by
  have c94881 := (publicBatchWrapper86_copies2965 a h).2.1
  have c94882 := (publicBatchWrapper86_copies2965 a h).2.2.1
  have c94883 := (publicBatchWrapper86_copies2965 a h).2.2.2.1
  have c94884 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.1
  have c94885 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.1
  have c94886 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1549_12 := arithEq_of_rows h (row := 1549) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_12
  simp only [← c94881, ← c94882, ← c94883] at e_1549_12
  have e_1549_13 := arithEq_of_rows h (row := 1549) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_13
  simp only [← c94884, ← c94885, k2, ← c94886] at e_1549_13
  have hr := e_1549_13
  simp only [e_1549_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1549 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15215)) := by
  have c94887 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.1
  have c94888 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.1
  have c94889 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.1
  have c94890 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94891 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94892 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1549_14 := arithEq_of_rows h (row := 1549) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_14
  simp only [← c94887, ← c94888, ← c94889] at e_1549_14
  have e_1549_15 := arithEq_of_rows h (row := 1549) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_15
  simp only [← c94890, ← c94891, k2, ← c94892] at e_1549_15
  have hr := e_1549_15
  simp only [e_1549_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1549 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15216)) := by
  have c94893 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94894 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94895 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94896 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94897 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94898 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1549_16 := arithEq_of_rows h (row := 1549) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_16
  simp only [← c94893, ← c94894, ← c94895] at e_1549_16
  have e_1549_17 := arithEq_of_rows h (row := 1549) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_17
  simp only [← c94896, ← c94897, k2, ← c94898] at e_1549_17
  have hr := e_1549_17
  simp only [e_1549_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1549 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15217)) := by
  have c94899 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94900 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94901 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94902 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94903 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94904 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1549_18 := arithEq_of_rows h (row := 1549) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_18
  simp only [← c94899, ← c94900, ← c94901] at e_1549_18
  have e_1549_19 := arithEq_of_rows h (row := 1549) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1549_19
  simp only [← c94902, ← c94903, k2, ← c94904] at e_1549_19
  have hr := e_1549_19
  simp only [e_1549_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1550 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15218)) := by
  have c94905 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94906 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94907 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94908 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94909 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94910 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1550_0 := arithEq_of_rows h (row := 1550) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_0
  simp only [← c94905, ← c94906, ← c94907] at e_1550_0
  have e_1550_1 := arithEq_of_rows h (row := 1550) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_1
  simp only [← c94908, ← c94909, k2, ← c94910] at e_1550_1
  have hr := e_1550_1
  simp only [e_1550_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1550 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15219)) := by
  have c94911 := (publicBatchWrapper86_copies2965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94912 := (publicBatchWrapper86_copies2966 a h).1
  have c94913 := (publicBatchWrapper86_copies2966 a h).2.1
  have c94914 := (publicBatchWrapper86_copies2966 a h).2.2.1
  have c94915 := (publicBatchWrapper86_copies2966 a h).2.2.2.1
  have c94916 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1550_2 := arithEq_of_rows h (row := 1550) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_2
  simp only [← c94911, ← c94912, ← c94913] at e_1550_2
  have e_1550_3 := arithEq_of_rows h (row := 1550) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_3
  simp only [← c94914, ← c94915, k2, ← c94916] at e_1550_3
  have hr := e_1550_3
  simp only [e_1550_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1550 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15220)) := by
  have c94917 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.1
  have c94918 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.1
  have c94919 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.1
  have c94920 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.1
  have c94921 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.1
  have c94922 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1550_4 := arithEq_of_rows h (row := 1550) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_4
  simp only [← c94917, ← c94918, ← c94919] at e_1550_4
  have e_1550_5 := arithEq_of_rows h (row := 1550) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_5
  simp only [← c94920, ← c94921, k2, ← c94922] at e_1550_5
  have hr := e_1550_5
  simp only [e_1550_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1550 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15221)) := by
  have c94923 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94924 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94925 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94926 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94927 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94928 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1550_6 := arithEq_of_rows h (row := 1550) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_6
  simp only [← c94923, ← c94924, ← c94925] at e_1550_6
  have e_1550_7 := arithEq_of_rows h (row := 1550) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_7
  simp only [← c94926, ← c94927, k2, ← c94928] at e_1550_7
  have hr := e_1550_7
  simp only [e_1550_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1550 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15222)) := by
  have c94929 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94930 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94931 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94932 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94933 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94934 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1550_8 := arithEq_of_rows h (row := 1550) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_8
  simp only [← c94929, ← c94930, ← c94931] at e_1550_8
  have e_1550_9 := arithEq_of_rows h (row := 1550) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_9
  simp only [← c94932, ← c94933, k2, ← c94934] at e_1550_9
  have hr := e_1550_9
  simp only [e_1550_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1550 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15223)) := by
  have c94935 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94936 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94937 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94938 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94939 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94940 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1550_10 := arithEq_of_rows h (row := 1550) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_10
  simp only [← c94935, ← c94936, ← c94937] at e_1550_10
  have e_1550_11 := arithEq_of_rows h (row := 1550) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_11
  simp only [← c94938, ← c94939, k2, ← c94940] at e_1550_11
  have hr := e_1550_11
  simp only [e_1550_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1550 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15224)) := by
  have c94941 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94942 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94943 := (publicBatchWrapper86_copies2966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94944 := (publicBatchWrapper86_copies2967 a h).1
  have c94945 := (publicBatchWrapper86_copies2967 a h).2.1
  have c94946 := (publicBatchWrapper86_copies2967 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1550_12 := arithEq_of_rows h (row := 1550) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_12
  simp only [← c94941, ← c94942, ← c94943] at e_1550_12
  have e_1550_13 := arithEq_of_rows h (row := 1550) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_13
  simp only [← c94944, ← c94945, k2, ← c94946] at e_1550_13
  have hr := e_1550_13
  simp only [e_1550_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1550 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15225)) := by
  have c94947 := (publicBatchWrapper86_copies2967 a h).2.2.2.1
  have c94948 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.1
  have c94949 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.1
  have c94950 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.1
  have c94951 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.1
  have c94952 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1550_14 := arithEq_of_rows h (row := 1550) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_14
  simp only [← c94947, ← c94948, ← c94949] at e_1550_14
  have e_1550_15 := arithEq_of_rows h (row := 1550) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_15
  simp only [← c94950, ← c94951, k2, ← c94952] at e_1550_15
  have hr := e_1550_15
  simp only [e_1550_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1550 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15226)) := by
  have c94953 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.1
  have c94954 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94955 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94956 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94957 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94958 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1550_16 := arithEq_of_rows h (row := 1550) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_16
  simp only [← c94953, ← c94954, ← c94955] at e_1550_16
  have e_1550_17 := arithEq_of_rows h (row := 1550) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_17
  simp only [← c94956, ← c94957, k2, ← c94958] at e_1550_17
  have hr := e_1550_17
  simp only [e_1550_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1550 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15227)) := by
  have c94959 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94960 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94961 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94962 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94963 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94964 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1550_18 := arithEq_of_rows h (row := 1550) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_18
  simp only [← c94959, ← c94960, ← c94961] at e_1550_18
  have e_1550_19 := arithEq_of_rows h (row := 1550) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1550_19
  simp only [← c94962, ← c94963, k2, ← c94964] at e_1550_19
  have hr := e_1550_19
  simp only [e_1550_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1551 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15228)) := by
  have c94965 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94966 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94967 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94968 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94969 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94970 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1551_0 := arithEq_of_rows h (row := 1551) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_0
  simp only [← c94965, ← c94966, ← c94967] at e_1551_0
  have e_1551_1 := arithEq_of_rows h (row := 1551) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_1
  simp only [← c94968, ← c94969, k2, ← c94970] at e_1551_1
  have hr := e_1551_1
  simp only [e_1551_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1551 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15229)) := by
  have c94971 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94972 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94973 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94974 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94975 := (publicBatchWrapper86_copies2967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c94976 := (publicBatchWrapper86_copies2968 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1551_2 := arithEq_of_rows h (row := 1551) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_2
  simp only [← c94971, ← c94972, ← c94973] at e_1551_2
  have e_1551_3 := arithEq_of_rows h (row := 1551) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_3
  simp only [← c94974, ← c94975, k2, ← c94976] at e_1551_3
  have hr := e_1551_3
  simp only [e_1551_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1551 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15230)) := by
  have c94977 := (publicBatchWrapper86_copies2968 a h).2.1
  have c94978 := (publicBatchWrapper86_copies2968 a h).2.2.1
  have c94979 := (publicBatchWrapper86_copies2968 a h).2.2.2.1
  have c94980 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.1
  have c94981 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.1
  have c94982 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1551_4 := arithEq_of_rows h (row := 1551) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_4
  simp only [← c94977, ← c94978, ← c94979] at e_1551_4
  have e_1551_5 := arithEq_of_rows h (row := 1551) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_5
  simp only [← c94980, ← c94981, k2, ← c94982] at e_1551_5
  have hr := e_1551_5
  simp only [e_1551_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1551 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15231)) := by
  have c94983 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.1
  have c94984 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.1
  have c94985 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.1
  have c94986 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.1
  have c94987 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c94988 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1551_6 := arithEq_of_rows h (row := 1551) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_6
  simp only [← c94983, ← c94984, ← c94985] at e_1551_6
  have e_1551_7 := arithEq_of_rows h (row := 1551) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_7
  simp only [← c94986, ← c94987, k2, ← c94988] at e_1551_7
  have hr := e_1551_7
  simp only [e_1551_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1551 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15232)) := by
  have c94989 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94990 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94991 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94992 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94993 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94994 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1551_8 := arithEq_of_rows h (row := 1551) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_8
  simp only [← c94989, ← c94990, ← c94991] at e_1551_8
  have e_1551_9 := arithEq_of_rows h (row := 1551) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_9
  simp only [← c94992, ← c94993, k2, ← c94994] at e_1551_9
  have hr := e_1551_9
  simp only [e_1551_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1551 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15233)) := by
  have c94995 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94996 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94997 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94998 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c94999 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95000 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1551_10 := arithEq_of_rows h (row := 1551) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_10
  simp only [← c94995, ← c94996, ← c94997] at e_1551_10
  have e_1551_11 := arithEq_of_rows h (row := 1551) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_11
  simp only [← c94998, ← c94999, k2, ← c95000] at e_1551_11
  have hr := e_1551_11
  simp only [e_1551_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1551 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15234)) := by
  have c95001 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95002 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95003 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95004 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95005 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95006 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1551_12 := arithEq_of_rows h (row := 1551) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_12
  simp only [← c95001, ← c95002, ← c95003] at e_1551_12
  have e_1551_13 := arithEq_of_rows h (row := 1551) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_13
  simp only [← c95004, ← c95005, k2, ← c95006] at e_1551_13
  have hr := e_1551_13
  simp only [e_1551_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1551 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15235)) := by
  have c95007 := (publicBatchWrapper86_copies2968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95008 := (publicBatchWrapper86_copies2969 a h).1
  have c95009 := (publicBatchWrapper86_copies2969 a h).2.1
  have c95010 := (publicBatchWrapper86_copies2969 a h).2.2.1
  have c95011 := (publicBatchWrapper86_copies2969 a h).2.2.2.1
  have c95012 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1551_14 := arithEq_of_rows h (row := 1551) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_14
  simp only [← c95007, ← c95008, ← c95009] at e_1551_14
  have e_1551_15 := arithEq_of_rows h (row := 1551) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_15
  simp only [← c95010, ← c95011, k2, ← c95012] at e_1551_15
  have hr := e_1551_15
  simp only [e_1551_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1551 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15236)) := by
  have c95013 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.1
  have c95014 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.1
  have c95015 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.1
  have c95016 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.1
  have c95017 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.1
  have c95018 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1551_16 := arithEq_of_rows h (row := 1551) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_16
  simp only [← c95013, ← c95014, ← c95015] at e_1551_16
  have e_1551_17 := arithEq_of_rows h (row := 1551) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_17
  simp only [← c95016, ← c95017, k2, ← c95018] at e_1551_17
  have hr := e_1551_17
  simp only [e_1551_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1551 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15237)) := by
  have c95019 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95020 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95021 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95022 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95023 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95024 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1551_18 := arithEq_of_rows h (row := 1551) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_18
  simp only [← c95019, ← c95020, ← c95021] at e_1551_18
  have e_1551_19 := arithEq_of_rows h (row := 1551) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1551_19
  simp only [← c95022, ← c95023, k2, ← c95024] at e_1551_19
  have hr := e_1551_19
  simp only [e_1551_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1552 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15238)) := by
  have c95025 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95026 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95027 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95028 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95029 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95030 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1552_0 := arithEq_of_rows h (row := 1552) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_0
  simp only [← c95025, ← c95026, ← c95027] at e_1552_0
  have e_1552_1 := arithEq_of_rows h (row := 1552) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_1
  simp only [← c95028, ← c95029, k2, ← c95030] at e_1552_1
  have hr := e_1552_1
  simp only [e_1552_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1552 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15239)) := by
  have c95031 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95032 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95033 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95034 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95035 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95036 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1552_2 := arithEq_of_rows h (row := 1552) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_2
  simp only [← c95031, ← c95032, ← c95033] at e_1552_2
  have e_1552_3 := arithEq_of_rows h (row := 1552) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_3
  simp only [← c95034, ← c95035, k2, ← c95036] at e_1552_3
  have hr := e_1552_3
  simp only [e_1552_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1552 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15240)) := by
  have c95037 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95038 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95039 := (publicBatchWrapper86_copies2969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95040 := (publicBatchWrapper86_copies2970 a h).1
  have c95041 := (publicBatchWrapper86_copies2970 a h).2.1
  have c95042 := (publicBatchWrapper86_copies2970 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1552_4 := arithEq_of_rows h (row := 1552) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_4
  simp only [← c95037, ← c95038, ← c95039] at e_1552_4
  have e_1552_5 := arithEq_of_rows h (row := 1552) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_5
  simp only [← c95040, ← c95041, k2, ← c95042] at e_1552_5
  have hr := e_1552_5
  simp only [e_1552_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1552 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15241)) := by
  have c95043 := (publicBatchWrapper86_copies2970 a h).2.2.2.1
  have c95044 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.1
  have c95045 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.1
  have c95046 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.1
  have c95047 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.1
  have c95048 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1552_6 := arithEq_of_rows h (row := 1552) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_6
  simp only [← c95043, ← c95044, ← c95045] at e_1552_6
  have e_1552_7 := arithEq_of_rows h (row := 1552) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_7
  simp only [← c95046, ← c95047, k2, ← c95048] at e_1552_7
  have hr := e_1552_7
  simp only [e_1552_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1552 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15242)) := by
  have c95049 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.1
  have c95050 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95051 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95052 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95053 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95054 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1552_8 := arithEq_of_rows h (row := 1552) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_8
  simp only [← c95049, ← c95050, ← c95051] at e_1552_8
  have e_1552_9 := arithEq_of_rows h (row := 1552) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_9
  simp only [← c95052, ← c95053, k2, ← c95054] at e_1552_9
  have hr := e_1552_9
  simp only [e_1552_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1552 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15243)) := by
  have c95055 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95056 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95057 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95058 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95059 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95060 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1552_10 := arithEq_of_rows h (row := 1552) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_10
  simp only [← c95055, ← c95056, ← c95057] at e_1552_10
  have e_1552_11 := arithEq_of_rows h (row := 1552) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_11
  simp only [← c95058, ← c95059, k2, ← c95060] at e_1552_11
  have hr := e_1552_11
  simp only [e_1552_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1552 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15244)) := by
  have c95061 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95062 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95063 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95064 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95065 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95066 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1552_12 := arithEq_of_rows h (row := 1552) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_12
  simp only [← c95061, ← c95062, ← c95063] at e_1552_12
  have e_1552_13 := arithEq_of_rows h (row := 1552) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_13
  simp only [← c95064, ← c95065, k2, ← c95066] at e_1552_13
  have hr := e_1552_13
  simp only [e_1552_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1552 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15245)) := by
  have c95067 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95068 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95069 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95070 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95071 := (publicBatchWrapper86_copies2970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95072 := (publicBatchWrapper86_copies2971 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1552_14 := arithEq_of_rows h (row := 1552) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_14
  simp only [← c95067, ← c95068, ← c95069] at e_1552_14
  have e_1552_15 := arithEq_of_rows h (row := 1552) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_15
  simp only [← c95070, ← c95071, k2, ← c95072] at e_1552_15
  have hr := e_1552_15
  simp only [e_1552_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1552 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15246)) := by
  have c95073 := (publicBatchWrapper86_copies2971 a h).2.1
  have c95074 := (publicBatchWrapper86_copies2971 a h).2.2.1
  have c95075 := (publicBatchWrapper86_copies2971 a h).2.2.2.1
  have c95076 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.1
  have c95077 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.1
  have c95078 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1552_16 := arithEq_of_rows h (row := 1552) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_16
  simp only [← c95073, ← c95074, ← c95075] at e_1552_16
  have e_1552_17 := arithEq_of_rows h (row := 1552) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_17
  simp only [← c95076, ← c95077, k2, ← c95078] at e_1552_17
  have hr := e_1552_17
  simp only [e_1552_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1552 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15247)) := by
  have c95079 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.1
  have c95080 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.1
  have c95081 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.1
  have c95082 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95083 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95084 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1552_18 := arithEq_of_rows h (row := 1552) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_18
  simp only [← c95079, ← c95080, ← c95081] at e_1552_18
  have e_1552_19 := arithEq_of_rows h (row := 1552) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1552_19
  simp only [← c95082, ← c95083, k2, ← c95084] at e_1552_19
  have hr := e_1552_19
  simp only [e_1552_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1553 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15248)) := by
  have c95085 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95086 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95087 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95088 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95089 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95090 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1553_0 := arithEq_of_rows h (row := 1553) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_0
  simp only [← c95085, ← c95086, ← c95087] at e_1553_0
  have e_1553_1 := arithEq_of_rows h (row := 1553) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_1
  simp only [← c95088, ← c95089, k2, ← c95090] at e_1553_1
  have hr := e_1553_1
  simp only [e_1553_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1553 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15249)) := by
  have c95091 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95092 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95093 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95094 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95095 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95096 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1553_2 := arithEq_of_rows h (row := 1553) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_2
  simp only [← c95091, ← c95092, ← c95093] at e_1553_2
  have e_1553_3 := arithEq_of_rows h (row := 1553) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_3
  simp only [← c95094, ← c95095, k2, ← c95096] at e_1553_3
  have hr := e_1553_3
  simp only [e_1553_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1553 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15250)) := by
  have c95097 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95098 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95099 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95100 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95101 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95102 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1553_4 := arithEq_of_rows h (row := 1553) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_4
  simp only [← c95097, ← c95098, ← c95099] at e_1553_4
  have e_1553_5 := arithEq_of_rows h (row := 1553) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_5
  simp only [← c95100, ← c95101, k2, ← c95102] at e_1553_5
  have hr := e_1553_5
  simp only [e_1553_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1553 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15251)) := by
  have c95103 := (publicBatchWrapper86_copies2971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95104 := (publicBatchWrapper86_copies2972 a h).1
  have c95105 := (publicBatchWrapper86_copies2972 a h).2.1
  have c95106 := (publicBatchWrapper86_copies2972 a h).2.2.1
  have c95107 := (publicBatchWrapper86_copies2972 a h).2.2.2.1
  have c95108 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1553_6 := arithEq_of_rows h (row := 1553) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_6
  simp only [← c95103, ← c95104, ← c95105] at e_1553_6
  have e_1553_7 := arithEq_of_rows h (row := 1553) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_7
  simp only [← c95106, ← c95107, k2, ← c95108] at e_1553_7
  have hr := e_1553_7
  simp only [e_1553_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1553 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15252)) := by
  have c95109 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.1
  have c95110 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.1
  have c95111 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.1
  have c95112 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.1
  have c95113 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.1
  have c95114 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1553_8 := arithEq_of_rows h (row := 1553) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_8
  simp only [← c95109, ← c95110, ← c95111] at e_1553_8
  have e_1553_9 := arithEq_of_rows h (row := 1553) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_9
  simp only [← c95112, ← c95113, k2, ← c95114] at e_1553_9
  have hr := e_1553_9
  simp only [e_1553_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1553 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15253)) := by
  have c95115 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95116 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95117 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95118 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95119 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95120 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1553_10 := arithEq_of_rows h (row := 1553) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_10
  simp only [← c95115, ← c95116, ← c95117] at e_1553_10
  have e_1553_11 := arithEq_of_rows h (row := 1553) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_11
  simp only [← c95118, ← c95119, k2, ← c95120] at e_1553_11
  have hr := e_1553_11
  simp only [e_1553_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1553 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15254)) := by
  have c95121 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95122 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95123 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95124 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95125 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95126 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1553_12 := arithEq_of_rows h (row := 1553) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_12
  simp only [← c95121, ← c95122, ← c95123] at e_1553_12
  have e_1553_13 := arithEq_of_rows h (row := 1553) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_13
  simp only [← c95124, ← c95125, k2, ← c95126] at e_1553_13
  have hr := e_1553_13
  simp only [e_1553_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1553 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15255)) := by
  have c95127 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95128 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95129 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95130 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95131 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95132 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1553_14 := arithEq_of_rows h (row := 1553) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_14
  simp only [← c95127, ← c95128, ← c95129] at e_1553_14
  have e_1553_15 := arithEq_of_rows h (row := 1553) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_15
  simp only [← c95130, ← c95131, k2, ← c95132] at e_1553_15
  have hr := e_1553_15
  simp only [e_1553_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1553 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15256)) := by
  have c95133 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95134 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95135 := (publicBatchWrapper86_copies2972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95136 := (publicBatchWrapper86_copies2973 a h).1
  have c95137 := (publicBatchWrapper86_copies2973 a h).2.1
  have c95138 := (publicBatchWrapper86_copies2973 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1553_16 := arithEq_of_rows h (row := 1553) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_16
  simp only [← c95133, ← c95134, ← c95135] at e_1553_16
  have e_1553_17 := arithEq_of_rows h (row := 1553) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_17
  simp only [← c95136, ← c95137, k2, ← c95138] at e_1553_17
  have hr := e_1553_17
  simp only [e_1553_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1553 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15257)) := by
  have c95139 := (publicBatchWrapper86_copies2973 a h).2.2.2.1
  have c95140 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.1
  have c95141 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.1
  have c95142 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.1
  have c95143 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.1
  have c95144 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1553_18 := arithEq_of_rows h (row := 1553) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_18
  simp only [← c95139, ← c95140, ← c95141] at e_1553_18
  have e_1553_19 := arithEq_of_rows h (row := 1553) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1553_19
  simp only [← c95142, ← c95143, k2, ← c95144] at e_1553_19
  have hr := e_1553_19
  simp only [e_1553_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1554 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15258)) := by
  have c95145 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.1
  have c95146 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95147 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95148 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95149 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95150 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1554_0 := arithEq_of_rows h (row := 1554) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_0
  simp only [← c95145, ← c95146, ← c95147] at e_1554_0
  have e_1554_1 := arithEq_of_rows h (row := 1554) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_1
  simp only [← c95148, ← c95149, k2, ← c95150] at e_1554_1
  have hr := e_1554_1
  simp only [e_1554_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1554 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15259)) := by
  have c95151 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95152 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95153 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95154 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95155 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95156 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1554_2 := arithEq_of_rows h (row := 1554) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_2
  simp only [← c95151, ← c95152, ← c95153] at e_1554_2
  have e_1554_3 := arithEq_of_rows h (row := 1554) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_3
  simp only [← c95154, ← c95155, k2, ← c95156] at e_1554_3
  have hr := e_1554_3
  simp only [e_1554_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1554 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15260)) := by
  have c95157 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95158 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95159 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95160 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95161 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95162 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1554_4 := arithEq_of_rows h (row := 1554) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_4
  simp only [← c95157, ← c95158, ← c95159] at e_1554_4
  have e_1554_5 := arithEq_of_rows h (row := 1554) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_5
  simp only [← c95160, ← c95161, k2, ← c95162] at e_1554_5
  have hr := e_1554_5
  simp only [e_1554_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1554 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15261)) := by
  have c95163 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95164 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95165 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95166 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95167 := (publicBatchWrapper86_copies2973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95168 := (publicBatchWrapper86_copies2974 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1554_6 := arithEq_of_rows h (row := 1554) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_6
  simp only [← c95163, ← c95164, ← c95165] at e_1554_6
  have e_1554_7 := arithEq_of_rows h (row := 1554) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_7
  simp only [← c95166, ← c95167, k2, ← c95168] at e_1554_7
  have hr := e_1554_7
  simp only [e_1554_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1554 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15262)) := by
  have c95169 := (publicBatchWrapper86_copies2974 a h).2.1
  have c95170 := (publicBatchWrapper86_copies2974 a h).2.2.1
  have c95171 := (publicBatchWrapper86_copies2974 a h).2.2.2.1
  have c95172 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.1
  have c95173 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.1
  have c95174 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1554_8 := arithEq_of_rows h (row := 1554) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_8
  simp only [← c95169, ← c95170, ← c95171] at e_1554_8
  have e_1554_9 := arithEq_of_rows h (row := 1554) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_9
  simp only [← c95172, ← c95173, k2, ← c95174] at e_1554_9
  have hr := e_1554_9
  simp only [e_1554_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1554 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15263)) := by
  have c95175 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.1
  have c95176 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.1
  have c95177 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.1
  have c95178 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95179 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95180 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1554_10 := arithEq_of_rows h (row := 1554) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_10
  simp only [← c95175, ← c95176, ← c95177] at e_1554_10
  have e_1554_11 := arithEq_of_rows h (row := 1554) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_11
  simp only [← c95178, ← c95179, k2, ← c95180] at e_1554_11
  have hr := e_1554_11
  simp only [e_1554_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1554 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15264)) := by
  have c95181 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95182 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95183 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95184 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95185 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95186 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1554_12 := arithEq_of_rows h (row := 1554) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_12
  simp only [← c95181, ← c95182, ← c95183] at e_1554_12
  have e_1554_13 := arithEq_of_rows h (row := 1554) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_13
  simp only [← c95184, ← c95185, k2, ← c95186] at e_1554_13
  have hr := e_1554_13
  simp only [e_1554_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1554 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15265)) := by
  have c95187 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95188 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95189 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95190 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95191 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95192 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1554_14 := arithEq_of_rows h (row := 1554) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_14
  simp only [← c95187, ← c95188, ← c95189] at e_1554_14
  have e_1554_15 := arithEq_of_rows h (row := 1554) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_15
  simp only [← c95190, ← c95191, k2, ← c95192] at e_1554_15
  have hr := e_1554_15
  simp only [e_1554_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1554 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15266)) := by
  have c95193 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95194 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95195 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95196 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95197 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95198 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1554_16 := arithEq_of_rows h (row := 1554) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_16
  simp only [← c95193, ← c95194, ← c95195] at e_1554_16
  have e_1554_17 := arithEq_of_rows h (row := 1554) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_17
  simp only [← c95196, ← c95197, k2, ← c95198] at e_1554_17
  have hr := e_1554_17
  simp only [e_1554_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1554 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15267)) := by
  have c95199 := (publicBatchWrapper86_copies2974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95200 := (publicBatchWrapper86_copies2975 a h).1
  have c95201 := (publicBatchWrapper86_copies2975 a h).2.1
  have c95202 := (publicBatchWrapper86_copies2975 a h).2.2.1
  have c95203 := (publicBatchWrapper86_copies2975 a h).2.2.2.1
  have c95204 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1554_18 := arithEq_of_rows h (row := 1554) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_18
  simp only [← c95199, ← c95200, ← c95201] at e_1554_18
  have e_1554_19 := arithEq_of_rows h (row := 1554) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1554_19
  simp only [← c95202, ← c95203, k2, ← c95204] at e_1554_19
  have hr := e_1554_19
  simp only [e_1554_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1555 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15268)) := by
  have c95205 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.1
  have c95206 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.1
  have c95207 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.1
  have c95208 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.1
  have c95209 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.1
  have c95210 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1555_0 := arithEq_of_rows h (row := 1555) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_0
  simp only [← c95205, ← c95206, ← c95207] at e_1555_0
  have e_1555_1 := arithEq_of_rows h (row := 1555) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_1
  simp only [← c95208, ← c95209, k2, ← c95210] at e_1555_1
  have hr := e_1555_1
  simp only [e_1555_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1555 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15269)) := by
  have c95211 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95212 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95213 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95214 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95215 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95216 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1555_2 := arithEq_of_rows h (row := 1555) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_2
  simp only [← c95211, ← c95212, ← c95213] at e_1555_2
  have e_1555_3 := arithEq_of_rows h (row := 1555) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_3
  simp only [← c95214, ← c95215, k2, ← c95216] at e_1555_3
  have hr := e_1555_3
  simp only [e_1555_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1555 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15270)) := by
  have c95217 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95218 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95219 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95220 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95221 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95222 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1555_4 := arithEq_of_rows h (row := 1555) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_4
  simp only [← c95217, ← c95218, ← c95219] at e_1555_4
  have e_1555_5 := arithEq_of_rows h (row := 1555) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_5
  simp only [← c95220, ← c95221, k2, ← c95222] at e_1555_5
  have hr := e_1555_5
  simp only [e_1555_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1555 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15271)) := by
  have c95223 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95224 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95225 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95226 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95227 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95228 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1555_6 := arithEq_of_rows h (row := 1555) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_6
  simp only [← c95223, ← c95224, ← c95225] at e_1555_6
  have e_1555_7 := arithEq_of_rows h (row := 1555) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_7
  simp only [← c95226, ← c95227, k2, ← c95228] at e_1555_7
  have hr := e_1555_7
  simp only [e_1555_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1555 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15272)) := by
  have c95229 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95230 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95231 := (publicBatchWrapper86_copies2975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95232 := (publicBatchWrapper86_copies2976 a h).1
  have c95233 := (publicBatchWrapper86_copies2976 a h).2.1
  have c95234 := (publicBatchWrapper86_copies2976 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1555_8 := arithEq_of_rows h (row := 1555) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_8
  simp only [← c95229, ← c95230, ← c95231] at e_1555_8
  have e_1555_9 := arithEq_of_rows h (row := 1555) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_9
  simp only [← c95232, ← c95233, k2, ← c95234] at e_1555_9
  have hr := e_1555_9
  simp only [e_1555_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1555 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15273)) := by
  have c95235 := (publicBatchWrapper86_copies2976 a h).2.2.2.1
  have c95236 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.1
  have c95237 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.1
  have c95238 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.1
  have c95239 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.1
  have c95240 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1555_10 := arithEq_of_rows h (row := 1555) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_10
  simp only [← c95235, ← c95236, ← c95237] at e_1555_10
  have e_1555_11 := arithEq_of_rows h (row := 1555) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_11
  simp only [← c95238, ← c95239, k2, ← c95240] at e_1555_11
  have hr := e_1555_11
  simp only [e_1555_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1555 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15274)) := by
  have c95241 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.1
  have c95242 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95243 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95244 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95245 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95246 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1555_12 := arithEq_of_rows h (row := 1555) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_12
  simp only [← c95241, ← c95242, ← c95243] at e_1555_12
  have e_1555_13 := arithEq_of_rows h (row := 1555) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_13
  simp only [← c95244, ← c95245, k2, ← c95246] at e_1555_13
  have hr := e_1555_13
  simp only [e_1555_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1555 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15275)) := by
  have c95247 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95248 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95249 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95250 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95251 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95252 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1555_14 := arithEq_of_rows h (row := 1555) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_14
  simp only [← c95247, ← c95248, ← c95249] at e_1555_14
  have e_1555_15 := arithEq_of_rows h (row := 1555) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_15
  simp only [← c95250, ← c95251, k2, ← c95252] at e_1555_15
  have hr := e_1555_15
  simp only [e_1555_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1555 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15276)) := by
  have c95253 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95254 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95255 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95256 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95257 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95258 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1555_16 := arithEq_of_rows h (row := 1555) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_16
  simp only [← c95253, ← c95254, ← c95255] at e_1555_16
  have e_1555_17 := arithEq_of_rows h (row := 1555) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_17
  simp only [← c95256, ← c95257, k2, ← c95258] at e_1555_17
  have hr := e_1555_17
  simp only [e_1555_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1555 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15277)) := by
  have c95259 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95260 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95261 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95262 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95263 := (publicBatchWrapper86_copies2976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95264 := (publicBatchWrapper86_copies2977 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1555_18 := arithEq_of_rows h (row := 1555) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_18
  simp only [← c95259, ← c95260, ← c95261] at e_1555_18
  have e_1555_19 := arithEq_of_rows h (row := 1555) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1555_19
  simp only [← c95262, ← c95263, k2, ← c95264] at e_1555_19
  have hr := e_1555_19
  simp only [e_1555_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1556 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15278)) := by
  have c95265 := (publicBatchWrapper86_copies2977 a h).2.1
  have c95266 := (publicBatchWrapper86_copies2977 a h).2.2.1
  have c95267 := (publicBatchWrapper86_copies2977 a h).2.2.2.1
  have c95268 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.1
  have c95269 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.1
  have c95270 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1556_0 := arithEq_of_rows h (row := 1556) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_0
  simp only [← c95265, ← c95266, ← c95267] at e_1556_0
  have e_1556_1 := arithEq_of_rows h (row := 1556) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_1
  simp only [← c95268, ← c95269, k2, ← c95270] at e_1556_1
  have hr := e_1556_1
  simp only [e_1556_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1556 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15279)) := by
  have c95271 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.1
  have c95272 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.1
  have c95273 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.1
  have c95274 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95275 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95276 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1556_2 := arithEq_of_rows h (row := 1556) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_2
  simp only [← c95271, ← c95272, ← c95273] at e_1556_2
  have e_1556_3 := arithEq_of_rows h (row := 1556) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_3
  simp only [← c95274, ← c95275, k2, ← c95276] at e_1556_3
  have hr := e_1556_3
  simp only [e_1556_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1556 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15280)) := by
  have c95277 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95278 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95279 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95280 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95281 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95282 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1556_4 := arithEq_of_rows h (row := 1556) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_4
  simp only [← c95277, ← c95278, ← c95279] at e_1556_4
  have e_1556_5 := arithEq_of_rows h (row := 1556) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_5
  simp only [← c95280, ← c95281, k2, ← c95282] at e_1556_5
  have hr := e_1556_5
  simp only [e_1556_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1556 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15281)) := by
  have c95283 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95284 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95285 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95286 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95287 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95288 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1556_6 := arithEq_of_rows h (row := 1556) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_6
  simp only [← c95283, ← c95284, ← c95285] at e_1556_6
  have e_1556_7 := arithEq_of_rows h (row := 1556) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_7
  simp only [← c95286, ← c95287, k2, ← c95288] at e_1556_7
  have hr := e_1556_7
  simp only [e_1556_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1556 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15282)) := by
  have c95289 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95290 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95291 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95292 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95293 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95294 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1556_8 := arithEq_of_rows h (row := 1556) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_8
  simp only [← c95289, ← c95290, ← c95291] at e_1556_8
  have e_1556_9 := arithEq_of_rows h (row := 1556) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_9
  simp only [← c95292, ← c95293, k2, ← c95294] at e_1556_9
  have hr := e_1556_9
  simp only [e_1556_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1556 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15283)) := by
  have c95295 := (publicBatchWrapper86_copies2977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95296 := (publicBatchWrapper86_copies2978 a h).1
  have c95297 := (publicBatchWrapper86_copies2978 a h).2.1
  have c95298 := (publicBatchWrapper86_copies2978 a h).2.2.1
  have c95299 := (publicBatchWrapper86_copies2978 a h).2.2.2.1
  have c95300 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1556_10 := arithEq_of_rows h (row := 1556) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_10
  simp only [← c95295, ← c95296, ← c95297] at e_1556_10
  have e_1556_11 := arithEq_of_rows h (row := 1556) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_11
  simp only [← c95298, ← c95299, k2, ← c95300] at e_1556_11
  have hr := e_1556_11
  simp only [e_1556_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1556 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15284)) := by
  have c95301 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.1
  have c95302 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.1
  have c95303 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.1
  have c95304 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.1
  have c95305 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.1
  have c95306 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1556_12 := arithEq_of_rows h (row := 1556) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_12
  simp only [← c95301, ← c95302, ← c95303] at e_1556_12
  have e_1556_13 := arithEq_of_rows h (row := 1556) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_13
  simp only [← c95304, ← c95305, k2, ← c95306] at e_1556_13
  have hr := e_1556_13
  simp only [e_1556_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1556 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15285)) := by
  have c95307 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95308 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95309 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95310 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95311 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95312 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1556_14 := arithEq_of_rows h (row := 1556) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_14
  simp only [← c95307, ← c95308, ← c95309] at e_1556_14
  have e_1556_15 := arithEq_of_rows h (row := 1556) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_15
  simp only [← c95310, ← c95311, k2, ← c95312] at e_1556_15
  have hr := e_1556_15
  simp only [e_1556_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1556 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15286)) := by
  have c95313 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95314 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95315 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95316 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95317 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95318 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1556_16 := arithEq_of_rows h (row := 1556) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_16
  simp only [← c95313, ← c95314, ← c95315] at e_1556_16
  have e_1556_17 := arithEq_of_rows h (row := 1556) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_17
  simp only [← c95316, ← c95317, k2, ← c95318] at e_1556_17
  have hr := e_1556_17
  simp only [e_1556_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1556 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15287)) := by
  have c95319 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95320 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95321 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95322 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95323 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95324 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1556_18 := arithEq_of_rows h (row := 1556) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_18
  simp only [← c95319, ← c95320, ← c95321] at e_1556_18
  have e_1556_19 := arithEq_of_rows h (row := 1556) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1556_19
  simp only [← c95322, ← c95323, k2, ← c95324] at e_1556_19
  have hr := e_1556_19
  simp only [e_1556_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1557 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15288)) := by
  have c95325 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95326 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95327 := (publicBatchWrapper86_copies2978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95328 := (publicBatchWrapper86_copies2979 a h).1
  have c95329 := (publicBatchWrapper86_copies2979 a h).2.1
  have c95330 := (publicBatchWrapper86_copies2979 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1557_0 := arithEq_of_rows h (row := 1557) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_0
  simp only [← c95325, ← c95326, ← c95327] at e_1557_0
  have e_1557_1 := arithEq_of_rows h (row := 1557) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_1
  simp only [← c95328, ← c95329, k2, ← c95330] at e_1557_1
  have hr := e_1557_1
  simp only [e_1557_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1557 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15289)) := by
  have c95331 := (publicBatchWrapper86_copies2979 a h).2.2.2.1
  have c95332 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.1
  have c95333 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.1
  have c95334 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.1
  have c95335 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.1
  have c95336 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1557_2 := arithEq_of_rows h (row := 1557) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_2
  simp only [← c95331, ← c95332, ← c95333] at e_1557_2
  have e_1557_3 := arithEq_of_rows h (row := 1557) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_3
  simp only [← c95334, ← c95335, k2, ← c95336] at e_1557_3
  have hr := e_1557_3
  simp only [e_1557_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1557 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15290)) := by
  have c95337 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.1
  have c95338 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95339 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95340 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95341 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95342 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1557_4 := arithEq_of_rows h (row := 1557) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_4
  simp only [← c95337, ← c95338, ← c95339] at e_1557_4
  have e_1557_5 := arithEq_of_rows h (row := 1557) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_5
  simp only [← c95340, ← c95341, k2, ← c95342] at e_1557_5
  have hr := e_1557_5
  simp only [e_1557_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1557 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15291)) := by
  have c95343 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95344 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95345 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95346 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95347 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95348 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1557_6 := arithEq_of_rows h (row := 1557) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_6
  simp only [← c95343, ← c95344, ← c95345] at e_1557_6
  have e_1557_7 := arithEq_of_rows h (row := 1557) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_7
  simp only [← c95346, ← c95347, k2, ← c95348] at e_1557_7
  have hr := e_1557_7
  simp only [e_1557_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1557 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15292)) := by
  have c95349 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95350 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95351 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95352 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95353 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95354 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1557_8 := arithEq_of_rows h (row := 1557) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_8
  simp only [← c95349, ← c95350, ← c95351] at e_1557_8
  have e_1557_9 := arithEq_of_rows h (row := 1557) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_9
  simp only [← c95352, ← c95353, k2, ← c95354] at e_1557_9
  have hr := e_1557_9
  simp only [e_1557_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1557 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15293)) := by
  have c95355 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95356 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95357 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95358 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95359 := (publicBatchWrapper86_copies2979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95360 := (publicBatchWrapper86_copies2980 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1557_10 := arithEq_of_rows h (row := 1557) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_10
  simp only [← c95355, ← c95356, ← c95357] at e_1557_10
  have e_1557_11 := arithEq_of_rows h (row := 1557) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_11
  simp only [← c95358, ← c95359, k2, ← c95360] at e_1557_11
  have hr := e_1557_11
  simp only [e_1557_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1557 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15294)) := by
  have c95361 := (publicBatchWrapper86_copies2980 a h).2.1
  have c95362 := (publicBatchWrapper86_copies2980 a h).2.2.1
  have c95363 := (publicBatchWrapper86_copies2980 a h).2.2.2.1
  have c95364 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.1
  have c95365 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.1
  have c95366 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1557_12 := arithEq_of_rows h (row := 1557) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_12
  simp only [← c95361, ← c95362, ← c95363] at e_1557_12
  have e_1557_13 := arithEq_of_rows h (row := 1557) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_13
  simp only [← c95364, ← c95365, k2, ← c95366] at e_1557_13
  have hr := e_1557_13
  simp only [e_1557_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1557 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15295)) := by
  have c95367 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.1
  have c95368 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.1
  have c95369 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.1
  have c95370 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95371 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95372 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1557_14 := arithEq_of_rows h (row := 1557) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_14
  simp only [← c95367, ← c95368, ← c95369] at e_1557_14
  have e_1557_15 := arithEq_of_rows h (row := 1557) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_15
  simp only [← c95370, ← c95371, k2, ← c95372] at e_1557_15
  have hr := e_1557_15
  simp only [e_1557_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1557 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15296)) := by
  have c95373 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95374 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95375 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95376 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95377 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95378 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1557_16 := arithEq_of_rows h (row := 1557) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_16
  simp only [← c95373, ← c95374, ← c95375] at e_1557_16
  have e_1557_17 := arithEq_of_rows h (row := 1557) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_17
  simp only [← c95376, ← c95377, k2, ← c95378] at e_1557_17
  have hr := e_1557_17
  simp only [e_1557_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1557 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15297)) := by
  have c95379 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95380 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95381 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95382 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95383 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95384 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1557_18 := arithEq_of_rows h (row := 1557) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_18
  simp only [← c95379, ← c95380, ← c95381] at e_1557_18
  have e_1557_19 := arithEq_of_rows h (row := 1557) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1557_19
  simp only [← c95382, ← c95383, k2, ← c95384] at e_1557_19
  have hr := e_1557_19
  simp only [e_1557_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1558 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15298)) := by
  have c95385 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95386 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95387 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95388 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95389 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95390 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1558_0 := arithEq_of_rows h (row := 1558) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_0
  simp only [← c95385, ← c95386, ← c95387] at e_1558_0
  have e_1558_1 := arithEq_of_rows h (row := 1558) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_1
  simp only [← c95388, ← c95389, k2, ← c95390] at e_1558_1
  have hr := e_1558_1
  simp only [e_1558_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1558 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15299)) := by
  have c95391 := (publicBatchWrapper86_copies2980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95392 := (publicBatchWrapper86_copies2981 a h).1
  have c95393 := (publicBatchWrapper86_copies2981 a h).2.1
  have c95394 := (publicBatchWrapper86_copies2981 a h).2.2.1
  have c95395 := (publicBatchWrapper86_copies2981 a h).2.2.2.1
  have c95396 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1558_2 := arithEq_of_rows h (row := 1558) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_2
  simp only [← c95391, ← c95392, ← c95393] at e_1558_2
  have e_1558_3 := arithEq_of_rows h (row := 1558) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_3
  simp only [← c95394, ← c95395, k2, ← c95396] at e_1558_3
  have hr := e_1558_3
  simp only [e_1558_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1558 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15300)) := by
  have c95397 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.1
  have c95398 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.1
  have c95399 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.1
  have c95400 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.1
  have c95401 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.1
  have c95402 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1558_4 := arithEq_of_rows h (row := 1558) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_4
  simp only [← c95397, ← c95398, ← c95399] at e_1558_4
  have e_1558_5 := arithEq_of_rows h (row := 1558) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_5
  simp only [← c95400, ← c95401, k2, ← c95402] at e_1558_5
  have hr := e_1558_5
  simp only [e_1558_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1558 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15301)) := by
  have c95403 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95404 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95405 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95406 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95407 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95408 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1558_6 := arithEq_of_rows h (row := 1558) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_6
  simp only [← c95403, ← c95404, ← c95405] at e_1558_6
  have e_1558_7 := arithEq_of_rows h (row := 1558) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_7
  simp only [← c95406, ← c95407, k2, ← c95408] at e_1558_7
  have hr := e_1558_7
  simp only [e_1558_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1558 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15302)) := by
  have c95409 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95410 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95411 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95412 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95413 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95414 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1558_8 := arithEq_of_rows h (row := 1558) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_8
  simp only [← c95409, ← c95410, ← c95411] at e_1558_8
  have e_1558_9 := arithEq_of_rows h (row := 1558) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_9
  simp only [← c95412, ← c95413, k2, ← c95414] at e_1558_9
  have hr := e_1558_9
  simp only [e_1558_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1558 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15303)) := by
  have c95415 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95416 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95417 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95418 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95419 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95420 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1558_10 := arithEq_of_rows h (row := 1558) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_10
  simp only [← c95415, ← c95416, ← c95417] at e_1558_10
  have e_1558_11 := arithEq_of_rows h (row := 1558) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_11
  simp only [← c95418, ← c95419, k2, ← c95420] at e_1558_11
  have hr := e_1558_11
  simp only [e_1558_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1558 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15304)) := by
  have c95421 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95422 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95423 := (publicBatchWrapper86_copies2981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95424 := (publicBatchWrapper86_copies2982 a h).1
  have c95425 := (publicBatchWrapper86_copies2982 a h).2.1
  have c95426 := (publicBatchWrapper86_copies2982 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1558_12 := arithEq_of_rows h (row := 1558) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_12
  simp only [← c95421, ← c95422, ← c95423] at e_1558_12
  have e_1558_13 := arithEq_of_rows h (row := 1558) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_13
  simp only [← c95424, ← c95425, k2, ← c95426] at e_1558_13
  have hr := e_1558_13
  simp only [e_1558_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1558 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15305)) := by
  have c95427 := (publicBatchWrapper86_copies2982 a h).2.2.2.1
  have c95428 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.1
  have c95429 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.1
  have c95430 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.1
  have c95431 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.1
  have c95432 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1558_14 := arithEq_of_rows h (row := 1558) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_14
  simp only [← c95427, ← c95428, ← c95429] at e_1558_14
  have e_1558_15 := arithEq_of_rows h (row := 1558) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_15
  simp only [← c95430, ← c95431, k2, ← c95432] at e_1558_15
  have hr := e_1558_15
  simp only [e_1558_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1558 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15306)) := by
  have c95433 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.1
  have c95434 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95435 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95436 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95437 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95438 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1558_16 := arithEq_of_rows h (row := 1558) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_16
  simp only [← c95433, ← c95434, ← c95435] at e_1558_16
  have e_1558_17 := arithEq_of_rows h (row := 1558) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_17
  simp only [← c95436, ← c95437, k2, ← c95438] at e_1558_17
  have hr := e_1558_17
  simp only [e_1558_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1558 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15307)) := by
  have c95439 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95440 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95441 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95442 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95443 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95444 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1558_18 := arithEq_of_rows h (row := 1558) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_18
  simp only [← c95439, ← c95440, ← c95441] at e_1558_18
  have e_1558_19 := arithEq_of_rows h (row := 1558) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1558_19
  simp only [← c95442, ← c95443, k2, ← c95444] at e_1558_19
  have hr := e_1558_19
  simp only [e_1558_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15308)) := by
  have c95445 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95446 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95447 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95448 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95449 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95450 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_0 := arithEq_of_rows h (row := 1559) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_0
  simp only [← c95445, ← c95446, ← c95447] at e_1559_0
  have e_1559_1 := arithEq_of_rows h (row := 1559) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_1
  simp only [← c95448, ← c95449, k2, ← c95450] at e_1559_1
  have hr := e_1559_1
  simp only [e_1559_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15309)) := by
  have c95451 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95452 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95453 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95454 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95455 := (publicBatchWrapper86_copies2982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95456 := (publicBatchWrapper86_copies2983 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_2 := arithEq_of_rows h (row := 1559) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_2
  simp only [← c95451, ← c95452, ← c95453] at e_1559_2
  have e_1559_3 := arithEq_of_rows h (row := 1559) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_3
  simp only [← c95454, ← c95455, k2, ← c95456] at e_1559_3
  have hr := e_1559_3
  simp only [e_1559_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15310)) := by
  have c95457 := (publicBatchWrapper86_copies2983 a h).2.1
  have c95458 := (publicBatchWrapper86_copies2983 a h).2.2.1
  have c95459 := (publicBatchWrapper86_copies2983 a h).2.2.2.1
  have c95460 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.1
  have c95461 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.1
  have c95462 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_4 := arithEq_of_rows h (row := 1559) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_4
  simp only [← c95457, ← c95458, ← c95459] at e_1559_4
  have e_1559_5 := arithEq_of_rows h (row := 1559) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_5
  simp only [← c95460, ← c95461, k2, ← c95462] at e_1559_5
  have hr := e_1559_5
  simp only [e_1559_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15311)) := by
  have c95463 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.1
  have c95464 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.1
  have c95465 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.1
  have c95466 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95467 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95468 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_6 := arithEq_of_rows h (row := 1559) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_6
  simp only [← c95463, ← c95464, ← c95465] at e_1559_6
  have e_1559_7 := arithEq_of_rows h (row := 1559) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_7
  simp only [← c95466, ← c95467, k2, ← c95468] at e_1559_7
  have hr := e_1559_7
  simp only [e_1559_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15312)) := by
  have c95469 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95470 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95471 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95472 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95473 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95474 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_8 := arithEq_of_rows h (row := 1559) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_8
  simp only [← c95469, ← c95470, ← c95471] at e_1559_8
  have e_1559_9 := arithEq_of_rows h (row := 1559) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_9
  simp only [← c95472, ← c95473, k2, ← c95474] at e_1559_9
  have hr := e_1559_9
  simp only [e_1559_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15313)) := by
  have c95475 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95476 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95477 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95478 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95479 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95480 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_10 := arithEq_of_rows h (row := 1559) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_10
  simp only [← c95475, ← c95476, ← c95477] at e_1559_10
  have e_1559_11 := arithEq_of_rows h (row := 1559) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_11
  simp only [← c95478, ← c95479, k2, ← c95480] at e_1559_11
  have hr := e_1559_11
  simp only [e_1559_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15314)) := by
  have c95481 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95482 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95483 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95484 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95485 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95486 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_12 := arithEq_of_rows h (row := 1559) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_12
  simp only [← c95481, ← c95482, ← c95483] at e_1559_12
  have e_1559_13 := arithEq_of_rows h (row := 1559) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_13
  simp only [← c95484, ← c95485, k2, ← c95486] at e_1559_13
  have hr := e_1559_13
  simp only [e_1559_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15315)) := by
  have c95487 := (publicBatchWrapper86_copies2983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95488 := (publicBatchWrapper86_copies2984 a h).1
  have c95489 := (publicBatchWrapper86_copies2984 a h).2.1
  have c95490 := (publicBatchWrapper86_copies2984 a h).2.2.1
  have c95491 := (publicBatchWrapper86_copies2984 a h).2.2.2.1
  have c95492 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_14 := arithEq_of_rows h (row := 1559) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_14
  simp only [← c95487, ← c95488, ← c95489] at e_1559_14
  have e_1559_15 := arithEq_of_rows h (row := 1559) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_15
  simp only [← c95490, ← c95491, k2, ← c95492] at e_1559_15
  have hr := e_1559_15
  simp only [e_1559_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15316)) := by
  have c95493 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.1
  have c95494 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.1
  have c95495 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.1
  have c95496 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.1
  have c95497 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.1
  have c95498 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_16 := arithEq_of_rows h (row := 1559) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_16
  simp only [← c95493, ← c95494, ← c95495] at e_1559_16
  have e_1559_17 := arithEq_of_rows h (row := 1559) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_17
  simp only [← c95496, ← c95497, k2, ← c95498] at e_1559_17
  have hr := e_1559_17
  simp only [e_1559_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1559 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15317)) := by
  have c95499 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95500 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95501 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95502 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95503 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95504 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1559_18 := arithEq_of_rows h (row := 1559) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_18
  simp only [← c95499, ← c95500, ← c95501] at e_1559_18
  have e_1559_19 := arithEq_of_rows h (row := 1559) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_19
  simp only [← c95502, ← c95503, k2, ← c95504] at e_1559_19
  have hr := e_1559_19
  simp only [e_1559_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15318)) := by
  have c95505 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95506 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95507 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95508 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95509 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95510 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_0 := arithEq_of_rows h (row := 1560) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_0
  simp only [← c95505, ← c95506, ← c95507] at e_1560_0
  have e_1560_1 := arithEq_of_rows h (row := 1560) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_1
  simp only [← c95508, ← c95509, k2, ← c95510] at e_1560_1
  have hr := e_1560_1
  simp only [e_1560_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15319)) := by
  have c95511 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95512 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95513 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95514 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95515 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95516 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_2 := arithEq_of_rows h (row := 1560) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_2
  simp only [← c95511, ← c95512, ← c95513] at e_1560_2
  have e_1560_3 := arithEq_of_rows h (row := 1560) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_3
  simp only [← c95514, ← c95515, k2, ← c95516] at e_1560_3
  have hr := e_1560_3
  simp only [e_1560_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15320)) := by
  have c95517 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95518 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95519 := (publicBatchWrapper86_copies2984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95520 := (publicBatchWrapper86_copies2985 a h).1
  have c95521 := (publicBatchWrapper86_copies2985 a h).2.1
  have c95522 := (publicBatchWrapper86_copies2985 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_4 := arithEq_of_rows h (row := 1560) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_4
  simp only [← c95517, ← c95518, ← c95519] at e_1560_4
  have e_1560_5 := arithEq_of_rows h (row := 1560) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_5
  simp only [← c95520, ← c95521, k2, ← c95522] at e_1560_5
  have hr := e_1560_5
  simp only [e_1560_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15321)) := by
  have c95523 := (publicBatchWrapper86_copies2985 a h).2.2.2.1
  have c95524 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.1
  have c95525 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.1
  have c95526 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.1
  have c95527 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.1
  have c95528 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_6 := arithEq_of_rows h (row := 1560) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_6
  simp only [← c95523, ← c95524, ← c95525] at e_1560_6
  have e_1560_7 := arithEq_of_rows h (row := 1560) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_7
  simp only [← c95526, ← c95527, k2, ← c95528] at e_1560_7
  have hr := e_1560_7
  simp only [e_1560_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15322)) := by
  have c95529 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.1
  have c95530 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95531 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95532 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95533 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95534 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_8 := arithEq_of_rows h (row := 1560) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_8
  simp only [← c95529, ← c95530, ← c95531] at e_1560_8
  have e_1560_9 := arithEq_of_rows h (row := 1560) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_9
  simp only [← c95532, ← c95533, k2, ← c95534] at e_1560_9
  have hr := e_1560_9
  simp only [e_1560_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15323)) := by
  have c95535 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95536 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95537 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95538 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95539 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95540 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_10 := arithEq_of_rows h (row := 1560) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_10
  simp only [← c95535, ← c95536, ← c95537] at e_1560_10
  have e_1560_11 := arithEq_of_rows h (row := 1560) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_11
  simp only [← c95538, ← c95539, k2, ← c95540] at e_1560_11
  have hr := e_1560_11
  simp only [e_1560_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15324)) := by
  have c95541 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95542 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95543 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95544 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95545 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95546 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_12 := arithEq_of_rows h (row := 1560) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_12
  simp only [← c95541, ← c95542, ← c95543] at e_1560_12
  have e_1560_13 := arithEq_of_rows h (row := 1560) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_13
  simp only [← c95544, ← c95545, k2, ← c95546] at e_1560_13
  have hr := e_1560_13
  simp only [e_1560_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15325)) := by
  have c95547 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95548 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95549 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95550 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95551 := (publicBatchWrapper86_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95552 := (publicBatchWrapper86_copies2986 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_14 := arithEq_of_rows h (row := 1560) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_14
  simp only [← c95547, ← c95548, ← c95549] at e_1560_14
  have e_1560_15 := arithEq_of_rows h (row := 1560) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_15
  simp only [← c95550, ← c95551, k2, ← c95552] at e_1560_15
  have hr := e_1560_15
  simp only [e_1560_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15326)) := by
  have c95553 := (publicBatchWrapper86_copies2986 a h).2.1
  have c95554 := (publicBatchWrapper86_copies2986 a h).2.2.1
  have c95555 := (publicBatchWrapper86_copies2986 a h).2.2.2.1
  have c95556 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.1
  have c95557 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.1
  have c95558 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_16 := arithEq_of_rows h (row := 1560) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_16
  simp only [← c95553, ← c95554, ← c95555] at e_1560_16
  have e_1560_17 := arithEq_of_rows h (row := 1560) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_17
  simp only [← c95556, ← c95557, k2, ← c95558] at e_1560_17
  have hr := e_1560_17
  simp only [e_1560_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1560 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15327)) := by
  have c95559 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.1
  have c95560 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.1
  have c95561 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.1
  have c95562 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95563 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95564 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1560_18 := arithEq_of_rows h (row := 1560) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_18
  simp only [← c95559, ← c95560, ← c95561] at e_1560_18
  have e_1560_19 := arithEq_of_rows h (row := 1560) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_19
  simp only [← c95562, ← c95563, k2, ← c95564] at e_1560_19
  have hr := e_1560_19
  simp only [e_1560_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15328)) := by
  have c95565 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95566 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95567 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95568 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95569 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95570 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_0 := arithEq_of_rows h (row := 1561) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_0
  simp only [← c95565, ← c95566, ← c95567] at e_1561_0
  have e_1561_1 := arithEq_of_rows h (row := 1561) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_1
  simp only [← c95568, ← c95569, k2, ← c95570] at e_1561_1
  have hr := e_1561_1
  simp only [e_1561_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15329)) := by
  have c95571 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95572 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95573 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95574 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95575 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95576 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_2 := arithEq_of_rows h (row := 1561) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_2
  simp only [← c95571, ← c95572, ← c95573] at e_1561_2
  have e_1561_3 := arithEq_of_rows h (row := 1561) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_3
  simp only [← c95574, ← c95575, k2, ← c95576] at e_1561_3
  have hr := e_1561_3
  simp only [e_1561_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15330)) := by
  have c95577 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95578 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95579 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95580 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95581 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95582 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_4 := arithEq_of_rows h (row := 1561) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_4
  simp only [← c95577, ← c95578, ← c95579] at e_1561_4
  have e_1561_5 := arithEq_of_rows h (row := 1561) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_5
  simp only [← c95580, ← c95581, k2, ← c95582] at e_1561_5
  have hr := e_1561_5
  simp only [e_1561_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15331)) := by
  have c95583 := (publicBatchWrapper86_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95584 := (publicBatchWrapper86_copies2987 a h).1
  have c95585 := (publicBatchWrapper86_copies2987 a h).2.1
  have c95586 := (publicBatchWrapper86_copies2987 a h).2.2.1
  have c95587 := (publicBatchWrapper86_copies2987 a h).2.2.2.1
  have c95588 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_6 := arithEq_of_rows h (row := 1561) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_6
  simp only [← c95583, ← c95584, ← c95585] at e_1561_6
  have e_1561_7 := arithEq_of_rows h (row := 1561) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_7
  simp only [← c95586, ← c95587, k2, ← c95588] at e_1561_7
  have hr := e_1561_7
  simp only [e_1561_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15332)) := by
  have c95589 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.1
  have c95590 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.1
  have c95591 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.1
  have c95592 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.1
  have c95593 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.1
  have c95594 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_8 := arithEq_of_rows h (row := 1561) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_8
  simp only [← c95589, ← c95590, ← c95591] at e_1561_8
  have e_1561_9 := arithEq_of_rows h (row := 1561) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_9
  simp only [← c95592, ← c95593, k2, ← c95594] at e_1561_9
  have hr := e_1561_9
  simp only [e_1561_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15333)) := by
  have c95595 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95596 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95597 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95598 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95599 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95600 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_10 := arithEq_of_rows h (row := 1561) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_10
  simp only [← c95595, ← c95596, ← c95597] at e_1561_10
  have e_1561_11 := arithEq_of_rows h (row := 1561) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_11
  simp only [← c95598, ← c95599, k2, ← c95600] at e_1561_11
  have hr := e_1561_11
  simp only [e_1561_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15334)) := by
  have c95601 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95602 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95603 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95604 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95605 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95606 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_12 := arithEq_of_rows h (row := 1561) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_12
  simp only [← c95601, ← c95602, ← c95603] at e_1561_12
  have e_1561_13 := arithEq_of_rows h (row := 1561) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_13
  simp only [← c95604, ← c95605, k2, ← c95606] at e_1561_13
  have hr := e_1561_13
  simp only [e_1561_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15335)) := by
  have c95607 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95608 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95609 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95610 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95611 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95612 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_14 := arithEq_of_rows h (row := 1561) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_14
  simp only [← c95607, ← c95608, ← c95609] at e_1561_14
  have e_1561_15 := arithEq_of_rows h (row := 1561) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_15
  simp only [← c95610, ← c95611, k2, ← c95612] at e_1561_15
  have hr := e_1561_15
  simp only [e_1561_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15336)) := by
  have c95613 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95614 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95615 := (publicBatchWrapper86_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95616 := (publicBatchWrapper86_copies2988 a h).1
  have c95617 := (publicBatchWrapper86_copies2988 a h).2.1
  have c95618 := (publicBatchWrapper86_copies2988 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_16 := arithEq_of_rows h (row := 1561) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_16
  simp only [← c95613, ← c95614, ← c95615] at e_1561_16
  have e_1561_17 := arithEq_of_rows h (row := 1561) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_17
  simp only [← c95616, ← c95617, k2, ← c95618] at e_1561_17
  have hr := e_1561_17
  simp only [e_1561_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1561 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15337)) := by
  have c95619 := (publicBatchWrapper86_copies2988 a h).2.2.2.1
  have c95620 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.1
  have c95621 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.1
  have c95622 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.1
  have c95623 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.1
  have c95624 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1561_18 := arithEq_of_rows h (row := 1561) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_18
  simp only [← c95619, ← c95620, ← c95621] at e_1561_18
  have e_1561_19 := arithEq_of_rows h (row := 1561) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_19
  simp only [← c95622, ← c95623, k2, ← c95624] at e_1561_19
  have hr := e_1561_19
  simp only [e_1561_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15338)) := by
  have c95625 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.1
  have c95626 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95627 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95628 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95629 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95630 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_0 := arithEq_of_rows h (row := 1562) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_0
  simp only [← c95625, ← c95626, ← c95627] at e_1562_0
  have e_1562_1 := arithEq_of_rows h (row := 1562) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_1
  simp only [← c95628, ← c95629, k2, ← c95630] at e_1562_1
  have hr := e_1562_1
  simp only [e_1562_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15339)) := by
  have c95631 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95632 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95633 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95634 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95635 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95636 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_2 := arithEq_of_rows h (row := 1562) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_2
  simp only [← c95631, ← c95632, ← c95633] at e_1562_2
  have e_1562_3 := arithEq_of_rows h (row := 1562) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_3
  simp only [← c95634, ← c95635, k2, ← c95636] at e_1562_3
  have hr := e_1562_3
  simp only [e_1562_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15340)) := by
  have c95637 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95638 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95639 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95640 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95641 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95642 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_4 := arithEq_of_rows h (row := 1562) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_4
  simp only [← c95637, ← c95638, ← c95639] at e_1562_4
  have e_1562_5 := arithEq_of_rows h (row := 1562) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_5
  simp only [← c95640, ← c95641, k2, ← c95642] at e_1562_5
  have hr := e_1562_5
  simp only [e_1562_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15341)) := by
  have c95643 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95644 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95645 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95646 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95647 := (publicBatchWrapper86_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95648 := (publicBatchWrapper86_copies2989 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_6 := arithEq_of_rows h (row := 1562) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_6
  simp only [← c95643, ← c95644, ← c95645] at e_1562_6
  have e_1562_7 := arithEq_of_rows h (row := 1562) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_7
  simp only [← c95646, ← c95647, k2, ← c95648] at e_1562_7
  have hr := e_1562_7
  simp only [e_1562_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15342)) := by
  have c95649 := (publicBatchWrapper86_copies2989 a h).2.1
  have c95650 := (publicBatchWrapper86_copies2989 a h).2.2.1
  have c95651 := (publicBatchWrapper86_copies2989 a h).2.2.2.1
  have c95652 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.1
  have c95653 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.1
  have c95654 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_8 := arithEq_of_rows h (row := 1562) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_8
  simp only [← c95649, ← c95650, ← c95651] at e_1562_8
  have e_1562_9 := arithEq_of_rows h (row := 1562) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_9
  simp only [← c95652, ← c95653, k2, ← c95654] at e_1562_9
  have hr := e_1562_9
  simp only [e_1562_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15343)) := by
  have c95655 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.1
  have c95656 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.1
  have c95657 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.1
  have c95658 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95659 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95660 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_10 := arithEq_of_rows h (row := 1562) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_10
  simp only [← c95655, ← c95656, ← c95657] at e_1562_10
  have e_1562_11 := arithEq_of_rows h (row := 1562) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_11
  simp only [← c95658, ← c95659, k2, ← c95660] at e_1562_11
  have hr := e_1562_11
  simp only [e_1562_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15344)) := by
  have c95661 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95662 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95663 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95664 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95665 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95666 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_12 := arithEq_of_rows h (row := 1562) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_12
  simp only [← c95661, ← c95662, ← c95663] at e_1562_12
  have e_1562_13 := arithEq_of_rows h (row := 1562) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_13
  simp only [← c95664, ← c95665, k2, ← c95666] at e_1562_13
  have hr := e_1562_13
  simp only [e_1562_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15345)) := by
  have c95667 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95668 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95669 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95670 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95671 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95672 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_14 := arithEq_of_rows h (row := 1562) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_14
  simp only [← c95667, ← c95668, ← c95669] at e_1562_14
  have e_1562_15 := arithEq_of_rows h (row := 1562) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_15
  simp only [← c95670, ← c95671, k2, ← c95672] at e_1562_15
  have hr := e_1562_15
  simp only [e_1562_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15346)) := by
  have c95673 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95674 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95675 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95676 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95677 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95678 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_16 := arithEq_of_rows h (row := 1562) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_16
  simp only [← c95673, ← c95674, ← c95675] at e_1562_16
  have e_1562_17 := arithEq_of_rows h (row := 1562) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_17
  simp only [← c95676, ← c95677, k2, ← c95678] at e_1562_17
  have hr := e_1562_17
  simp only [e_1562_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1562 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15347)) := by
  have c95679 := (publicBatchWrapper86_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95680 := (publicBatchWrapper86_copies2990 a h).1
  have c95681 := (publicBatchWrapper86_copies2990 a h).2.1
  have c95682 := (publicBatchWrapper86_copies2990 a h).2.2.1
  have c95683 := (publicBatchWrapper86_copies2990 a h).2.2.2.1
  have c95684 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1562_18 := arithEq_of_rows h (row := 1562) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_18
  simp only [← c95679, ← c95680, ← c95681] at e_1562_18
  have e_1562_19 := arithEq_of_rows h (row := 1562) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_19
  simp only [← c95682, ← c95683, k2, ← c95684] at e_1562_19
  have hr := e_1562_19
  simp only [e_1562_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15348)) := by
  have c95685 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.1
  have c95686 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.1
  have c95687 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.1
  have c95688 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.1
  have c95689 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.1
  have c95690 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_0 := arithEq_of_rows h (row := 1563) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_0
  simp only [← c95685, ← c95686, ← c95687] at e_1563_0
  have e_1563_1 := arithEq_of_rows h (row := 1563) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_1
  simp only [← c95688, ← c95689, k2, ← c95690] at e_1563_1
  have hr := e_1563_1
  simp only [e_1563_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15349)) := by
  have c95691 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95692 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95693 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95694 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95695 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95696 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_2 := arithEq_of_rows h (row := 1563) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_2
  simp only [← c95691, ← c95692, ← c95693] at e_1563_2
  have e_1563_3 := arithEq_of_rows h (row := 1563) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_3
  simp only [← c95694, ← c95695, k2, ← c95696] at e_1563_3
  have hr := e_1563_3
  simp only [e_1563_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15350)) := by
  have c95697 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95698 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95699 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95700 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95701 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95702 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_4 := arithEq_of_rows h (row := 1563) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_4
  simp only [← c95697, ← c95698, ← c95699] at e_1563_4
  have e_1563_5 := arithEq_of_rows h (row := 1563) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_5
  simp only [← c95700, ← c95701, k2, ← c95702] at e_1563_5
  have hr := e_1563_5
  simp only [e_1563_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15351)) := by
  have c95703 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95704 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95705 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95706 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95707 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95708 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_6 := arithEq_of_rows h (row := 1563) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_6
  simp only [← c95703, ← c95704, ← c95705] at e_1563_6
  have e_1563_7 := arithEq_of_rows h (row := 1563) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_7
  simp only [← c95706, ← c95707, k2, ← c95708] at e_1563_7
  have hr := e_1563_7
  simp only [e_1563_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15352)) := by
  have c95709 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95710 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95711 := (publicBatchWrapper86_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95712 := (publicBatchWrapper86_copies2991 a h).1
  have c95713 := (publicBatchWrapper86_copies2991 a h).2.1
  have c95714 := (publicBatchWrapper86_copies2991 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_8 := arithEq_of_rows h (row := 1563) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_8
  simp only [← c95709, ← c95710, ← c95711] at e_1563_8
  have e_1563_9 := arithEq_of_rows h (row := 1563) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_9
  simp only [← c95712, ← c95713, k2, ← c95714] at e_1563_9
  have hr := e_1563_9
  simp only [e_1563_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15353)) := by
  have c95715 := (publicBatchWrapper86_copies2991 a h).2.2.2.1
  have c95716 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.1
  have c95717 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.1
  have c95718 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.1
  have c95719 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.1
  have c95720 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_10 := arithEq_of_rows h (row := 1563) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_10
  simp only [← c95715, ← c95716, ← c95717] at e_1563_10
  have e_1563_11 := arithEq_of_rows h (row := 1563) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_11
  simp only [← c95718, ← c95719, k2, ← c95720] at e_1563_11
  have hr := e_1563_11
  simp only [e_1563_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15354)) := by
  have c95721 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.1
  have c95722 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95723 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95724 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95725 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95726 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_12 := arithEq_of_rows h (row := 1563) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_12
  simp only [← c95721, ← c95722, ← c95723] at e_1563_12
  have e_1563_13 := arithEq_of_rows h (row := 1563) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_13
  simp only [← c95724, ← c95725, k2, ← c95726] at e_1563_13
  have hr := e_1563_13
  simp only [e_1563_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15355)) := by
  have c95727 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95728 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95729 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95730 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95731 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95732 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_14 := arithEq_of_rows h (row := 1563) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_14
  simp only [← c95727, ← c95728, ← c95729] at e_1563_14
  have e_1563_15 := arithEq_of_rows h (row := 1563) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_15
  simp only [← c95730, ← c95731, k2, ← c95732] at e_1563_15
  have hr := e_1563_15
  simp only [e_1563_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15356)) := by
  have c95733 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95734 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95735 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95736 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95737 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95738 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_16 := arithEq_of_rows h (row := 1563) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_16
  simp only [← c95733, ← c95734, ← c95735] at e_1563_16
  have e_1563_17 := arithEq_of_rows h (row := 1563) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_17
  simp only [← c95736, ← c95737, k2, ← c95738] at e_1563_17
  have hr := e_1563_17
  simp only [e_1563_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1563 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15357)) := by
  have c95739 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95740 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95741 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95742 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95743 := (publicBatchWrapper86_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95744 := (publicBatchWrapper86_copies2992 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1563_18 := arithEq_of_rows h (row := 1563) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_18
  simp only [← c95739, ← c95740, ← c95741] at e_1563_18
  have e_1563_19 := arithEq_of_rows h (row := 1563) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_19
  simp only [← c95742, ← c95743, k2, ← c95744] at e_1563_19
  have hr := e_1563_19
  simp only [e_1563_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15358)) := by
  have c95745 := (publicBatchWrapper86_copies2992 a h).2.1
  have c95746 := (publicBatchWrapper86_copies2992 a h).2.2.1
  have c95747 := (publicBatchWrapper86_copies2992 a h).2.2.2.1
  have c95748 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.1
  have c95749 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.1
  have c95750 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_0 := arithEq_of_rows h (row := 1564) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_0
  simp only [← c95745, ← c95746, ← c95747] at e_1564_0
  have e_1564_1 := arithEq_of_rows h (row := 1564) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_1
  simp only [← c95748, ← c95749, k2, ← c95750] at e_1564_1
  have hr := e_1564_1
  simp only [e_1564_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15359)) := by
  have c95751 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.1
  have c95752 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.1
  have c95753 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.1
  have c95754 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95755 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95756 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_2 := arithEq_of_rows h (row := 1564) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_2
  simp only [← c95751, ← c95752, ← c95753] at e_1564_2
  have e_1564_3 := arithEq_of_rows h (row := 1564) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_3
  simp only [← c95754, ← c95755, k2, ← c95756] at e_1564_3
  have hr := e_1564_3
  simp only [e_1564_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 23) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15360)) := by
  have c95757 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95758 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95759 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95760 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95761 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95762 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_4 := arithEq_of_rows h (row := 1564) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_4
  simp only [← c95757, ← c95758, ← c95759] at e_1564_4
  have e_1564_5 := arithEq_of_rows h (row := 1564) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_5
  simp only [← c95760, ← c95761, k2, ← c95762] at e_1564_5
  have hr := e_1564_5
  simp only [e_1564_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 31) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15361)) := by
  have c95763 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95764 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95765 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95766 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95767 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95768 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_6 := arithEq_of_rows h (row := 1564) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_6
  simp only [← c95763, ← c95764, ← c95765] at e_1564_6
  have e_1564_7 := arithEq_of_rows h (row := 1564) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_7
  simp only [← c95766, ← c95767, k2, ← c95768] at e_1564_7
  have hr := e_1564_7
  simp only [e_1564_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 39) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15362)) := by
  have c95769 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95770 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95771 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95772 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95773 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95774 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_8 := arithEq_of_rows h (row := 1564) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_8
  simp only [← c95769, ← c95770, ← c95771] at e_1564_8
  have e_1564_9 := arithEq_of_rows h (row := 1564) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_9
  simp only [← c95772, ← c95773, k2, ← c95774] at e_1564_9
  have hr := e_1564_9
  simp only [e_1564_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 47) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15363)) := by
  have c95775 := (publicBatchWrapper86_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95776 := (publicBatchWrapper86_copies2993 a h).1
  have c95777 := (publicBatchWrapper86_copies2993 a h).2.1
  have c95778 := (publicBatchWrapper86_copies2993 a h).2.2.1
  have c95779 := (publicBatchWrapper86_copies2993 a h).2.2.2.1
  have c95780 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_10 := arithEq_of_rows h (row := 1564) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_10
  simp only [← c95775, ← c95776, ← c95777] at e_1564_10
  have e_1564_11 := arithEq_of_rows h (row := 1564) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_11
  simp only [← c95778, ← c95779, k2, ← c95780] at e_1564_11
  have hr := e_1564_11
  simp only [e_1564_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 55) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15364)) := by
  have c95781 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.1
  have c95782 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.1
  have c95783 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.1
  have c95784 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.1
  have c95785 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.1
  have c95786 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_12 := arithEq_of_rows h (row := 1564) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_12
  simp only [← c95781, ← c95782, ← c95783] at e_1564_12
  have e_1564_13 := arithEq_of_rows h (row := 1564) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_13
  simp only [← c95784, ← c95785, k2, ← c95786] at e_1564_13
  have hr := e_1564_13
  simp only [e_1564_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 63) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15365)) := by
  have c95787 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95788 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95789 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95790 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95791 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95792 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_14 := arithEq_of_rows h (row := 1564) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_14
  simp only [← c95787, ← c95788, ← c95789] at e_1564_14
  have e_1564_15 := arithEq_of_rows h (row := 1564) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_15
  simp only [← c95790, ← c95791, k2, ← c95792] at e_1564_15
  have hr := e_1564_15
  simp only [e_1564_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 71) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15366)) := by
  have c95793 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95794 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95795 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95796 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95797 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95798 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_16 := arithEq_of_rows h (row := 1564) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_16
  simp only [← c95793, ← c95794, ← c95795] at e_1564_16
  have e_1564_17 := arithEq_of_rows h (row := 1564) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_17
  simp only [← c95796, ← c95797, k2, ← c95798] at e_1564_17
  have hr := e_1564_17
  simp only [e_1564_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1564 79) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15367)) := by
  have c95799 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95800 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95801 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95802 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95803 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95804 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1564_18 := arithEq_of_rows h (row := 1564) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_18
  simp only [← c95799, ← c95800, ← c95801] at e_1564_18
  have e_1564_19 := arithEq_of_rows h (row := 1564) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_19
  simp only [← c95802, ← c95803, k2, ← c95804] at e_1564_19
  have hr := e_1564_19
  simp only [e_1564_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 7) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15368)) := by
  have c95805 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95806 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95807 := (publicBatchWrapper86_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95808 := (publicBatchWrapper86_copies2994 a h).1
  have c95809 := (publicBatchWrapper86_copies2994 a h).2.1
  have c95810 := (publicBatchWrapper86_copies2994 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_0 := arithEq_of_rows h (row := 1565) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_0
  simp only [← c95805, ← c95806, ← c95807] at e_1565_0
  have e_1565_1 := arithEq_of_rows h (row := 1565) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_1
  simp only [← c95808, ← c95809, k2, ← c95810] at e_1565_1
  have hr := e_1565_1
  simp only [e_1565_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 15) = bselect (a (.wire 75 11)) (a (.virt 200)) (a (.virt 15369)) := by
  have c95811 := (publicBatchWrapper86_copies2994 a h).2.2.2.1
  have c95812 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.1
  have c95813 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.1
  have c95814 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.1
  have c95815 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.1
  have c95816 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_2 := arithEq_of_rows h (row := 1565) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_2
  simp only [← c95811, ← c95812, ← c95813] at e_1565_2
  have e_1565_3 := arithEq_of_rows h (row := 1565) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_3
  simp only [← c95814, ← c95815, k2, ← c95816] at e_1565_3
  have hr := e_1565_3
  simp only [e_1565_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15410)) := by
  have c95817 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.1
  have c95818 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95819 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95820 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95821 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95822 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_4 := arithEq_of_rows h (row := 1565) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_4
  simp only [← c95817, ← c95818, ← c95819] at e_1565_4
  have e_1565_5 := arithEq_of_rows h (row := 1565) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_5
  simp only [← c95820, ← c95821, k2, ← c95822] at e_1565_5
  have hr := e_1565_5
  simp only [e_1565_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15411)) := by
  have c95823 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95824 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95825 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95826 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95827 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95828 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_6 := arithEq_of_rows h (row := 1565) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_6
  simp only [← c95823, ← c95824, ← c95825] at e_1565_6
  have e_1565_7 := arithEq_of_rows h (row := 1565) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_7
  simp only [← c95826, ← c95827, k2, ← c95828] at e_1565_7
  have hr := e_1565_7
  simp only [e_1565_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15412)) := by
  have c95829 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95830 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95831 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95832 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95833 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95834 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_8 := arithEq_of_rows h (row := 1565) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_8
  simp only [← c95829, ← c95830, ← c95831] at e_1565_8
  have e_1565_9 := arithEq_of_rows h (row := 1565) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_9
  simp only [← c95832, ← c95833, k2, ← c95834] at e_1565_9
  have hr := e_1565_9
  simp only [e_1565_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15413)) := by
  have c95835 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95836 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95837 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95838 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95839 := (publicBatchWrapper86_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95840 := (publicBatchWrapper86_copies2995 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_10 := arithEq_of_rows h (row := 1565) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_10
  simp only [← c95835, ← c95836, ← c95837] at e_1565_10
  have e_1565_11 := arithEq_of_rows h (row := 1565) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_11
  simp only [← c95838, ← c95839, k2, ← c95840] at e_1565_11
  have hr := e_1565_11
  simp only [e_1565_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15414)) := by
  have c95841 := (publicBatchWrapper86_copies2995 a h).2.1
  have c95842 := (publicBatchWrapper86_copies2995 a h).2.2.1
  have c95843 := (publicBatchWrapper86_copies2995 a h).2.2.2.1
  have c95844 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.1
  have c95845 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.1
  have c95846 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_12 := arithEq_of_rows h (row := 1565) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_12
  simp only [← c95841, ← c95842, ← c95843] at e_1565_12
  have e_1565_13 := arithEq_of_rows h (row := 1565) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_13
  simp only [← c95844, ← c95845, k2, ← c95846] at e_1565_13
  have hr := e_1565_13
  simp only [e_1565_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15415)) := by
  have c95847 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.1
  have c95848 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.1
  have c95849 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.1
  have c95850 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95851 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95852 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_14 := arithEq_of_rows h (row := 1565) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_14
  simp only [← c95847, ← c95848, ← c95849] at e_1565_14
  have e_1565_15 := arithEq_of_rows h (row := 1565) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_15
  simp only [← c95850, ← c95851, k2, ← c95852] at e_1565_15
  have hr := e_1565_15
  simp only [e_1565_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15416)) := by
  have c95853 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95854 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95855 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95856 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95857 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95858 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_16 := arithEq_of_rows h (row := 1565) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_16
  simp only [← c95853, ← c95854, ← c95855] at e_1565_16
  have e_1565_17 := arithEq_of_rows h (row := 1565) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_17
  simp only [← c95856, ← c95857, k2, ← c95858] at e_1565_17
  have hr := e_1565_17
  simp only [e_1565_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1565 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15417)) := by
  have c95859 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95860 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95861 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95862 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95863 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95864 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1565_18 := arithEq_of_rows h (row := 1565) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_18
  simp only [← c95859, ← c95860, ← c95861] at e_1565_18
  have e_1565_19 := arithEq_of_rows h (row := 1565) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_19
  simp only [← c95862, ← c95863, k2, ← c95864] at e_1565_19
  have hr := e_1565_19
  simp only [e_1565_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15418)) := by
  have c95865 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95866 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95867 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95868 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95869 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95870 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_0 := arithEq_of_rows h (row := 1566) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_0
  simp only [← c95865, ← c95866, ← c95867] at e_1566_0
  have e_1566_1 := arithEq_of_rows h (row := 1566) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_1
  simp only [← c95868, ← c95869, k2, ← c95870] at e_1566_1
  have hr := e_1566_1
  simp only [e_1566_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15419)) := by
  have c95871 := (publicBatchWrapper86_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95872 := (publicBatchWrapper86_copies2996 a h).1
  have c95873 := (publicBatchWrapper86_copies2996 a h).2.1
  have c95874 := (publicBatchWrapper86_copies2996 a h).2.2.1
  have c95875 := (publicBatchWrapper86_copies2996 a h).2.2.2.1
  have c95876 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_2 := arithEq_of_rows h (row := 1566) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_2
  simp only [← c95871, ← c95872, ← c95873] at e_1566_2
  have e_1566_3 := arithEq_of_rows h (row := 1566) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_3
  simp only [← c95874, ← c95875, k2, ← c95876] at e_1566_3
  have hr := e_1566_3
  simp only [e_1566_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15420)) := by
  have c95877 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.1
  have c95878 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.1
  have c95879 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.1
  have c95880 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.1
  have c95881 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.1
  have c95882 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_4 := arithEq_of_rows h (row := 1566) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_4
  simp only [← c95877, ← c95878, ← c95879] at e_1566_4
  have e_1566_5 := arithEq_of_rows h (row := 1566) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_5
  simp only [← c95880, ← c95881, k2, ← c95882] at e_1566_5
  have hr := e_1566_5
  simp only [e_1566_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15421)) := by
  have c95883 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95884 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95885 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95886 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95887 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95888 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_6 := arithEq_of_rows h (row := 1566) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_6
  simp only [← c95883, ← c95884, ← c95885] at e_1566_6
  have e_1566_7 := arithEq_of_rows h (row := 1566) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_7
  simp only [← c95886, ← c95887, k2, ← c95888] at e_1566_7
  have hr := e_1566_7
  simp only [e_1566_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15422)) := by
  have c95889 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95890 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95891 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95892 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95893 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95894 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_8 := arithEq_of_rows h (row := 1566) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_8
  simp only [← c95889, ← c95890, ← c95891] at e_1566_8
  have e_1566_9 := arithEq_of_rows h (row := 1566) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_9
  simp only [← c95892, ← c95893, k2, ← c95894] at e_1566_9
  have hr := e_1566_9
  simp only [e_1566_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15423)) := by
  have c95895 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95896 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95897 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95898 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95899 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95900 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_10 := arithEq_of_rows h (row := 1566) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_10
  simp only [← c95895, ← c95896, ← c95897] at e_1566_10
  have e_1566_11 := arithEq_of_rows h (row := 1566) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_11
  simp only [← c95898, ← c95899, k2, ← c95900] at e_1566_11
  have hr := e_1566_11
  simp only [e_1566_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15424)) := by
  have c95901 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95902 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95903 := (publicBatchWrapper86_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95904 := (publicBatchWrapper86_copies2997 a h).1
  have c95905 := (publicBatchWrapper86_copies2997 a h).2.1
  have c95906 := (publicBatchWrapper86_copies2997 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_12 := arithEq_of_rows h (row := 1566) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_12
  simp only [← c95901, ← c95902, ← c95903] at e_1566_12
  have e_1566_13 := arithEq_of_rows h (row := 1566) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_13
  simp only [← c95904, ← c95905, k2, ← c95906] at e_1566_13
  have hr := e_1566_13
  simp only [e_1566_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15425)) := by
  have c95907 := (publicBatchWrapper86_copies2997 a h).2.2.2.1
  have c95908 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.1
  have c95909 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.1
  have c95910 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.1
  have c95911 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.1
  have c95912 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_14 := arithEq_of_rows h (row := 1566) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_14
  simp only [← c95907, ← c95908, ← c95909] at e_1566_14
  have e_1566_15 := arithEq_of_rows h (row := 1566) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_15
  simp only [← c95910, ← c95911, k2, ← c95912] at e_1566_15
  have hr := e_1566_15
  simp only [e_1566_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15426)) := by
  have c95913 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.1
  have c95914 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95915 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95916 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95917 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95918 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_16 := arithEq_of_rows h (row := 1566) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_16
  simp only [← c95913, ← c95914, ← c95915] at e_1566_16
  have e_1566_17 := arithEq_of_rows h (row := 1566) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_17
  simp only [← c95916, ← c95917, k2, ← c95918] at e_1566_17
  have hr := e_1566_17
  simp only [e_1566_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1566 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15427)) := by
  have c95919 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95920 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95921 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95922 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95923 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95924 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1566_18 := arithEq_of_rows h (row := 1566) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_18
  simp only [← c95919, ← c95920, ← c95921] at e_1566_18
  have e_1566_19 := arithEq_of_rows h (row := 1566) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_19
  simp only [← c95922, ← c95923, k2, ← c95924] at e_1566_19
  have hr := e_1566_19
  simp only [e_1566_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15428)) := by
  have c95925 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95926 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95927 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95928 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95929 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95930 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_0 := arithEq_of_rows h (row := 1567) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_0
  simp only [← c95925, ← c95926, ← c95927] at e_1567_0
  have e_1567_1 := arithEq_of_rows h (row := 1567) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_1
  simp only [← c95928, ← c95929, k2, ← c95930] at e_1567_1
  have hr := e_1567_1
  simp only [e_1567_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15429)) := by
  have c95931 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95932 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95933 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95934 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95935 := (publicBatchWrapper86_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95936 := (publicBatchWrapper86_copies2998 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_2 := arithEq_of_rows h (row := 1567) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_2
  simp only [← c95931, ← c95932, ← c95933] at e_1567_2
  have e_1567_3 := arithEq_of_rows h (row := 1567) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_3
  simp only [← c95934, ← c95935, k2, ← c95936] at e_1567_3
  have hr := e_1567_3
  simp only [e_1567_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15430)) := by
  have c95937 := (publicBatchWrapper86_copies2998 a h).2.1
  have c95938 := (publicBatchWrapper86_copies2998 a h).2.2.1
  have c95939 := (publicBatchWrapper86_copies2998 a h).2.2.2.1
  have c95940 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.1
  have c95941 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.1
  have c95942 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_4 := arithEq_of_rows h (row := 1567) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_4
  simp only [← c95937, ← c95938, ← c95939] at e_1567_4
  have e_1567_5 := arithEq_of_rows h (row := 1567) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_5
  simp only [← c95940, ← c95941, k2, ← c95942] at e_1567_5
  have hr := e_1567_5
  simp only [e_1567_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15431)) := by
  have c95943 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.1
  have c95944 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.1
  have c95945 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.1
  have c95946 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95947 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95948 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_6 := arithEq_of_rows h (row := 1567) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_6
  simp only [← c95943, ← c95944, ← c95945] at e_1567_6
  have e_1567_7 := arithEq_of_rows h (row := 1567) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_7
  simp only [← c95946, ← c95947, k2, ← c95948] at e_1567_7
  have hr := e_1567_7
  simp only [e_1567_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15432)) := by
  have c95949 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95950 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95951 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95952 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95953 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95954 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_8 := arithEq_of_rows h (row := 1567) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_8
  simp only [← c95949, ← c95950, ← c95951] at e_1567_8
  have e_1567_9 := arithEq_of_rows h (row := 1567) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_9
  simp only [← c95952, ← c95953, k2, ← c95954] at e_1567_9
  have hr := e_1567_9
  simp only [e_1567_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15433)) := by
  have c95955 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95956 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95957 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95958 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95959 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95960 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_10 := arithEq_of_rows h (row := 1567) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_10
  simp only [← c95955, ← c95956, ← c95957] at e_1567_10
  have e_1567_11 := arithEq_of_rows h (row := 1567) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_11
  simp only [← c95958, ← c95959, k2, ← c95960] at e_1567_11
  have hr := e_1567_11
  simp only [e_1567_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15434)) := by
  have c95961 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95962 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95963 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95964 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95965 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95966 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_12 := arithEq_of_rows h (row := 1567) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_12
  simp only [← c95961, ← c95962, ← c95963] at e_1567_12
  have e_1567_13 := arithEq_of_rows h (row := 1567) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_13
  simp only [← c95964, ← c95965, k2, ← c95966] at e_1567_13
  have hr := e_1567_13
  simp only [e_1567_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15435)) := by
  have c95967 := (publicBatchWrapper86_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95968 := (publicBatchWrapper86_copies2999 a h).1
  have c95969 := (publicBatchWrapper86_copies2999 a h).2.1
  have c95970 := (publicBatchWrapper86_copies2999 a h).2.2.1
  have c95971 := (publicBatchWrapper86_copies2999 a h).2.2.2.1
  have c95972 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_14 := arithEq_of_rows h (row := 1567) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_14
  simp only [← c95967, ← c95968, ← c95969] at e_1567_14
  have e_1567_15 := arithEq_of_rows h (row := 1567) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_15
  simp only [← c95970, ← c95971, k2, ← c95972] at e_1567_15
  have hr := e_1567_15
  simp only [e_1567_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15436)) := by
  have c95973 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.1
  have c95974 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.1
  have c95975 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.1
  have c95976 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.1
  have c95977 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.1
  have c95978 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_16 := arithEq_of_rows h (row := 1567) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_16
  simp only [← c95973, ← c95974, ← c95975] at e_1567_16
  have e_1567_17 := arithEq_of_rows h (row := 1567) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_17
  simp only [← c95976, ← c95977, k2, ← c95978] at e_1567_17
  have hr := e_1567_17
  simp only [e_1567_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1567 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15437)) := by
  have c95979 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95980 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95981 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95982 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95983 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95984 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1567_18 := arithEq_of_rows h (row := 1567) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_18
  simp only [← c95979, ← c95980, ← c95981] at e_1567_18
  have e_1567_19 := arithEq_of_rows h (row := 1567) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_19
  simp only [← c95982, ← c95983, k2, ← c95984] at e_1567_19
  have hr := e_1567_19
  simp only [e_1567_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15438)) := by
  have c95985 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95986 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95987 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95988 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95989 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95990 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_0 := arithEq_of_rows h (row := 1568) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_0
  simp only [← c95985, ← c95986, ← c95987] at e_1568_0
  have e_1568_1 := arithEq_of_rows h (row := 1568) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_1
  simp only [← c95988, ← c95989, k2, ← c95990] at e_1568_1
  have hr := e_1568_1
  simp only [e_1568_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15439)) := by
  have c95991 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95992 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95993 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95994 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95995 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95996 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_2 := arithEq_of_rows h (row := 1568) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_2
  simp only [← c95991, ← c95992, ← c95993] at e_1568_2
  have e_1568_3 := arithEq_of_rows h (row := 1568) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_3
  simp only [← c95994, ← c95995, k2, ← c95996] at e_1568_3
  have hr := e_1568_3
  simp only [e_1568_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15440)) := by
  have c95997 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95998 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95999 := (publicBatchWrapper86_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96000 := (publicBatchWrapper86_copies3000 a h).1
  have c96001 := (publicBatchWrapper86_copies3000 a h).2.1
  have c96002 := (publicBatchWrapper86_copies3000 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_4 := arithEq_of_rows h (row := 1568) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_4
  simp only [← c95997, ← c95998, ← c95999] at e_1568_4
  have e_1568_5 := arithEq_of_rows h (row := 1568) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_5
  simp only [← c96000, ← c96001, k2, ← c96002] at e_1568_5
  have hr := e_1568_5
  simp only [e_1568_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15441)) := by
  have c96003 := (publicBatchWrapper86_copies3000 a h).2.2.2.1
  have c96004 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.1
  have c96005 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.1
  have c96006 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.1
  have c96007 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.1
  have c96008 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_6 := arithEq_of_rows h (row := 1568) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_6
  simp only [← c96003, ← c96004, ← c96005] at e_1568_6
  have e_1568_7 := arithEq_of_rows h (row := 1568) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_7
  simp only [← c96006, ← c96007, k2, ← c96008] at e_1568_7
  have hr := e_1568_7
  simp only [e_1568_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15442)) := by
  have c96009 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.1
  have c96010 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96011 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96012 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96013 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96014 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_8 := arithEq_of_rows h (row := 1568) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_8
  simp only [← c96009, ← c96010, ← c96011] at e_1568_8
  have e_1568_9 := arithEq_of_rows h (row := 1568) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_9
  simp only [← c96012, ← c96013, k2, ← c96014] at e_1568_9
  have hr := e_1568_9
  simp only [e_1568_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15443)) := by
  have c96015 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96016 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96017 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96018 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96019 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96020 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_10 := arithEq_of_rows h (row := 1568) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_10
  simp only [← c96015, ← c96016, ← c96017] at e_1568_10
  have e_1568_11 := arithEq_of_rows h (row := 1568) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_11
  simp only [← c96018, ← c96019, k2, ← c96020] at e_1568_11
  have hr := e_1568_11
  simp only [e_1568_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15444)) := by
  have c96021 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96022 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96023 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96024 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96025 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96026 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_12 := arithEq_of_rows h (row := 1568) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_12
  simp only [← c96021, ← c96022, ← c96023] at e_1568_12
  have e_1568_13 := arithEq_of_rows h (row := 1568) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_13
  simp only [← c96024, ← c96025, k2, ← c96026] at e_1568_13
  have hr := e_1568_13
  simp only [e_1568_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15445)) := by
  have c96027 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96028 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96029 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96030 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96031 := (publicBatchWrapper86_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96032 := (publicBatchWrapper86_copies3001 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_14 := arithEq_of_rows h (row := 1568) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_14
  simp only [← c96027, ← c96028, ← c96029] at e_1568_14
  have e_1568_15 := arithEq_of_rows h (row := 1568) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_15
  simp only [← c96030, ← c96031, k2, ← c96032] at e_1568_15
  have hr := e_1568_15
  simp only [e_1568_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15446)) := by
  have c96033 := (publicBatchWrapper86_copies3001 a h).2.1
  have c96034 := (publicBatchWrapper86_copies3001 a h).2.2.1
  have c96035 := (publicBatchWrapper86_copies3001 a h).2.2.2.1
  have c96036 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.1
  have c96037 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.1
  have c96038 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_16 := arithEq_of_rows h (row := 1568) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_16
  simp only [← c96033, ← c96034, ← c96035] at e_1568_16
  have e_1568_17 := arithEq_of_rows h (row := 1568) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_17
  simp only [← c96036, ← c96037, k2, ← c96038] at e_1568_17
  have hr := e_1568_17
  simp only [e_1568_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1568 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15447)) := by
  have c96039 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.1
  have c96040 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.1
  have c96041 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.1
  have c96042 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96043 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96044 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1568_18 := arithEq_of_rows h (row := 1568) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_18
  simp only [← c96039, ← c96040, ← c96041] at e_1568_18
  have e_1568_19 := arithEq_of_rows h (row := 1568) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_19
  simp only [← c96042, ← c96043, k2, ← c96044] at e_1568_19
  have hr := e_1568_19
  simp only [e_1568_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15448)) := by
  have c96045 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96046 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96047 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96048 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96049 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96050 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_0 := arithEq_of_rows h (row := 1569) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_0
  simp only [← c96045, ← c96046, ← c96047] at e_1569_0
  have e_1569_1 := arithEq_of_rows h (row := 1569) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_1
  simp only [← c96048, ← c96049, k2, ← c96050] at e_1569_1
  have hr := e_1569_1
  simp only [e_1569_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15449)) := by
  have c96051 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96052 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96053 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96054 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96055 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96056 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_2 := arithEq_of_rows h (row := 1569) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_2
  simp only [← c96051, ← c96052, ← c96053] at e_1569_2
  have e_1569_3 := arithEq_of_rows h (row := 1569) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_3
  simp only [← c96054, ← c96055, k2, ← c96056] at e_1569_3
  have hr := e_1569_3
  simp only [e_1569_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15450)) := by
  have c96057 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96058 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96059 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96060 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96061 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96062 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_4 := arithEq_of_rows h (row := 1569) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_4
  simp only [← c96057, ← c96058, ← c96059] at e_1569_4
  have e_1569_5 := arithEq_of_rows h (row := 1569) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_5
  simp only [← c96060, ← c96061, k2, ← c96062] at e_1569_5
  have hr := e_1569_5
  simp only [e_1569_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15451)) := by
  have c96063 := (publicBatchWrapper86_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96064 := (publicBatchWrapper86_copies3002 a h).1
  have c96065 := (publicBatchWrapper86_copies3002 a h).2.1
  have c96066 := (publicBatchWrapper86_copies3002 a h).2.2.1
  have c96067 := (publicBatchWrapper86_copies3002 a h).2.2.2.1
  have c96068 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_6 := arithEq_of_rows h (row := 1569) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_6
  simp only [← c96063, ← c96064, ← c96065] at e_1569_6
  have e_1569_7 := arithEq_of_rows h (row := 1569) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_7
  simp only [← c96066, ← c96067, k2, ← c96068] at e_1569_7
  have hr := e_1569_7
  simp only [e_1569_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15452)) := by
  have c96069 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.1
  have c96070 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.1
  have c96071 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.1
  have c96072 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.1
  have c96073 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.1
  have c96074 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_8 := arithEq_of_rows h (row := 1569) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_8
  simp only [← c96069, ← c96070, ← c96071] at e_1569_8
  have e_1569_9 := arithEq_of_rows h (row := 1569) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_9
  simp only [← c96072, ← c96073, k2, ← c96074] at e_1569_9
  have hr := e_1569_9
  simp only [e_1569_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15453)) := by
  have c96075 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96076 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96077 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96078 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96079 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96080 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_10 := arithEq_of_rows h (row := 1569) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_10
  simp only [← c96075, ← c96076, ← c96077] at e_1569_10
  have e_1569_11 := arithEq_of_rows h (row := 1569) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_11
  simp only [← c96078, ← c96079, k2, ← c96080] at e_1569_11
  have hr := e_1569_11
  simp only [e_1569_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15454)) := by
  have c96081 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96082 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96083 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96084 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96085 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96086 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_12 := arithEq_of_rows h (row := 1569) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_12
  simp only [← c96081, ← c96082, ← c96083] at e_1569_12
  have e_1569_13 := arithEq_of_rows h (row := 1569) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_13
  simp only [← c96084, ← c96085, k2, ← c96086] at e_1569_13
  have hr := e_1569_13
  simp only [e_1569_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15455)) := by
  have c96087 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96088 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96089 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96090 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96091 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96092 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_14 := arithEq_of_rows h (row := 1569) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_14
  simp only [← c96087, ← c96088, ← c96089] at e_1569_14
  have e_1569_15 := arithEq_of_rows h (row := 1569) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_15
  simp only [← c96090, ← c96091, k2, ← c96092] at e_1569_15
  have hr := e_1569_15
  simp only [e_1569_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15456)) := by
  have c96093 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96094 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96095 := (publicBatchWrapper86_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96096 := (publicBatchWrapper86_copies3003 a h).1
  have c96097 := (publicBatchWrapper86_copies3003 a h).2.1
  have c96098 := (publicBatchWrapper86_copies3003 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_16 := arithEq_of_rows h (row := 1569) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_16
  simp only [← c96093, ← c96094, ← c96095] at e_1569_16
  have e_1569_17 := arithEq_of_rows h (row := 1569) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_17
  simp only [← c96096, ← c96097, k2, ← c96098] at e_1569_17
  have hr := e_1569_17
  simp only [e_1569_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1569 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15457)) := by
  have c96099 := (publicBatchWrapper86_copies3003 a h).2.2.2.1
  have c96100 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.1
  have c96101 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.1
  have c96102 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.1
  have c96103 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.1
  have c96104 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1569_18 := arithEq_of_rows h (row := 1569) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_18
  simp only [← c96099, ← c96100, ← c96101] at e_1569_18
  have e_1569_19 := arithEq_of_rows h (row := 1569) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_19
  simp only [← c96102, ← c96103, k2, ← c96104] at e_1569_19
  have hr := e_1569_19
  simp only [e_1569_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15458)) := by
  have c96105 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.1
  have c96106 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96107 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96108 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96109 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96110 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_0 := arithEq_of_rows h (row := 1570) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_0
  simp only [← c96105, ← c96106, ← c96107] at e_1570_0
  have e_1570_1 := arithEq_of_rows h (row := 1570) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_1
  simp only [← c96108, ← c96109, k2, ← c96110] at e_1570_1
  have hr := e_1570_1
  simp only [e_1570_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15459)) := by
  have c96111 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96112 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96113 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96114 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96115 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96116 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_2 := arithEq_of_rows h (row := 1570) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_2
  simp only [← c96111, ← c96112, ← c96113] at e_1570_2
  have e_1570_3 := arithEq_of_rows h (row := 1570) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_3
  simp only [← c96114, ← c96115, k2, ← c96116] at e_1570_3
  have hr := e_1570_3
  simp only [e_1570_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15460)) := by
  have c96117 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96118 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96119 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96120 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96121 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96122 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_4 := arithEq_of_rows h (row := 1570) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_4
  simp only [← c96117, ← c96118, ← c96119] at e_1570_4
  have e_1570_5 := arithEq_of_rows h (row := 1570) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_5
  simp only [← c96120, ← c96121, k2, ← c96122] at e_1570_5
  have hr := e_1570_5
  simp only [e_1570_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15461)) := by
  have c96123 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96124 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96125 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96126 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96127 := (publicBatchWrapper86_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96128 := (publicBatchWrapper86_copies3004 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_6 := arithEq_of_rows h (row := 1570) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_6
  simp only [← c96123, ← c96124, ← c96125] at e_1570_6
  have e_1570_7 := arithEq_of_rows h (row := 1570) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_7
  simp only [← c96126, ← c96127, k2, ← c96128] at e_1570_7
  have hr := e_1570_7
  simp only [e_1570_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15462)) := by
  have c96129 := (publicBatchWrapper86_copies3004 a h).2.1
  have c96130 := (publicBatchWrapper86_copies3004 a h).2.2.1
  have c96131 := (publicBatchWrapper86_copies3004 a h).2.2.2.1
  have c96132 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.1
  have c96133 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.1
  have c96134 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_8 := arithEq_of_rows h (row := 1570) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_8
  simp only [← c96129, ← c96130, ← c96131] at e_1570_8
  have e_1570_9 := arithEq_of_rows h (row := 1570) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_9
  simp only [← c96132, ← c96133, k2, ← c96134] at e_1570_9
  have hr := e_1570_9
  simp only [e_1570_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15463)) := by
  have c96135 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.1
  have c96136 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.1
  have c96137 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.1
  have c96138 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96139 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96140 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_10 := arithEq_of_rows h (row := 1570) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_10
  simp only [← c96135, ← c96136, ← c96137] at e_1570_10
  have e_1570_11 := arithEq_of_rows h (row := 1570) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_11
  simp only [← c96138, ← c96139, k2, ← c96140] at e_1570_11
  have hr := e_1570_11
  simp only [e_1570_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15464)) := by
  have c96141 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96142 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96143 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96144 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96145 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96146 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_12 := arithEq_of_rows h (row := 1570) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_12
  simp only [← c96141, ← c96142, ← c96143] at e_1570_12
  have e_1570_13 := arithEq_of_rows h (row := 1570) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_13
  simp only [← c96144, ← c96145, k2, ← c96146] at e_1570_13
  have hr := e_1570_13
  simp only [e_1570_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15465)) := by
  have c96147 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96148 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96149 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96150 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96151 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96152 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_14 := arithEq_of_rows h (row := 1570) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_14
  simp only [← c96147, ← c96148, ← c96149] at e_1570_14
  have e_1570_15 := arithEq_of_rows h (row := 1570) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_15
  simp only [← c96150, ← c96151, k2, ← c96152] at e_1570_15
  have hr := e_1570_15
  simp only [e_1570_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15466)) := by
  have c96153 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96154 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96155 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96156 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96157 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96158 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_16 := arithEq_of_rows h (row := 1570) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_16
  simp only [← c96153, ← c96154, ← c96155] at e_1570_16
  have e_1570_17 := arithEq_of_rows h (row := 1570) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_17
  simp only [← c96156, ← c96157, k2, ← c96158] at e_1570_17
  have hr := e_1570_17
  simp only [e_1570_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1570 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15467)) := by
  have c96159 := (publicBatchWrapper86_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96160 := (publicBatchWrapper86_copies3005 a h).1
  have c96161 := (publicBatchWrapper86_copies3005 a h).2.1
  have c96162 := (publicBatchWrapper86_copies3005 a h).2.2.1
  have c96163 := (publicBatchWrapper86_copies3005 a h).2.2.2.1
  have c96164 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1570_18 := arithEq_of_rows h (row := 1570) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_18
  simp only [← c96159, ← c96160, ← c96161] at e_1570_18
  have e_1570_19 := arithEq_of_rows h (row := 1570) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_19
  simp only [← c96162, ← c96163, k2, ← c96164] at e_1570_19
  have hr := e_1570_19
  simp only [e_1570_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15468)) := by
  have c96165 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.1
  have c96166 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.1
  have c96167 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.1
  have c96168 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.1
  have c96169 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.1
  have c96170 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_0 := arithEq_of_rows h (row := 1571) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_0
  simp only [← c96165, ← c96166, ← c96167] at e_1571_0
  have e_1571_1 := arithEq_of_rows h (row := 1571) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_1
  simp only [← c96168, ← c96169, k2, ← c96170] at e_1571_1
  have hr := e_1571_1
  simp only [e_1571_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15469)) := by
  have c96171 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96172 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96173 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96174 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96175 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96176 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_2 := arithEq_of_rows h (row := 1571) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_2
  simp only [← c96171, ← c96172, ← c96173] at e_1571_2
  have e_1571_3 := arithEq_of_rows h (row := 1571) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_3
  simp only [← c96174, ← c96175, k2, ← c96176] at e_1571_3
  have hr := e_1571_3
  simp only [e_1571_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15470)) := by
  have c96177 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96178 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96179 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96180 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96181 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96182 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_4 := arithEq_of_rows h (row := 1571) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_4
  simp only [← c96177, ← c96178, ← c96179] at e_1571_4
  have e_1571_5 := arithEq_of_rows h (row := 1571) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_5
  simp only [← c96180, ← c96181, k2, ← c96182] at e_1571_5
  have hr := e_1571_5
  simp only [e_1571_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15471)) := by
  have c96183 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96184 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96185 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96186 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96187 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96188 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_6 := arithEq_of_rows h (row := 1571) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_6
  simp only [← c96183, ← c96184, ← c96185] at e_1571_6
  have e_1571_7 := arithEq_of_rows h (row := 1571) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_7
  simp only [← c96186, ← c96187, k2, ← c96188] at e_1571_7
  have hr := e_1571_7
  simp only [e_1571_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15472)) := by
  have c96189 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96190 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96191 := (publicBatchWrapper86_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96192 := (publicBatchWrapper86_copies3006 a h).1
  have c96193 := (publicBatchWrapper86_copies3006 a h).2.1
  have c96194 := (publicBatchWrapper86_copies3006 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_8 := arithEq_of_rows h (row := 1571) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_8
  simp only [← c96189, ← c96190, ← c96191] at e_1571_8
  have e_1571_9 := arithEq_of_rows h (row := 1571) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_9
  simp only [← c96192, ← c96193, k2, ← c96194] at e_1571_9
  have hr := e_1571_9
  simp only [e_1571_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15473)) := by
  have c96195 := (publicBatchWrapper86_copies3006 a h).2.2.2.1
  have c96196 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.1
  have c96197 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.1
  have c96198 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.1
  have c96199 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.1
  have c96200 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_10 := arithEq_of_rows h (row := 1571) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_10
  simp only [← c96195, ← c96196, ← c96197] at e_1571_10
  have e_1571_11 := arithEq_of_rows h (row := 1571) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_11
  simp only [← c96198, ← c96199, k2, ← c96200] at e_1571_11
  have hr := e_1571_11
  simp only [e_1571_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15474)) := by
  have c96201 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.1
  have c96202 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96203 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96204 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96205 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96206 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_12 := arithEq_of_rows h (row := 1571) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_12
  simp only [← c96201, ← c96202, ← c96203] at e_1571_12
  have e_1571_13 := arithEq_of_rows h (row := 1571) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_13
  simp only [← c96204, ← c96205, k2, ← c96206] at e_1571_13
  have hr := e_1571_13
  simp only [e_1571_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15475)) := by
  have c96207 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96208 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96209 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96210 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96211 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96212 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_14 := arithEq_of_rows h (row := 1571) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_14
  simp only [← c96207, ← c96208, ← c96209] at e_1571_14
  have e_1571_15 := arithEq_of_rows h (row := 1571) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_15
  simp only [← c96210, ← c96211, k2, ← c96212] at e_1571_15
  have hr := e_1571_15
  simp only [e_1571_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15476)) := by
  have c96213 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96214 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96215 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96216 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96217 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96218 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_16 := arithEq_of_rows h (row := 1571) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_16
  simp only [← c96213, ← c96214, ← c96215] at e_1571_16
  have e_1571_17 := arithEq_of_rows h (row := 1571) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_17
  simp only [← c96216, ← c96217, k2, ← c96218] at e_1571_17
  have hr := e_1571_17
  simp only [e_1571_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1571 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15477)) := by
  have c96219 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96220 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96221 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96222 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96223 := (publicBatchWrapper86_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96224 := (publicBatchWrapper86_copies3007 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1571_18 := arithEq_of_rows h (row := 1571) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_18
  simp only [← c96219, ← c96220, ← c96221] at e_1571_18
  have e_1571_19 := arithEq_of_rows h (row := 1571) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_19
  simp only [← c96222, ← c96223, k2, ← c96224] at e_1571_19
  have hr := e_1571_19
  simp only [e_1571_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15478)) := by
  have c96225 := (publicBatchWrapper86_copies3007 a h).2.1
  have c96226 := (publicBatchWrapper86_copies3007 a h).2.2.1
  have c96227 := (publicBatchWrapper86_copies3007 a h).2.2.2.1
  have c96228 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.1
  have c96229 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.1
  have c96230 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_0 := arithEq_of_rows h (row := 1572) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_0
  simp only [← c96225, ← c96226, ← c96227] at e_1572_0
  have e_1572_1 := arithEq_of_rows h (row := 1572) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_1
  simp only [← c96228, ← c96229, k2, ← c96230] at e_1572_1
  have hr := e_1572_1
  simp only [e_1572_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15479)) := by
  have c96231 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.1
  have c96232 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.1
  have c96233 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.1
  have c96234 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96235 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96236 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_2 := arithEq_of_rows h (row := 1572) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_2
  simp only [← c96231, ← c96232, ← c96233] at e_1572_2
  have e_1572_3 := arithEq_of_rows h (row := 1572) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_3
  simp only [← c96234, ← c96235, k2, ← c96236] at e_1572_3
  have hr := e_1572_3
  simp only [e_1572_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15480)) := by
  have c96237 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96238 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96239 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96240 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96241 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96242 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_4 := arithEq_of_rows h (row := 1572) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_4
  simp only [← c96237, ← c96238, ← c96239] at e_1572_4
  have e_1572_5 := arithEq_of_rows h (row := 1572) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_5
  simp only [← c96240, ← c96241, k2, ← c96242] at e_1572_5
  have hr := e_1572_5
  simp only [e_1572_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15481)) := by
  have c96243 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96244 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96245 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96246 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96247 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96248 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_6 := arithEq_of_rows h (row := 1572) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_6
  simp only [← c96243, ← c96244, ← c96245] at e_1572_6
  have e_1572_7 := arithEq_of_rows h (row := 1572) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_7
  simp only [← c96246, ← c96247, k2, ← c96248] at e_1572_7
  have hr := e_1572_7
  simp only [e_1572_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15482)) := by
  have c96249 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96250 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96251 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96252 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96253 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96254 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_8 := arithEq_of_rows h (row := 1572) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_8
  simp only [← c96249, ← c96250, ← c96251] at e_1572_8
  have e_1572_9 := arithEq_of_rows h (row := 1572) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_9
  simp only [← c96252, ← c96253, k2, ← c96254] at e_1572_9
  have hr := e_1572_9
  simp only [e_1572_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15483)) := by
  have c96255 := (publicBatchWrapper86_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96256 := (publicBatchWrapper86_copies3008 a h).1
  have c96257 := (publicBatchWrapper86_copies3008 a h).2.1
  have c96258 := (publicBatchWrapper86_copies3008 a h).2.2.1
  have c96259 := (publicBatchWrapper86_copies3008 a h).2.2.2.1
  have c96260 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_10 := arithEq_of_rows h (row := 1572) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_10
  simp only [← c96255, ← c96256, ← c96257] at e_1572_10
  have e_1572_11 := arithEq_of_rows h (row := 1572) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_11
  simp only [← c96258, ← c96259, k2, ← c96260] at e_1572_11
  have hr := e_1572_11
  simp only [e_1572_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15484)) := by
  have c96261 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.1
  have c96262 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.1
  have c96263 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.1
  have c96264 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.1
  have c96265 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.1
  have c96266 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_12 := arithEq_of_rows h (row := 1572) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_12
  simp only [← c96261, ← c96262, ← c96263] at e_1572_12
  have e_1572_13 := arithEq_of_rows h (row := 1572) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_13
  simp only [← c96264, ← c96265, k2, ← c96266] at e_1572_13
  have hr := e_1572_13
  simp only [e_1572_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15485)) := by
  have c96267 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96268 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96269 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96270 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96271 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96272 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_14 := arithEq_of_rows h (row := 1572) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_14
  simp only [← c96267, ← c96268, ← c96269] at e_1572_14
  have e_1572_15 := arithEq_of_rows h (row := 1572) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_15
  simp only [← c96270, ← c96271, k2, ← c96272] at e_1572_15
  have hr := e_1572_15
  simp only [e_1572_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15486)) := by
  have c96273 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96274 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96275 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96276 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96277 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96278 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_16 := arithEq_of_rows h (row := 1572) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_16
  simp only [← c96273, ← c96274, ← c96275] at e_1572_16
  have e_1572_17 := arithEq_of_rows h (row := 1572) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_17
  simp only [← c96276, ← c96277, k2, ← c96278] at e_1572_17
  have hr := e_1572_17
  simp only [e_1572_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1572 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15487)) := by
  have c96279 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96280 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96281 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96282 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96283 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96284 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1572_18 := arithEq_of_rows h (row := 1572) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_18
  simp only [← c96279, ← c96280, ← c96281] at e_1572_18
  have e_1572_19 := arithEq_of_rows h (row := 1572) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_19
  simp only [← c96282, ← c96283, k2, ← c96284] at e_1572_19
  have hr := e_1572_19
  simp only [e_1572_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15488)) := by
  have c96285 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96286 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96287 := (publicBatchWrapper86_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96288 := (publicBatchWrapper86_copies3009 a h).1
  have c96289 := (publicBatchWrapper86_copies3009 a h).2.1
  have c96290 := (publicBatchWrapper86_copies3009 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_0 := arithEq_of_rows h (row := 1573) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_0
  simp only [← c96285, ← c96286, ← c96287] at e_1573_0
  have e_1573_1 := arithEq_of_rows h (row := 1573) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_1
  simp only [← c96288, ← c96289, k2, ← c96290] at e_1573_1
  have hr := e_1573_1
  simp only [e_1573_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15489)) := by
  have c96291 := (publicBatchWrapper86_copies3009 a h).2.2.2.1
  have c96292 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.1
  have c96293 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.1
  have c96294 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.1
  have c96295 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.1
  have c96296 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_2 := arithEq_of_rows h (row := 1573) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_2
  simp only [← c96291, ← c96292, ← c96293] at e_1573_2
  have e_1573_3 := arithEq_of_rows h (row := 1573) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_3
  simp only [← c96294, ← c96295, k2, ← c96296] at e_1573_3
  have hr := e_1573_3
  simp only [e_1573_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15490)) := by
  have c96297 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.1
  have c96298 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96299 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96300 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96301 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96302 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_4 := arithEq_of_rows h (row := 1573) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_4
  simp only [← c96297, ← c96298, ← c96299] at e_1573_4
  have e_1573_5 := arithEq_of_rows h (row := 1573) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_5
  simp only [← c96300, ← c96301, k2, ← c96302] at e_1573_5
  have hr := e_1573_5
  simp only [e_1573_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15491)) := by
  have c96303 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96304 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96305 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96306 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96307 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96308 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_6 := arithEq_of_rows h (row := 1573) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_6
  simp only [← c96303, ← c96304, ← c96305] at e_1573_6
  have e_1573_7 := arithEq_of_rows h (row := 1573) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_7
  simp only [← c96306, ← c96307, k2, ← c96308] at e_1573_7
  have hr := e_1573_7
  simp only [e_1573_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15492)) := by
  have c96309 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96310 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96311 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96312 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96313 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96314 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_8 := arithEq_of_rows h (row := 1573) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_8
  simp only [← c96309, ← c96310, ← c96311] at e_1573_8
  have e_1573_9 := arithEq_of_rows h (row := 1573) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_9
  simp only [← c96312, ← c96313, k2, ← c96314] at e_1573_9
  have hr := e_1573_9
  simp only [e_1573_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15493)) := by
  have c96315 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96316 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96317 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96318 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96319 := (publicBatchWrapper86_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96320 := (publicBatchWrapper86_copies3010 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_10 := arithEq_of_rows h (row := 1573) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_10
  simp only [← c96315, ← c96316, ← c96317] at e_1573_10
  have e_1573_11 := arithEq_of_rows h (row := 1573) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_11
  simp only [← c96318, ← c96319, k2, ← c96320] at e_1573_11
  have hr := e_1573_11
  simp only [e_1573_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15494)) := by
  have c96321 := (publicBatchWrapper86_copies3010 a h).2.1
  have c96322 := (publicBatchWrapper86_copies3010 a h).2.2.1
  have c96323 := (publicBatchWrapper86_copies3010 a h).2.2.2.1
  have c96324 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.1
  have c96325 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.1
  have c96326 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_12 := arithEq_of_rows h (row := 1573) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_12
  simp only [← c96321, ← c96322, ← c96323] at e_1573_12
  have e_1573_13 := arithEq_of_rows h (row := 1573) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_13
  simp only [← c96324, ← c96325, k2, ← c96326] at e_1573_13
  have hr := e_1573_13
  simp only [e_1573_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15495)) := by
  have c96327 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.1
  have c96328 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.1
  have c96329 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.1
  have c96330 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96331 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96332 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_14 := arithEq_of_rows h (row := 1573) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_14
  simp only [← c96327, ← c96328, ← c96329] at e_1573_14
  have e_1573_15 := arithEq_of_rows h (row := 1573) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_15
  simp only [← c96330, ← c96331, k2, ← c96332] at e_1573_15
  have hr := e_1573_15
  simp only [e_1573_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15496)) := by
  have c96333 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96334 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96335 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96336 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96337 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96338 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_16 := arithEq_of_rows h (row := 1573) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_16
  simp only [← c96333, ← c96334, ← c96335] at e_1573_16
  have e_1573_17 := arithEq_of_rows h (row := 1573) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_17
  simp only [← c96336, ← c96337, k2, ← c96338] at e_1573_17
  have hr := e_1573_17
  simp only [e_1573_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1573 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15497)) := by
  have c96339 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96340 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96341 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96342 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96343 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96344 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1573_18 := arithEq_of_rows h (row := 1573) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_18
  simp only [← c96339, ← c96340, ← c96341] at e_1573_18
  have e_1573_19 := arithEq_of_rows h (row := 1573) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_19
  simp only [← c96342, ← c96343, k2, ← c96344] at e_1573_19
  have hr := e_1573_19
  simp only [e_1573_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15498)) := by
  have c96345 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96346 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96347 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96348 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96349 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96350 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_0 := arithEq_of_rows h (row := 1574) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_0
  simp only [← c96345, ← c96346, ← c96347] at e_1574_0
  have e_1574_1 := arithEq_of_rows h (row := 1574) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_1
  simp only [← c96348, ← c96349, k2, ← c96350] at e_1574_1
  have hr := e_1574_1
  simp only [e_1574_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15499)) := by
  have c96351 := (publicBatchWrapper86_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96352 := (publicBatchWrapper86_copies3011 a h).1
  have c96353 := (publicBatchWrapper86_copies3011 a h).2.1
  have c96354 := (publicBatchWrapper86_copies3011 a h).2.2.1
  have c96355 := (publicBatchWrapper86_copies3011 a h).2.2.2.1
  have c96356 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_2 := arithEq_of_rows h (row := 1574) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_2
  simp only [← c96351, ← c96352, ← c96353] at e_1574_2
  have e_1574_3 := arithEq_of_rows h (row := 1574) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_3
  simp only [← c96354, ← c96355, k2, ← c96356] at e_1574_3
  have hr := e_1574_3
  simp only [e_1574_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15500)) := by
  have c96357 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.1
  have c96358 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.1
  have c96359 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.1
  have c96360 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.1
  have c96361 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.1
  have c96362 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_4 := arithEq_of_rows h (row := 1574) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_4
  simp only [← c96357, ← c96358, ← c96359] at e_1574_4
  have e_1574_5 := arithEq_of_rows h (row := 1574) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_5
  simp only [← c96360, ← c96361, k2, ← c96362] at e_1574_5
  have hr := e_1574_5
  simp only [e_1574_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15501)) := by
  have c96363 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96364 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96365 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96366 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96367 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96368 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_6 := arithEq_of_rows h (row := 1574) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_6
  simp only [← c96363, ← c96364, ← c96365] at e_1574_6
  have e_1574_7 := arithEq_of_rows h (row := 1574) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_7
  simp only [← c96366, ← c96367, k2, ← c96368] at e_1574_7
  have hr := e_1574_7
  simp only [e_1574_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15502)) := by
  have c96369 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96370 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96371 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96372 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96373 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96374 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_8 := arithEq_of_rows h (row := 1574) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_8
  simp only [← c96369, ← c96370, ← c96371] at e_1574_8
  have e_1574_9 := arithEq_of_rows h (row := 1574) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_9
  simp only [← c96372, ← c96373, k2, ← c96374] at e_1574_9
  have hr := e_1574_9
  simp only [e_1574_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15503)) := by
  have c96375 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96376 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96377 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96378 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96379 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96380 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_10 := arithEq_of_rows h (row := 1574) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_10
  simp only [← c96375, ← c96376, ← c96377] at e_1574_10
  have e_1574_11 := arithEq_of_rows h (row := 1574) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_11
  simp only [← c96378, ← c96379, k2, ← c96380] at e_1574_11
  have hr := e_1574_11
  simp only [e_1574_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15504)) := by
  have c96381 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96382 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96383 := (publicBatchWrapper86_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96384 := (publicBatchWrapper86_copies3012 a h).1
  have c96385 := (publicBatchWrapper86_copies3012 a h).2.1
  have c96386 := (publicBatchWrapper86_copies3012 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_12 := arithEq_of_rows h (row := 1574) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_12
  simp only [← c96381, ← c96382, ← c96383] at e_1574_12
  have e_1574_13 := arithEq_of_rows h (row := 1574) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_13
  simp only [← c96384, ← c96385, k2, ← c96386] at e_1574_13
  have hr := e_1574_13
  simp only [e_1574_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15505)) := by
  have c96387 := (publicBatchWrapper86_copies3012 a h).2.2.2.1
  have c96388 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.1
  have c96389 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.1
  have c96390 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.1
  have c96391 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.1
  have c96392 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_14 := arithEq_of_rows h (row := 1574) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_14
  simp only [← c96387, ← c96388, ← c96389] at e_1574_14
  have e_1574_15 := arithEq_of_rows h (row := 1574) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_15
  simp only [← c96390, ← c96391, k2, ← c96392] at e_1574_15
  have hr := e_1574_15
  simp only [e_1574_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15506)) := by
  have c96393 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.1
  have c96394 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96395 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96396 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96397 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96398 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_16 := arithEq_of_rows h (row := 1574) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_16
  simp only [← c96393, ← c96394, ← c96395] at e_1574_16
  have e_1574_17 := arithEq_of_rows h (row := 1574) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_17
  simp only [← c96396, ← c96397, k2, ← c96398] at e_1574_17
  have hr := e_1574_17
  simp only [e_1574_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1574 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15507)) := by
  have c96399 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96400 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96401 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96402 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96403 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96404 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1574_18 := arithEq_of_rows h (row := 1574) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_18
  simp only [← c96399, ← c96400, ← c96401] at e_1574_18
  have e_1574_19 := arithEq_of_rows h (row := 1574) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_19
  simp only [← c96402, ← c96403, k2, ← c96404] at e_1574_19
  have hr := e_1574_19
  simp only [e_1574_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15508)) := by
  have c96405 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96406 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96407 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96408 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96409 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96410 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_0 := arithEq_of_rows h (row := 1575) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_0
  simp only [← c96405, ← c96406, ← c96407] at e_1575_0
  have e_1575_1 := arithEq_of_rows h (row := 1575) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_1
  simp only [← c96408, ← c96409, k2, ← c96410] at e_1575_1
  have hr := e_1575_1
  simp only [e_1575_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15509)) := by
  have c96411 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96412 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96413 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96414 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96415 := (publicBatchWrapper86_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96416 := (publicBatchWrapper86_copies3013 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_2 := arithEq_of_rows h (row := 1575) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_2
  simp only [← c96411, ← c96412, ← c96413] at e_1575_2
  have e_1575_3 := arithEq_of_rows h (row := 1575) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_3
  simp only [← c96414, ← c96415, k2, ← c96416] at e_1575_3
  have hr := e_1575_3
  simp only [e_1575_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15510)) := by
  have c96417 := (publicBatchWrapper86_copies3013 a h).2.1
  have c96418 := (publicBatchWrapper86_copies3013 a h).2.2.1
  have c96419 := (publicBatchWrapper86_copies3013 a h).2.2.2.1
  have c96420 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.1
  have c96421 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.1
  have c96422 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_4 := arithEq_of_rows h (row := 1575) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_4
  simp only [← c96417, ← c96418, ← c96419] at e_1575_4
  have e_1575_5 := arithEq_of_rows h (row := 1575) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_5
  simp only [← c96420, ← c96421, k2, ← c96422] at e_1575_5
  have hr := e_1575_5
  simp only [e_1575_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15511)) := by
  have c96423 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.1
  have c96424 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.1
  have c96425 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.1
  have c96426 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96427 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96428 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_6 := arithEq_of_rows h (row := 1575) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_6
  simp only [← c96423, ← c96424, ← c96425] at e_1575_6
  have e_1575_7 := arithEq_of_rows h (row := 1575) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_7
  simp only [← c96426, ← c96427, k2, ← c96428] at e_1575_7
  have hr := e_1575_7
  simp only [e_1575_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15512)) := by
  have c96429 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96430 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96431 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96432 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96433 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96434 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_8 := arithEq_of_rows h (row := 1575) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_8
  simp only [← c96429, ← c96430, ← c96431] at e_1575_8
  have e_1575_9 := arithEq_of_rows h (row := 1575) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_9
  simp only [← c96432, ← c96433, k2, ← c96434] at e_1575_9
  have hr := e_1575_9
  simp only [e_1575_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15513)) := by
  have c96435 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96436 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96437 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96438 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96439 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96440 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_10 := arithEq_of_rows h (row := 1575) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_10
  simp only [← c96435, ← c96436, ← c96437] at e_1575_10
  have e_1575_11 := arithEq_of_rows h (row := 1575) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_11
  simp only [← c96438, ← c96439, k2, ← c96440] at e_1575_11
  have hr := e_1575_11
  simp only [e_1575_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15514)) := by
  have c96441 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96442 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96443 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96444 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96445 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96446 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_12 := arithEq_of_rows h (row := 1575) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_12
  simp only [← c96441, ← c96442, ← c96443] at e_1575_12
  have e_1575_13 := arithEq_of_rows h (row := 1575) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_13
  simp only [← c96444, ← c96445, k2, ← c96446] at e_1575_13
  have hr := e_1575_13
  simp only [e_1575_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15515)) := by
  have c96447 := (publicBatchWrapper86_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96448 := (publicBatchWrapper86_copies3014 a h).1
  have c96449 := (publicBatchWrapper86_copies3014 a h).2.1
  have c96450 := (publicBatchWrapper86_copies3014 a h).2.2.1
  have c96451 := (publicBatchWrapper86_copies3014 a h).2.2.2.1
  have c96452 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_14 := arithEq_of_rows h (row := 1575) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_14
  simp only [← c96447, ← c96448, ← c96449] at e_1575_14
  have e_1575_15 := arithEq_of_rows h (row := 1575) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_15
  simp only [← c96450, ← c96451, k2, ← c96452] at e_1575_15
  have hr := e_1575_15
  simp only [e_1575_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15516)) := by
  have c96453 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.1
  have c96454 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.1
  have c96455 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.1
  have c96456 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.1
  have c96457 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.1
  have c96458 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_16 := arithEq_of_rows h (row := 1575) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_16
  simp only [← c96453, ← c96454, ← c96455] at e_1575_16
  have e_1575_17 := arithEq_of_rows h (row := 1575) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_17
  simp only [← c96456, ← c96457, k2, ← c96458] at e_1575_17
  have hr := e_1575_17
  simp only [e_1575_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1575 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15517)) := by
  have c96459 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96460 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96461 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96462 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96463 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96464 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1575_18 := arithEq_of_rows h (row := 1575) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_18
  simp only [← c96459, ← c96460, ← c96461] at e_1575_18
  have e_1575_19 := arithEq_of_rows h (row := 1575) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_19
  simp only [← c96462, ← c96463, k2, ← c96464] at e_1575_19
  have hr := e_1575_19
  simp only [e_1575_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15518)) := by
  have c96465 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96466 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96467 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96468 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96469 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96470 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_0 := arithEq_of_rows h (row := 1576) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_0
  simp only [← c96465, ← c96466, ← c96467] at e_1576_0
  have e_1576_1 := arithEq_of_rows h (row := 1576) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_1
  simp only [← c96468, ← c96469, k2, ← c96470] at e_1576_1
  have hr := e_1576_1
  simp only [e_1576_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15519)) := by
  have c96471 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96472 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96473 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96474 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96475 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96476 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_2 := arithEq_of_rows h (row := 1576) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_2
  simp only [← c96471, ← c96472, ← c96473] at e_1576_2
  have e_1576_3 := arithEq_of_rows h (row := 1576) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_3
  simp only [← c96474, ← c96475, k2, ← c96476] at e_1576_3
  have hr := e_1576_3
  simp only [e_1576_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15520)) := by
  have c96477 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96478 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96479 := (publicBatchWrapper86_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96480 := (publicBatchWrapper86_copies3015 a h).1
  have c96481 := (publicBatchWrapper86_copies3015 a h).2.1
  have c96482 := (publicBatchWrapper86_copies3015 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_4 := arithEq_of_rows h (row := 1576) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_4
  simp only [← c96477, ← c96478, ← c96479] at e_1576_4
  have e_1576_5 := arithEq_of_rows h (row := 1576) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_5
  simp only [← c96480, ← c96481, k2, ← c96482] at e_1576_5
  have hr := e_1576_5
  simp only [e_1576_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15521)) := by
  have c96483 := (publicBatchWrapper86_copies3015 a h).2.2.2.1
  have c96484 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.1
  have c96485 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.1
  have c96486 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.1
  have c96487 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.1
  have c96488 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_6 := arithEq_of_rows h (row := 1576) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_6
  simp only [← c96483, ← c96484, ← c96485] at e_1576_6
  have e_1576_7 := arithEq_of_rows h (row := 1576) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_7
  simp only [← c96486, ← c96487, k2, ← c96488] at e_1576_7
  have hr := e_1576_7
  simp only [e_1576_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15522)) := by
  have c96489 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.1
  have c96490 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96491 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96492 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96493 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96494 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_8 := arithEq_of_rows h (row := 1576) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_8
  simp only [← c96489, ← c96490, ← c96491] at e_1576_8
  have e_1576_9 := arithEq_of_rows h (row := 1576) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_9
  simp only [← c96492, ← c96493, k2, ← c96494] at e_1576_9
  have hr := e_1576_9
  simp only [e_1576_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15523)) := by
  have c96495 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96496 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96497 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96498 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96499 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96500 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_10 := arithEq_of_rows h (row := 1576) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_10
  simp only [← c96495, ← c96496, ← c96497] at e_1576_10
  have e_1576_11 := arithEq_of_rows h (row := 1576) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_11
  simp only [← c96498, ← c96499, k2, ← c96500] at e_1576_11
  have hr := e_1576_11
  simp only [e_1576_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15524)) := by
  have c96501 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96502 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96503 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96504 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96505 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96506 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_12 := arithEq_of_rows h (row := 1576) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_12
  simp only [← c96501, ← c96502, ← c96503] at e_1576_12
  have e_1576_13 := arithEq_of_rows h (row := 1576) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_13
  simp only [← c96504, ← c96505, k2, ← c96506] at e_1576_13
  have hr := e_1576_13
  simp only [e_1576_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15525)) := by
  have c96507 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96508 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96509 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96510 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96511 := (publicBatchWrapper86_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96512 := (publicBatchWrapper86_copies3016 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_14 := arithEq_of_rows h (row := 1576) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_14
  simp only [← c96507, ← c96508, ← c96509] at e_1576_14
  have e_1576_15 := arithEq_of_rows h (row := 1576) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_15
  simp only [← c96510, ← c96511, k2, ← c96512] at e_1576_15
  have hr := e_1576_15
  simp only [e_1576_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15526)) := by
  have c96513 := (publicBatchWrapper86_copies3016 a h).2.1
  have c96514 := (publicBatchWrapper86_copies3016 a h).2.2.1
  have c96515 := (publicBatchWrapper86_copies3016 a h).2.2.2.1
  have c96516 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.1
  have c96517 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.1
  have c96518 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_16 := arithEq_of_rows h (row := 1576) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_16
  simp only [← c96513, ← c96514, ← c96515] at e_1576_16
  have e_1576_17 := arithEq_of_rows h (row := 1576) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_17
  simp only [← c96516, ← c96517, k2, ← c96518] at e_1576_17
  have hr := e_1576_17
  simp only [e_1576_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1576 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15527)) := by
  have c96519 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.1
  have c96520 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.1
  have c96521 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.1
  have c96522 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96523 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96524 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1576_18 := arithEq_of_rows h (row := 1576) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_18
  simp only [← c96519, ← c96520, ← c96521] at e_1576_18
  have e_1576_19 := arithEq_of_rows h (row := 1576) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_19
  simp only [← c96522, ← c96523, k2, ← c96524] at e_1576_19
  have hr := e_1576_19
  simp only [e_1576_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15528)) := by
  have c96525 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96526 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96527 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96528 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96529 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96530 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_0 := arithEq_of_rows h (row := 1577) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_0
  simp only [← c96525, ← c96526, ← c96527] at e_1577_0
  have e_1577_1 := arithEq_of_rows h (row := 1577) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_1
  simp only [← c96528, ← c96529, k2, ← c96530] at e_1577_1
  have hr := e_1577_1
  simp only [e_1577_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15529)) := by
  have c96531 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96532 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96533 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96534 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96535 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96536 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_2 := arithEq_of_rows h (row := 1577) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_2
  simp only [← c96531, ← c96532, ← c96533] at e_1577_2
  have e_1577_3 := arithEq_of_rows h (row := 1577) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_3
  simp only [← c96534, ← c96535, k2, ← c96536] at e_1577_3
  have hr := e_1577_3
  simp only [e_1577_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15530)) := by
  have c96537 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96538 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96539 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96540 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96541 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96542 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_4 := arithEq_of_rows h (row := 1577) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_4
  simp only [← c96537, ← c96538, ← c96539] at e_1577_4
  have e_1577_5 := arithEq_of_rows h (row := 1577) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_5
  simp only [← c96540, ← c96541, k2, ← c96542] at e_1577_5
  have hr := e_1577_5
  simp only [e_1577_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15531)) := by
  have c96543 := (publicBatchWrapper86_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96544 := (publicBatchWrapper86_copies3017 a h).1
  have c96545 := (publicBatchWrapper86_copies3017 a h).2.1
  have c96546 := (publicBatchWrapper86_copies3017 a h).2.2.1
  have c96547 := (publicBatchWrapper86_copies3017 a h).2.2.2.1
  have c96548 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_6 := arithEq_of_rows h (row := 1577) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_6
  simp only [← c96543, ← c96544, ← c96545] at e_1577_6
  have e_1577_7 := arithEq_of_rows h (row := 1577) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_7
  simp only [← c96546, ← c96547, k2, ← c96548] at e_1577_7
  have hr := e_1577_7
  simp only [e_1577_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15532)) := by
  have c96549 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.1
  have c96550 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.1
  have c96551 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.1
  have c96552 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.1
  have c96553 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.1
  have c96554 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_8 := arithEq_of_rows h (row := 1577) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_8
  simp only [← c96549, ← c96550, ← c96551] at e_1577_8
  have e_1577_9 := arithEq_of_rows h (row := 1577) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_9
  simp only [← c96552, ← c96553, k2, ← c96554] at e_1577_9
  have hr := e_1577_9
  simp only [e_1577_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15533)) := by
  have c96555 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96556 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96557 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96558 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96559 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96560 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_10 := arithEq_of_rows h (row := 1577) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_10
  simp only [← c96555, ← c96556, ← c96557] at e_1577_10
  have e_1577_11 := arithEq_of_rows h (row := 1577) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_11
  simp only [← c96558, ← c96559, k2, ← c96560] at e_1577_11
  have hr := e_1577_11
  simp only [e_1577_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15534)) := by
  have c96561 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96562 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96563 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96564 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96565 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96566 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_12 := arithEq_of_rows h (row := 1577) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_12
  simp only [← c96561, ← c96562, ← c96563] at e_1577_12
  have e_1577_13 := arithEq_of_rows h (row := 1577) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_13
  simp only [← c96564, ← c96565, k2, ← c96566] at e_1577_13
  have hr := e_1577_13
  simp only [e_1577_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15535)) := by
  have c96567 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96568 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96569 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96570 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96571 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96572 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_14 := arithEq_of_rows h (row := 1577) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_14
  simp only [← c96567, ← c96568, ← c96569] at e_1577_14
  have e_1577_15 := arithEq_of_rows h (row := 1577) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_15
  simp only [← c96570, ← c96571, k2, ← c96572] at e_1577_15
  have hr := e_1577_15
  simp only [e_1577_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15536)) := by
  have c96573 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96574 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96575 := (publicBatchWrapper86_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96576 := (publicBatchWrapper86_copies3018 a h).1
  have c96577 := (publicBatchWrapper86_copies3018 a h).2.1
  have c96578 := (publicBatchWrapper86_copies3018 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_16 := arithEq_of_rows h (row := 1577) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_16
  simp only [← c96573, ← c96574, ← c96575] at e_1577_16
  have e_1577_17 := arithEq_of_rows h (row := 1577) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_17
  simp only [← c96576, ← c96577, k2, ← c96578] at e_1577_17
  have hr := e_1577_17
  simp only [e_1577_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1577 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15537)) := by
  have c96579 := (publicBatchWrapper86_copies3018 a h).2.2.2.1
  have c96580 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.1
  have c96581 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.1
  have c96582 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.1
  have c96583 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.1
  have c96584 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1577_18 := arithEq_of_rows h (row := 1577) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_18
  simp only [← c96579, ← c96580, ← c96581] at e_1577_18
  have e_1577_19 := arithEq_of_rows h (row := 1577) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_19
  simp only [← c96582, ← c96583, k2, ← c96584] at e_1577_19
  have hr := e_1577_19
  simp only [e_1577_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15538)) := by
  have c96585 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.1
  have c96586 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96587 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96588 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96589 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96590 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_0 := arithEq_of_rows h (row := 1578) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_0
  simp only [← c96585, ← c96586, ← c96587] at e_1578_0
  have e_1578_1 := arithEq_of_rows h (row := 1578) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_1
  simp only [← c96588, ← c96589, k2, ← c96590] at e_1578_1
  have hr := e_1578_1
  simp only [e_1578_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15539)) := by
  have c96591 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96592 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96593 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96594 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96595 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96596 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_2 := arithEq_of_rows h (row := 1578) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_2
  simp only [← c96591, ← c96592, ← c96593] at e_1578_2
  have e_1578_3 := arithEq_of_rows h (row := 1578) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_3
  simp only [← c96594, ← c96595, k2, ← c96596] at e_1578_3
  have hr := e_1578_3
  simp only [e_1578_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15540)) := by
  have c96597 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96598 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96599 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96600 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96601 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96602 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_4 := arithEq_of_rows h (row := 1578) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_4
  simp only [← c96597, ← c96598, ← c96599] at e_1578_4
  have e_1578_5 := arithEq_of_rows h (row := 1578) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_5
  simp only [← c96600, ← c96601, k2, ← c96602] at e_1578_5
  have hr := e_1578_5
  simp only [e_1578_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15541)) := by
  have c96603 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96604 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96605 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96606 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96607 := (publicBatchWrapper86_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96608 := (publicBatchWrapper86_copies3019 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_6 := arithEq_of_rows h (row := 1578) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_6
  simp only [← c96603, ← c96604, ← c96605] at e_1578_6
  have e_1578_7 := arithEq_of_rows h (row := 1578) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_7
  simp only [← c96606, ← c96607, k2, ← c96608] at e_1578_7
  have hr := e_1578_7
  simp only [e_1578_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15542)) := by
  have c96609 := (publicBatchWrapper86_copies3019 a h).2.1
  have c96610 := (publicBatchWrapper86_copies3019 a h).2.2.1
  have c96611 := (publicBatchWrapper86_copies3019 a h).2.2.2.1
  have c96612 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.1
  have c96613 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.1
  have c96614 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_8 := arithEq_of_rows h (row := 1578) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_8
  simp only [← c96609, ← c96610, ← c96611] at e_1578_8
  have e_1578_9 := arithEq_of_rows h (row := 1578) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_9
  simp only [← c96612, ← c96613, k2, ← c96614] at e_1578_9
  have hr := e_1578_9
  simp only [e_1578_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15543)) := by
  have c96615 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.1
  have c96616 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.1
  have c96617 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.1
  have c96618 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96619 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96620 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_10 := arithEq_of_rows h (row := 1578) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_10
  simp only [← c96615, ← c96616, ← c96617] at e_1578_10
  have e_1578_11 := arithEq_of_rows h (row := 1578) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_11
  simp only [← c96618, ← c96619, k2, ← c96620] at e_1578_11
  have hr := e_1578_11
  simp only [e_1578_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15544)) := by
  have c96621 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96622 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96623 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96624 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96625 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96626 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_12 := arithEq_of_rows h (row := 1578) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_12
  simp only [← c96621, ← c96622, ← c96623] at e_1578_12
  have e_1578_13 := arithEq_of_rows h (row := 1578) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_13
  simp only [← c96624, ← c96625, k2, ← c96626] at e_1578_13
  have hr := e_1578_13
  simp only [e_1578_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15545)) := by
  have c96627 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96628 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96629 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96630 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96631 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96632 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_14 := arithEq_of_rows h (row := 1578) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_14
  simp only [← c96627, ← c96628, ← c96629] at e_1578_14
  have e_1578_15 := arithEq_of_rows h (row := 1578) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_15
  simp only [← c96630, ← c96631, k2, ← c96632] at e_1578_15
  have hr := e_1578_15
  simp only [e_1578_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15546)) := by
  have c96633 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96634 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96635 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96636 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96637 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96638 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_16 := arithEq_of_rows h (row := 1578) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_16
  simp only [← c96633, ← c96634, ← c96635] at e_1578_16
  have e_1578_17 := arithEq_of_rows h (row := 1578) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_17
  simp only [← c96636, ← c96637, k2, ← c96638] at e_1578_17
  have hr := e_1578_17
  simp only [e_1578_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1578 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15547)) := by
  have c96639 := (publicBatchWrapper86_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96640 := (publicBatchWrapper86_copies3020 a h).1
  have c96641 := (publicBatchWrapper86_copies3020 a h).2.1
  have c96642 := (publicBatchWrapper86_copies3020 a h).2.2.1
  have c96643 := (publicBatchWrapper86_copies3020 a h).2.2.2.1
  have c96644 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1578_18 := arithEq_of_rows h (row := 1578) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_18
  simp only [← c96639, ← c96640, ← c96641] at e_1578_18
  have e_1578_19 := arithEq_of_rows h (row := 1578) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_19
  simp only [← c96642, ← c96643, k2, ← c96644] at e_1578_19
  have hr := e_1578_19
  simp only [e_1578_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15548)) := by
  have c96645 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.1
  have c96646 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.1
  have c96647 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.1
  have c96648 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.1
  have c96649 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.1
  have c96650 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_0 := arithEq_of_rows h (row := 1579) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_0
  simp only [← c96645, ← c96646, ← c96647] at e_1579_0
  have e_1579_1 := arithEq_of_rows h (row := 1579) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_1
  simp only [← c96648, ← c96649, k2, ← c96650] at e_1579_1
  have hr := e_1579_1
  simp only [e_1579_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15549)) := by
  have c96651 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96652 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96653 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96654 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96655 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96656 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_2 := arithEq_of_rows h (row := 1579) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_2
  simp only [← c96651, ← c96652, ← c96653] at e_1579_2
  have e_1579_3 := arithEq_of_rows h (row := 1579) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_3
  simp only [← c96654, ← c96655, k2, ← c96656] at e_1579_3
  have hr := e_1579_3
  simp only [e_1579_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15550)) := by
  have c96657 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96658 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96659 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96660 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96661 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96662 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_4 := arithEq_of_rows h (row := 1579) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_4
  simp only [← c96657, ← c96658, ← c96659] at e_1579_4
  have e_1579_5 := arithEq_of_rows h (row := 1579) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_5
  simp only [← c96660, ← c96661, k2, ← c96662] at e_1579_5
  have hr := e_1579_5
  simp only [e_1579_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15551)) := by
  have c96663 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96664 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96665 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96666 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96667 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96668 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_6 := arithEq_of_rows h (row := 1579) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_6
  simp only [← c96663, ← c96664, ← c96665] at e_1579_6
  have e_1579_7 := arithEq_of_rows h (row := 1579) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_7
  simp only [← c96666, ← c96667, k2, ← c96668] at e_1579_7
  have hr := e_1579_7
  simp only [e_1579_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15552)) := by
  have c96669 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96670 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96671 := (publicBatchWrapper86_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96672 := (publicBatchWrapper86_copies3021 a h).1
  have c96673 := (publicBatchWrapper86_copies3021 a h).2.1
  have c96674 := (publicBatchWrapper86_copies3021 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_8 := arithEq_of_rows h (row := 1579) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_8
  simp only [← c96669, ← c96670, ← c96671] at e_1579_8
  have e_1579_9 := arithEq_of_rows h (row := 1579) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_9
  simp only [← c96672, ← c96673, k2, ← c96674] at e_1579_9
  have hr := e_1579_9
  simp only [e_1579_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15553)) := by
  have c96675 := (publicBatchWrapper86_copies3021 a h).2.2.2.1
  have c96676 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.1
  have c96677 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.1
  have c96678 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.1
  have c96679 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.1
  have c96680 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_10 := arithEq_of_rows h (row := 1579) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_10
  simp only [← c96675, ← c96676, ← c96677] at e_1579_10
  have e_1579_11 := arithEq_of_rows h (row := 1579) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_11
  simp only [← c96678, ← c96679, k2, ← c96680] at e_1579_11
  have hr := e_1579_11
  simp only [e_1579_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15554)) := by
  have c96681 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.1
  have c96682 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96683 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96684 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96685 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96686 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_12 := arithEq_of_rows h (row := 1579) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_12
  simp only [← c96681, ← c96682, ← c96683] at e_1579_12
  have e_1579_13 := arithEq_of_rows h (row := 1579) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_13
  simp only [← c96684, ← c96685, k2, ← c96686] at e_1579_13
  have hr := e_1579_13
  simp only [e_1579_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15555)) := by
  have c96687 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96688 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96689 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96690 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96691 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96692 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_14 := arithEq_of_rows h (row := 1579) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_14
  simp only [← c96687, ← c96688, ← c96689] at e_1579_14
  have e_1579_15 := arithEq_of_rows h (row := 1579) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_15
  simp only [← c96690, ← c96691, k2, ← c96692] at e_1579_15
  have hr := e_1579_15
  simp only [e_1579_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15556)) := by
  have c96693 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96694 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96695 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96696 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96697 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96698 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_16 := arithEq_of_rows h (row := 1579) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_16
  simp only [← c96693, ← c96694, ← c96695] at e_1579_16
  have e_1579_17 := arithEq_of_rows h (row := 1579) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_17
  simp only [← c96696, ← c96697, k2, ← c96698] at e_1579_17
  have hr := e_1579_17
  simp only [e_1579_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1579 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15557)) := by
  have c96699 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96700 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96701 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96702 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96703 := (publicBatchWrapper86_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96704 := (publicBatchWrapper86_copies3022 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1579_18 := arithEq_of_rows h (row := 1579) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_18
  simp only [← c96699, ← c96700, ← c96701] at e_1579_18
  have e_1579_19 := arithEq_of_rows h (row := 1579) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_19
  simp only [← c96702, ← c96703, k2, ← c96704] at e_1579_19
  have hr := e_1579_19
  simp only [e_1579_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15558)) := by
  have c96705 := (publicBatchWrapper86_copies3022 a h).2.1
  have c96706 := (publicBatchWrapper86_copies3022 a h).2.2.1
  have c96707 := (publicBatchWrapper86_copies3022 a h).2.2.2.1
  have c96708 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.1
  have c96709 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.1
  have c96710 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_0 := arithEq_of_rows h (row := 1580) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_0
  simp only [← c96705, ← c96706, ← c96707] at e_1580_0
  have e_1580_1 := arithEq_of_rows h (row := 1580) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_1
  simp only [← c96708, ← c96709, k2, ← c96710] at e_1580_1
  have hr := e_1580_1
  simp only [e_1580_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15559)) := by
  have c96711 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.1
  have c96712 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.1
  have c96713 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.1
  have c96714 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96715 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96716 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_2 := arithEq_of_rows h (row := 1580) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_2
  simp only [← c96711, ← c96712, ← c96713] at e_1580_2
  have e_1580_3 := arithEq_of_rows h (row := 1580) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_3
  simp only [← c96714, ← c96715, k2, ← c96716] at e_1580_3
  have hr := e_1580_3
  simp only [e_1580_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 23) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15560)) := by
  have c96717 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96718 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96719 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96720 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96721 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96722 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_4 := arithEq_of_rows h (row := 1580) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_4
  simp only [← c96717, ← c96718, ← c96719] at e_1580_4
  have e_1580_5 := arithEq_of_rows h (row := 1580) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_5
  simp only [← c96720, ← c96721, k2, ← c96722] at e_1580_5
  have hr := e_1580_5
  simp only [e_1580_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 31) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15561)) := by
  have c96723 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96724 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96725 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96726 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96727 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96728 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_6 := arithEq_of_rows h (row := 1580) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_6
  simp only [← c96723, ← c96724, ← c96725] at e_1580_6
  have e_1580_7 := arithEq_of_rows h (row := 1580) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_7
  simp only [← c96726, ← c96727, k2, ← c96728] at e_1580_7
  have hr := e_1580_7
  simp only [e_1580_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 39) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15562)) := by
  have c96729 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96730 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96731 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96732 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96733 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96734 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_8 := arithEq_of_rows h (row := 1580) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_8
  simp only [← c96729, ← c96730, ← c96731] at e_1580_8
  have e_1580_9 := arithEq_of_rows h (row := 1580) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_9
  simp only [← c96732, ← c96733, k2, ← c96734] at e_1580_9
  have hr := e_1580_9
  simp only [e_1580_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 47) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15563)) := by
  have c96735 := (publicBatchWrapper86_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96736 := (publicBatchWrapper86_copies3023 a h).1
  have c96737 := (publicBatchWrapper86_copies3023 a h).2.1
  have c96738 := (publicBatchWrapper86_copies3023 a h).2.2.1
  have c96739 := (publicBatchWrapper86_copies3023 a h).2.2.2.1
  have c96740 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_10 := arithEq_of_rows h (row := 1580) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_10
  simp only [← c96735, ← c96736, ← c96737] at e_1580_10
  have e_1580_11 := arithEq_of_rows h (row := 1580) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_11
  simp only [← c96738, ← c96739, k2, ← c96740] at e_1580_11
  have hr := e_1580_11
  simp only [e_1580_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 55) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15564)) := by
  have c96741 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.1
  have c96742 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.1
  have c96743 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.1
  have c96744 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.1
  have c96745 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.1
  have c96746 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_12 := arithEq_of_rows h (row := 1580) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_12
  simp only [← c96741, ← c96742, ← c96743] at e_1580_12
  have e_1580_13 := arithEq_of_rows h (row := 1580) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_13
  simp only [← c96744, ← c96745, k2, ← c96746] at e_1580_13
  have hr := e_1580_13
  simp only [e_1580_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 63) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15565)) := by
  have c96747 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96748 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96749 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96750 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96751 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96752 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_14 := arithEq_of_rows h (row := 1580) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_14
  simp only [← c96747, ← c96748, ← c96749] at e_1580_14
  have e_1580_15 := arithEq_of_rows h (row := 1580) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_15
  simp only [← c96750, ← c96751, k2, ← c96752] at e_1580_15
  have hr := e_1580_15
  simp only [e_1580_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 71) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15566)) := by
  have c96753 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96754 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96755 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96756 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96757 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96758 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_16 := arithEq_of_rows h (row := 1580) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_16
  simp only [← c96753, ← c96754, ← c96755] at e_1580_16
  have e_1580_17 := arithEq_of_rows h (row := 1580) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_17
  simp only [← c96756, ← c96757, k2, ← c96758] at e_1580_17
  have hr := e_1580_17
  simp only [e_1580_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1580 79) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15567)) := by
  have c96759 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96760 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96761 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96762 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96763 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96764 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1580_18 := arithEq_of_rows h (row := 1580) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_18
  simp only [← c96759, ← c96760, ← c96761] at e_1580_18
  have e_1580_19 := arithEq_of_rows h (row := 1580) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_19
  simp only [← c96762, ← c96763, k2, ← c96764] at e_1580_19
  have hr := e_1580_19
  simp only [e_1580_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 7) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15568)) := by
  have c96765 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96766 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96767 := (publicBatchWrapper86_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96768 := (publicBatchWrapper86_copies3024 a h).1
  have c96769 := (publicBatchWrapper86_copies3024 a h).2.1
  have c96770 := (publicBatchWrapper86_copies3024 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_0 := arithEq_of_rows h (row := 1581) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_0
  simp only [← c96765, ← c96766, ← c96767] at e_1581_0
  have e_1581_1 := arithEq_of_rows h (row := 1581) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_1
  simp only [← c96768, ← c96769, k2, ← c96770] at e_1581_1
  have hr := e_1581_1
  simp only [e_1581_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 15) = bselect (a (.wire 75 55)) (a (.virt 200)) (a (.virt 15569)) := by
  have c96771 := (publicBatchWrapper86_copies3024 a h).2.2.2.1
  have c96772 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.1
  have c96773 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.1
  have c96774 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.1
  have c96775 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.1
  have c96776 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_2 := arithEq_of_rows h (row := 1581) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_2
  simp only [← c96771, ← c96772, ← c96773] at e_1581_2
  have e_1581_3 := arithEq_of_rows h (row := 1581) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_3
  simp only [← c96774, ← c96775, k2, ← c96776] at e_1581_3
  have hr := e_1581_3
  simp only [e_1581_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15610)) := by
  have c96777 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.1
  have c96778 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96779 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96780 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96781 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96782 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_4 := arithEq_of_rows h (row := 1581) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_4
  simp only [← c96777, ← c96778, ← c96779] at e_1581_4
  have e_1581_5 := arithEq_of_rows h (row := 1581) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_5
  simp only [← c96780, ← c96781, k2, ← c96782] at e_1581_5
  have hr := e_1581_5
  simp only [e_1581_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15611)) := by
  have c96783 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96784 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96785 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96786 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96787 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96788 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_6 := arithEq_of_rows h (row := 1581) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_6
  simp only [← c96783, ← c96784, ← c96785] at e_1581_6
  have e_1581_7 := arithEq_of_rows h (row := 1581) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_7
  simp only [← c96786, ← c96787, k2, ← c96788] at e_1581_7
  have hr := e_1581_7
  simp only [e_1581_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15612)) := by
  have c96789 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96790 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96791 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96792 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96793 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96794 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_8 := arithEq_of_rows h (row := 1581) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_8
  simp only [← c96789, ← c96790, ← c96791] at e_1581_8
  have e_1581_9 := arithEq_of_rows h (row := 1581) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_9
  simp only [← c96792, ← c96793, k2, ← c96794] at e_1581_9
  have hr := e_1581_9
  simp only [e_1581_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15613)) := by
  have c96795 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96796 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96797 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96798 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96799 := (publicBatchWrapper86_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96800 := (publicBatchWrapper86_copies3025 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_10 := arithEq_of_rows h (row := 1581) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_10
  simp only [← c96795, ← c96796, ← c96797] at e_1581_10
  have e_1581_11 := arithEq_of_rows h (row := 1581) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_11
  simp only [← c96798, ← c96799, k2, ← c96800] at e_1581_11
  have hr := e_1581_11
  simp only [e_1581_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15614)) := by
  have c96801 := (publicBatchWrapper86_copies3025 a h).2.1
  have c96802 := (publicBatchWrapper86_copies3025 a h).2.2.1
  have c96803 := (publicBatchWrapper86_copies3025 a h).2.2.2.1
  have c96804 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.1
  have c96805 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.1
  have c96806 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_12 := arithEq_of_rows h (row := 1581) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_12
  simp only [← c96801, ← c96802, ← c96803] at e_1581_12
  have e_1581_13 := arithEq_of_rows h (row := 1581) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_13
  simp only [← c96804, ← c96805, k2, ← c96806] at e_1581_13
  have hr := e_1581_13
  simp only [e_1581_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15615)) := by
  have c96807 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.1
  have c96808 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.1
  have c96809 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.1
  have c96810 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96811 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96812 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_14 := arithEq_of_rows h (row := 1581) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_14
  simp only [← c96807, ← c96808, ← c96809] at e_1581_14
  have e_1581_15 := arithEq_of_rows h (row := 1581) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_15
  simp only [← c96810, ← c96811, k2, ← c96812] at e_1581_15
  have hr := e_1581_15
  simp only [e_1581_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15616)) := by
  have c96813 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96814 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96815 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96816 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96817 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96818 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_16 := arithEq_of_rows h (row := 1581) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_16
  simp only [← c96813, ← c96814, ← c96815] at e_1581_16
  have e_1581_17 := arithEq_of_rows h (row := 1581) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_17
  simp only [← c96816, ← c96817, k2, ← c96818] at e_1581_17
  have hr := e_1581_17
  simp only [e_1581_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1581 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15617)) := by
  have c96819 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96820 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96821 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96822 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96823 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96824 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1581_18 := arithEq_of_rows h (row := 1581) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_18
  simp only [← c96819, ← c96820, ← c96821] at e_1581_18
  have e_1581_19 := arithEq_of_rows h (row := 1581) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_19
  simp only [← c96822, ← c96823, k2, ← c96824] at e_1581_19
  have hr := e_1581_19
  simp only [e_1581_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15618)) := by
  have c96825 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96826 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96827 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96828 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96829 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96830 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_0 := arithEq_of_rows h (row := 1582) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_0
  simp only [← c96825, ← c96826, ← c96827] at e_1582_0
  have e_1582_1 := arithEq_of_rows h (row := 1582) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_1
  simp only [← c96828, ← c96829, k2, ← c96830] at e_1582_1
  have hr := e_1582_1
  simp only [e_1582_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15619)) := by
  have c96831 := (publicBatchWrapper86_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96832 := (publicBatchWrapper86_copies3026 a h).1
  have c96833 := (publicBatchWrapper86_copies3026 a h).2.1
  have c96834 := (publicBatchWrapper86_copies3026 a h).2.2.1
  have c96835 := (publicBatchWrapper86_copies3026 a h).2.2.2.1
  have c96836 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_2 := arithEq_of_rows h (row := 1582) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_2
  simp only [← c96831, ← c96832, ← c96833] at e_1582_2
  have e_1582_3 := arithEq_of_rows h (row := 1582) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_3
  simp only [← c96834, ← c96835, k2, ← c96836] at e_1582_3
  have hr := e_1582_3
  simp only [e_1582_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15620)) := by
  have c96837 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.1
  have c96838 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.1
  have c96839 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.1
  have c96840 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.1
  have c96841 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.1
  have c96842 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_4 := arithEq_of_rows h (row := 1582) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_4
  simp only [← c96837, ← c96838, ← c96839] at e_1582_4
  have e_1582_5 := arithEq_of_rows h (row := 1582) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_5
  simp only [← c96840, ← c96841, k2, ← c96842] at e_1582_5
  have hr := e_1582_5
  simp only [e_1582_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15621)) := by
  have c96843 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96844 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96845 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96846 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96847 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96848 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_6 := arithEq_of_rows h (row := 1582) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_6
  simp only [← c96843, ← c96844, ← c96845] at e_1582_6
  have e_1582_7 := arithEq_of_rows h (row := 1582) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_7
  simp only [← c96846, ← c96847, k2, ← c96848] at e_1582_7
  have hr := e_1582_7
  simp only [e_1582_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15622)) := by
  have c96849 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96850 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96851 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96852 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96853 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96854 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_8 := arithEq_of_rows h (row := 1582) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_8
  simp only [← c96849, ← c96850, ← c96851] at e_1582_8
  have e_1582_9 := arithEq_of_rows h (row := 1582) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_9
  simp only [← c96852, ← c96853, k2, ← c96854] at e_1582_9
  have hr := e_1582_9
  simp only [e_1582_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15623)) := by
  have c96855 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96856 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96857 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96858 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96859 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96860 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_10 := arithEq_of_rows h (row := 1582) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_10
  simp only [← c96855, ← c96856, ← c96857] at e_1582_10
  have e_1582_11 := arithEq_of_rows h (row := 1582) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_11
  simp only [← c96858, ← c96859, k2, ← c96860] at e_1582_11
  have hr := e_1582_11
  simp only [e_1582_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15624)) := by
  have c96861 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96862 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96863 := (publicBatchWrapper86_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96864 := (publicBatchWrapper86_copies3027 a h).1
  have c96865 := (publicBatchWrapper86_copies3027 a h).2.1
  have c96866 := (publicBatchWrapper86_copies3027 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_12 := arithEq_of_rows h (row := 1582) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_12
  simp only [← c96861, ← c96862, ← c96863] at e_1582_12
  have e_1582_13 := arithEq_of_rows h (row := 1582) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_13
  simp only [← c96864, ← c96865, k2, ← c96866] at e_1582_13
  have hr := e_1582_13
  simp only [e_1582_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15625)) := by
  have c96867 := (publicBatchWrapper86_copies3027 a h).2.2.2.1
  have c96868 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.1
  have c96869 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.1
  have c96870 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.1
  have c96871 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.1
  have c96872 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_14 := arithEq_of_rows h (row := 1582) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_14
  simp only [← c96867, ← c96868, ← c96869] at e_1582_14
  have e_1582_15 := arithEq_of_rows h (row := 1582) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_15
  simp only [← c96870, ← c96871, k2, ← c96872] at e_1582_15
  have hr := e_1582_15
  simp only [e_1582_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15626)) := by
  have c96873 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.1
  have c96874 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96875 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96876 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96877 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96878 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_16 := arithEq_of_rows h (row := 1582) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_16
  simp only [← c96873, ← c96874, ← c96875] at e_1582_16
  have e_1582_17 := arithEq_of_rows h (row := 1582) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_17
  simp only [← c96876, ← c96877, k2, ← c96878] at e_1582_17
  have hr := e_1582_17
  simp only [e_1582_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1582 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15627)) := by
  have c96879 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96880 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96881 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96882 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96883 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96884 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1582_18 := arithEq_of_rows h (row := 1582) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_18
  simp only [← c96879, ← c96880, ← c96881] at e_1582_18
  have e_1582_19 := arithEq_of_rows h (row := 1582) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_19
  simp only [← c96882, ← c96883, k2, ← c96884] at e_1582_19
  have hr := e_1582_19
  simp only [e_1582_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15628)) := by
  have c96885 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96886 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96887 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96888 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96889 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96890 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_0 := arithEq_of_rows h (row := 1583) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_0
  simp only [← c96885, ← c96886, ← c96887] at e_1583_0
  have e_1583_1 := arithEq_of_rows h (row := 1583) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_1
  simp only [← c96888, ← c96889, k2, ← c96890] at e_1583_1
  have hr := e_1583_1
  simp only [e_1583_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15629)) := by
  have c96891 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96892 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96893 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96894 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96895 := (publicBatchWrapper86_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96896 := (publicBatchWrapper86_copies3028 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_2 := arithEq_of_rows h (row := 1583) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_2
  simp only [← c96891, ← c96892, ← c96893] at e_1583_2
  have e_1583_3 := arithEq_of_rows h (row := 1583) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_3
  simp only [← c96894, ← c96895, k2, ← c96896] at e_1583_3
  have hr := e_1583_3
  simp only [e_1583_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15630)) := by
  have c96897 := (publicBatchWrapper86_copies3028 a h).2.1
  have c96898 := (publicBatchWrapper86_copies3028 a h).2.2.1
  have c96899 := (publicBatchWrapper86_copies3028 a h).2.2.2.1
  have c96900 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.1
  have c96901 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.1
  have c96902 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_4 := arithEq_of_rows h (row := 1583) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_4
  simp only [← c96897, ← c96898, ← c96899] at e_1583_4
  have e_1583_5 := arithEq_of_rows h (row := 1583) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_5
  simp only [← c96900, ← c96901, k2, ← c96902] at e_1583_5
  have hr := e_1583_5
  simp only [e_1583_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15631)) := by
  have c96903 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.1
  have c96904 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.1
  have c96905 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.1
  have c96906 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96907 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96908 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_6 := arithEq_of_rows h (row := 1583) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_6
  simp only [← c96903, ← c96904, ← c96905] at e_1583_6
  have e_1583_7 := arithEq_of_rows h (row := 1583) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_7
  simp only [← c96906, ← c96907, k2, ← c96908] at e_1583_7
  have hr := e_1583_7
  simp only [e_1583_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15632)) := by
  have c96909 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96910 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96911 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96912 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96913 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96914 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_8 := arithEq_of_rows h (row := 1583) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_8
  simp only [← c96909, ← c96910, ← c96911] at e_1583_8
  have e_1583_9 := arithEq_of_rows h (row := 1583) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_9
  simp only [← c96912, ← c96913, k2, ← c96914] at e_1583_9
  have hr := e_1583_9
  simp only [e_1583_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15633)) := by
  have c96915 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96916 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96917 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96918 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96919 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96920 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_10 := arithEq_of_rows h (row := 1583) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_10
  simp only [← c96915, ← c96916, ← c96917] at e_1583_10
  have e_1583_11 := arithEq_of_rows h (row := 1583) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_11
  simp only [← c96918, ← c96919, k2, ← c96920] at e_1583_11
  have hr := e_1583_11
  simp only [e_1583_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15634)) := by
  have c96921 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96922 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96923 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96924 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96925 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96926 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_12 := arithEq_of_rows h (row := 1583) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_12
  simp only [← c96921, ← c96922, ← c96923] at e_1583_12
  have e_1583_13 := arithEq_of_rows h (row := 1583) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_13
  simp only [← c96924, ← c96925, k2, ← c96926] at e_1583_13
  have hr := e_1583_13
  simp only [e_1583_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15635)) := by
  have c96927 := (publicBatchWrapper86_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96928 := (publicBatchWrapper86_copies3029 a h).1
  have c96929 := (publicBatchWrapper86_copies3029 a h).2.1
  have c96930 := (publicBatchWrapper86_copies3029 a h).2.2.1
  have c96931 := (publicBatchWrapper86_copies3029 a h).2.2.2.1
  have c96932 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_14 := arithEq_of_rows h (row := 1583) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_14
  simp only [← c96927, ← c96928, ← c96929] at e_1583_14
  have e_1583_15 := arithEq_of_rows h (row := 1583) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_15
  simp only [← c96930, ← c96931, k2, ← c96932] at e_1583_15
  have hr := e_1583_15
  simp only [e_1583_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15636)) := by
  have c96933 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.1
  have c96934 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.1
  have c96935 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.1
  have c96936 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.1
  have c96937 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.1
  have c96938 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_16 := arithEq_of_rows h (row := 1583) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_16
  simp only [← c96933, ← c96934, ← c96935] at e_1583_16
  have e_1583_17 := arithEq_of_rows h (row := 1583) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_17
  simp only [← c96936, ← c96937, k2, ← c96938] at e_1583_17
  have hr := e_1583_17
  simp only [e_1583_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1583 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15637)) := by
  have c96939 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96940 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96941 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96942 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96943 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96944 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1583_18 := arithEq_of_rows h (row := 1583) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_18
  simp only [← c96939, ← c96940, ← c96941] at e_1583_18
  have e_1583_19 := arithEq_of_rows h (row := 1583) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_19
  simp only [← c96942, ← c96943, k2, ← c96944] at e_1583_19
  have hr := e_1583_19
  simp only [e_1583_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15638)) := by
  have c96945 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96946 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96947 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96948 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96949 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96950 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_0 := arithEq_of_rows h (row := 1584) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_0
  simp only [← c96945, ← c96946, ← c96947] at e_1584_0
  have e_1584_1 := arithEq_of_rows h (row := 1584) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_1
  simp only [← c96948, ← c96949, k2, ← c96950] at e_1584_1
  have hr := e_1584_1
  simp only [e_1584_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15639)) := by
  have c96951 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96952 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96953 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96954 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96955 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96956 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_2 := arithEq_of_rows h (row := 1584) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_2
  simp only [← c96951, ← c96952, ← c96953] at e_1584_2
  have e_1584_3 := arithEq_of_rows h (row := 1584) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_3
  simp only [← c96954, ← c96955, k2, ← c96956] at e_1584_3
  have hr := e_1584_3
  simp only [e_1584_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15640)) := by
  have c96957 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96958 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96959 := (publicBatchWrapper86_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96960 := (publicBatchWrapper86_copies3030 a h).1
  have c96961 := (publicBatchWrapper86_copies3030 a h).2.1
  have c96962 := (publicBatchWrapper86_copies3030 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_4 := arithEq_of_rows h (row := 1584) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_4
  simp only [← c96957, ← c96958, ← c96959] at e_1584_4
  have e_1584_5 := arithEq_of_rows h (row := 1584) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_5
  simp only [← c96960, ← c96961, k2, ← c96962] at e_1584_5
  have hr := e_1584_5
  simp only [e_1584_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15641)) := by
  have c96963 := (publicBatchWrapper86_copies3030 a h).2.2.2.1
  have c96964 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.1
  have c96965 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.1
  have c96966 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.1
  have c96967 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.1
  have c96968 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_6 := arithEq_of_rows h (row := 1584) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_6
  simp only [← c96963, ← c96964, ← c96965] at e_1584_6
  have e_1584_7 := arithEq_of_rows h (row := 1584) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_7
  simp only [← c96966, ← c96967, k2, ← c96968] at e_1584_7
  have hr := e_1584_7
  simp only [e_1584_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15642)) := by
  have c96969 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.1
  have c96970 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96971 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96972 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96973 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96974 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_8 := arithEq_of_rows h (row := 1584) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_8
  simp only [← c96969, ← c96970, ← c96971] at e_1584_8
  have e_1584_9 := arithEq_of_rows h (row := 1584) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_9
  simp only [← c96972, ← c96973, k2, ← c96974] at e_1584_9
  have hr := e_1584_9
  simp only [e_1584_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15643)) := by
  have c96975 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96976 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96977 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96978 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96979 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96980 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_10 := arithEq_of_rows h (row := 1584) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_10
  simp only [← c96975, ← c96976, ← c96977] at e_1584_10
  have e_1584_11 := arithEq_of_rows h (row := 1584) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_11
  simp only [← c96978, ← c96979, k2, ← c96980] at e_1584_11
  have hr := e_1584_11
  simp only [e_1584_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15644)) := by
  have c96981 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96982 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96983 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96984 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96985 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96986 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_12 := arithEq_of_rows h (row := 1584) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_12
  simp only [← c96981, ← c96982, ← c96983] at e_1584_12
  have e_1584_13 := arithEq_of_rows h (row := 1584) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_13
  simp only [← c96984, ← c96985, k2, ← c96986] at e_1584_13
  have hr := e_1584_13
  simp only [e_1584_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 63) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15645)) := by
  have c96987 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96988 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96989 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96990 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96991 := (publicBatchWrapper86_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96992 := (publicBatchWrapper86_copies3031 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_14 := arithEq_of_rows h (row := 1584) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_14
  simp only [← c96987, ← c96988, ← c96989] at e_1584_14
  have e_1584_15 := arithEq_of_rows h (row := 1584) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_15
  simp only [← c96990, ← c96991, k2, ← c96992] at e_1584_15
  have hr := e_1584_15
  simp only [e_1584_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 71) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15646)) := by
  have c96993 := (publicBatchWrapper86_copies3031 a h).2.1
  have c96994 := (publicBatchWrapper86_copies3031 a h).2.2.1
  have c96995 := (publicBatchWrapper86_copies3031 a h).2.2.2.1
  have c96996 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.1
  have c96997 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.1
  have c96998 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_16 := arithEq_of_rows h (row := 1584) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_16
  simp only [← c96993, ← c96994, ← c96995] at e_1584_16
  have e_1584_17 := arithEq_of_rows h (row := 1584) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_17
  simp only [← c96996, ← c96997, k2, ← c96998] at e_1584_17
  have hr := e_1584_17
  simp only [e_1584_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1584 79) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15647)) := by
  have c96999 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.1
  have c97000 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.1
  have c97001 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.1
  have c97002 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97003 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97004 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1584_18 := arithEq_of_rows h (row := 1584) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_18
  simp only [← c96999, ← c97000, ← c97001] at e_1584_18
  have e_1584_19 := arithEq_of_rows h (row := 1584) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_19
  simp only [← c97002, ← c97003, k2, ← c97004] at e_1584_19
  have hr := e_1584_19
  simp only [e_1584_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 7) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15648)) := by
  have c97005 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97006 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97007 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97008 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97009 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97010 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_0 := arithEq_of_rows h (row := 1585) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_0
  simp only [← c97005, ← c97006, ← c97007] at e_1585_0
  have e_1585_1 := arithEq_of_rows h (row := 1585) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_1
  simp only [← c97008, ← c97009, k2, ← c97010] at e_1585_1
  have hr := e_1585_1
  simp only [e_1585_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 15) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15649)) := by
  have c97011 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97012 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97013 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97014 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97015 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97016 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_2 := arithEq_of_rows h (row := 1585) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_2
  simp only [← c97011, ← c97012, ← c97013] at e_1585_2
  have e_1585_3 := arithEq_of_rows h (row := 1585) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_3
  simp only [← c97014, ← c97015, k2, ← c97016] at e_1585_3
  have hr := e_1585_3
  simp only [e_1585_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 23) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15650)) := by
  have c97017 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97018 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97019 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97020 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97021 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97022 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_4 := arithEq_of_rows h (row := 1585) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_4
  simp only [← c97017, ← c97018, ← c97019] at e_1585_4
  have e_1585_5 := arithEq_of_rows h (row := 1585) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_5
  simp only [← c97020, ← c97021, k2, ← c97022] at e_1585_5
  have hr := e_1585_5
  simp only [e_1585_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 31) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15651)) := by
  have c97023 := (publicBatchWrapper86_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97024 := (publicBatchWrapper86_copies3032 a h).1
  have c97025 := (publicBatchWrapper86_copies3032 a h).2.1
  have c97026 := (publicBatchWrapper86_copies3032 a h).2.2.1
  have c97027 := (publicBatchWrapper86_copies3032 a h).2.2.2.1
  have c97028 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_6 := arithEq_of_rows h (row := 1585) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_6
  simp only [← c97023, ← c97024, ← c97025] at e_1585_6
  have e_1585_7 := arithEq_of_rows h (row := 1585) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_7
  simp only [← c97026, ← c97027, k2, ← c97028] at e_1585_7
  have hr := e_1585_7
  simp only [e_1585_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 39) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15652)) := by
  have c97029 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.1
  have c97030 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.1
  have c97031 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.1
  have c97032 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.1
  have c97033 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.1
  have c97034 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_8 := arithEq_of_rows h (row := 1585) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_8
  simp only [← c97029, ← c97030, ← c97031] at e_1585_8
  have e_1585_9 := arithEq_of_rows h (row := 1585) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_9
  simp only [← c97032, ← c97033, k2, ← c97034] at e_1585_9
  have hr := e_1585_9
  simp only [e_1585_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 47) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15653)) := by
  have c97035 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97036 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97037 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97038 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97039 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97040 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_10 := arithEq_of_rows h (row := 1585) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_10
  simp only [← c97035, ← c97036, ← c97037] at e_1585_10
  have e_1585_11 := arithEq_of_rows h (row := 1585) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_11
  simp only [← c97038, ← c97039, k2, ← c97040] at e_1585_11
  have hr := e_1585_11
  simp only [e_1585_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f15448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1585 55) = bselect (a (.wire 77 19)) (a (.virt 200)) (a (.virt 15654)) := by
  have c97041 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97042 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97043 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97044 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97045 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97046 := (publicBatchWrapper86_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1585_12 := arithEq_of_rows h (row := 1585) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_12
  simp only [← c97041, ← c97042, ← c97043] at e_1585_12
  have e_1585_13 := arithEq_of_rows h (row := 1585) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_13
  simp only [← c97044, ← c97045, k2, ← c97046] at e_1585_13
  have hr := e_1585_13
  simp only [e_1585_12] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
