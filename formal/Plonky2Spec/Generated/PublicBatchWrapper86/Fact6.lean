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

theorem publicBatchWrapper86_f4362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 477 7) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1768)) := by
  have c30525 := (publicBatchWrapper86_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30526 := (publicBatchWrapper86_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30527 := (publicBatchWrapper86_copies953 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30528 := (publicBatchWrapper86_copies954 a h).1
  have c30529 := (publicBatchWrapper86_copies954 a h).2.1
  have c30530 := (publicBatchWrapper86_copies954 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_477_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_0
  simp only [← c30525, ← c30526, ← c30527] at e_477_0
  have e_477_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_1
  simp only [← c30528, ← c30529, k2, ← c30530] at e_477_1
  have hr := e_477_1
  simp only [e_477_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 477 15) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1769)) := by
  have c30531 := (publicBatchWrapper86_copies954 a h).2.2.2.1
  have c30532 := (publicBatchWrapper86_copies954 a h).2.2.2.2.1
  have c30533 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.1
  have c30534 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.1
  have c30535 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.1
  have c30536 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_477_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_2
  simp only [← c30531, ← c30532, ← c30533] at e_477_2
  have e_477_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_3
  simp only [← c30534, ← c30535, k2, ← c30536] at e_477_3
  have hr := e_477_3
  simp only [e_477_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 477 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1810)) := by
  have c30537 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.1
  have c30538 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30539 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30540 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30541 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30542 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_477_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_4
  simp only [← c30537, ← c30538, ← c30539] at e_477_4
  have e_477_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_5
  simp only [← c30540, ← c30541, k2, ← c30542] at e_477_5
  have hr := e_477_5
  simp only [e_477_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 477 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1811)) := by
  have c30543 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30544 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30545 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30546 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30547 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30548 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_477_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_6
  simp only [← c30543, ← c30544, ← c30545] at e_477_6
  have e_477_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_7
  simp only [← c30546, ← c30547, k2, ← c30548] at e_477_7
  have hr := e_477_7
  simp only [e_477_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 477 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1812)) := by
  have c30549 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30550 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30551 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30552 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30553 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30554 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_477_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_8
  simp only [← c30549, ← c30550, ← c30551] at e_477_8
  have e_477_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_9
  simp only [← c30552, ← c30553, k2, ← c30554] at e_477_9
  have hr := e_477_9
  simp only [e_477_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 477 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1813)) := by
  have c30555 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30556 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30557 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30558 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30559 := (publicBatchWrapper86_copies954 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30560 := (publicBatchWrapper86_copies955 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_477_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_10
  simp only [← c30555, ← c30556, ← c30557] at e_477_10
  have e_477_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_11
  simp only [← c30558, ← c30559, k2, ← c30560] at e_477_11
  have hr := e_477_11
  simp only [e_477_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 477 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1814)) := by
  have c30561 := (publicBatchWrapper86_copies955 a h).2.1
  have c30562 := (publicBatchWrapper86_copies955 a h).2.2.1
  have c30563 := (publicBatchWrapper86_copies955 a h).2.2.2.1
  have c30564 := (publicBatchWrapper86_copies955 a h).2.2.2.2.1
  have c30565 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.1
  have c30566 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_477_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_12
  simp only [← c30561, ← c30562, ← c30563] at e_477_12
  have e_477_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_13
  simp only [← c30564, ← c30565, k2, ← c30566] at e_477_13
  have hr := e_477_13
  simp only [e_477_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 477 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1815)) := by
  have c30567 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.1
  have c30568 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.1
  have c30569 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.1
  have c30570 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30571 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30572 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_477_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_14
  simp only [← c30567, ← c30568, ← c30569] at e_477_14
  have e_477_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_15
  simp only [← c30570, ← c30571, k2, ← c30572] at e_477_15
  have hr := e_477_15
  simp only [e_477_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 477 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1816)) := by
  have c30573 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30574 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30575 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30576 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30577 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30578 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_477_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_16
  simp only [← c30573, ← c30574, ← c30575] at e_477_16
  have e_477_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_17
  simp only [← c30576, ← c30577, k2, ← c30578] at e_477_17
  have hr := e_477_17
  simp only [e_477_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 477 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1817)) := by
  have c30579 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30580 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30581 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30582 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30583 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30584 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_477_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_18
  simp only [← c30579, ← c30580, ← c30581] at e_477_18
  have e_477_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row477 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_477_19
  simp only [← c30582, ← c30583, k2, ← c30584] at e_477_19
  have hr := e_477_19
  simp only [e_477_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 478 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1818)) := by
  have c30585 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30586 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30587 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30588 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30589 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30590 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_478_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_0
  simp only [← c30585, ← c30586, ← c30587] at e_478_0
  have e_478_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_1
  simp only [← c30588, ← c30589, k2, ← c30590] at e_478_1
  have hr := e_478_1
  simp only [e_478_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 478 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1819)) := by
  have c30591 := (publicBatchWrapper86_copies955 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30592 := (publicBatchWrapper86_copies956 a h).1
  have c30593 := (publicBatchWrapper86_copies956 a h).2.1
  have c30594 := (publicBatchWrapper86_copies956 a h).2.2.1
  have c30595 := (publicBatchWrapper86_copies956 a h).2.2.2.1
  have c30596 := (publicBatchWrapper86_copies956 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_478_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_2
  simp only [← c30591, ← c30592, ← c30593] at e_478_2
  have e_478_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_3
  simp only [← c30594, ← c30595, k2, ← c30596] at e_478_3
  have hr := e_478_3
  simp only [e_478_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 478 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1820)) := by
  have c30597 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.1
  have c30598 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.1
  have c30599 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.1
  have c30600 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.1
  have c30601 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.1
  have c30602 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_478_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_4
  simp only [← c30597, ← c30598, ← c30599] at e_478_4
  have e_478_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_5
  simp only [← c30600, ← c30601, k2, ← c30602] at e_478_5
  have hr := e_478_5
  simp only [e_478_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 478 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1821)) := by
  have c30603 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30604 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30605 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30606 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30607 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30608 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_478_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_6
  simp only [← c30603, ← c30604, ← c30605] at e_478_6
  have e_478_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_7
  simp only [← c30606, ← c30607, k2, ← c30608] at e_478_7
  have hr := e_478_7
  simp only [e_478_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 478 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1822)) := by
  have c30609 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30610 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30611 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30612 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30613 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30614 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_478_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_8
  simp only [← c30609, ← c30610, ← c30611] at e_478_8
  have e_478_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_9
  simp only [← c30612, ← c30613, k2, ← c30614] at e_478_9
  have hr := e_478_9
  simp only [e_478_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 478 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1823)) := by
  have c30615 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30616 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30617 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30618 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30619 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30620 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_478_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_10
  simp only [← c30615, ← c30616, ← c30617] at e_478_10
  have e_478_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_11
  simp only [← c30618, ← c30619, k2, ← c30620] at e_478_11
  have hr := e_478_11
  simp only [e_478_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 478 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1824)) := by
  have c30621 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30622 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30623 := (publicBatchWrapper86_copies956 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30624 := (publicBatchWrapper86_copies957 a h).1
  have c30625 := (publicBatchWrapper86_copies957 a h).2.1
  have c30626 := (publicBatchWrapper86_copies957 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_478_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_12
  simp only [← c30621, ← c30622, ← c30623] at e_478_12
  have e_478_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_13
  simp only [← c30624, ← c30625, k2, ← c30626] at e_478_13
  have hr := e_478_13
  simp only [e_478_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 478 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1825)) := by
  have c30627 := (publicBatchWrapper86_copies957 a h).2.2.2.1
  have c30628 := (publicBatchWrapper86_copies957 a h).2.2.2.2.1
  have c30629 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.1
  have c30630 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.1
  have c30631 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.1
  have c30632 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_478_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_14
  simp only [← c30627, ← c30628, ← c30629] at e_478_14
  have e_478_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_15
  simp only [← c30630, ← c30631, k2, ← c30632] at e_478_15
  have hr := e_478_15
  simp only [e_478_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 478 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1826)) := by
  have c30633 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.1
  have c30634 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30635 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30636 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30637 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30638 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_478_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_16
  simp only [← c30633, ← c30634, ← c30635] at e_478_16
  have e_478_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_17
  simp only [← c30636, ← c30637, k2, ← c30638] at e_478_17
  have hr := e_478_17
  simp only [e_478_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 478 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1827)) := by
  have c30639 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30640 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30641 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30642 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30643 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30644 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_478_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_18
  simp only [← c30639, ← c30640, ← c30641] at e_478_18
  have e_478_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row478 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_478_19
  simp only [← c30642, ← c30643, k2, ← c30644] at e_478_19
  have hr := e_478_19
  simp only [e_478_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 479 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1828)) := by
  have c30645 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30646 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30647 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30648 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30649 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30650 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_479_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_0
  simp only [← c30645, ← c30646, ← c30647] at e_479_0
  have e_479_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_1
  simp only [← c30648, ← c30649, k2, ← c30650] at e_479_1
  have hr := e_479_1
  simp only [e_479_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 479 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1829)) := by
  have c30651 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30652 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30653 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30654 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30655 := (publicBatchWrapper86_copies957 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30656 := (publicBatchWrapper86_copies958 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_479_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_2
  simp only [← c30651, ← c30652, ← c30653] at e_479_2
  have e_479_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_3
  simp only [← c30654, ← c30655, k2, ← c30656] at e_479_3
  have hr := e_479_3
  simp only [e_479_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 479 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1830)) := by
  have c30657 := (publicBatchWrapper86_copies958 a h).2.1
  have c30658 := (publicBatchWrapper86_copies958 a h).2.2.1
  have c30659 := (publicBatchWrapper86_copies958 a h).2.2.2.1
  have c30660 := (publicBatchWrapper86_copies958 a h).2.2.2.2.1
  have c30661 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.1
  have c30662 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_479_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_4
  simp only [← c30657, ← c30658, ← c30659] at e_479_4
  have e_479_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_5
  simp only [← c30660, ← c30661, k2, ← c30662] at e_479_5
  have hr := e_479_5
  simp only [e_479_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 479 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1831)) := by
  have c30663 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.1
  have c30664 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.1
  have c30665 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.1
  have c30666 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30667 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30668 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_479_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_6
  simp only [← c30663, ← c30664, ← c30665] at e_479_6
  have e_479_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_7
  simp only [← c30666, ← c30667, k2, ← c30668] at e_479_7
  have hr := e_479_7
  simp only [e_479_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 479 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1832)) := by
  have c30669 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30670 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30671 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30672 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30673 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30674 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_479_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_8
  simp only [← c30669, ← c30670, ← c30671] at e_479_8
  have e_479_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_9
  simp only [← c30672, ← c30673, k2, ← c30674] at e_479_9
  have hr := e_479_9
  simp only [e_479_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 479 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1833)) := by
  have c30675 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30676 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30677 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30678 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30679 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30680 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_479_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_10
  simp only [← c30675, ← c30676, ← c30677] at e_479_10
  have e_479_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_11
  simp only [← c30678, ← c30679, k2, ← c30680] at e_479_11
  have hr := e_479_11
  simp only [e_479_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 479 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1834)) := by
  have c30681 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30682 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30683 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30684 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30685 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30686 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_479_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_12
  simp only [← c30681, ← c30682, ← c30683] at e_479_12
  have e_479_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_13
  simp only [← c30684, ← c30685, k2, ← c30686] at e_479_13
  have hr := e_479_13
  simp only [e_479_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 479 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1835)) := by
  have c30687 := (publicBatchWrapper86_copies958 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30688 := (publicBatchWrapper86_copies959 a h).1
  have c30689 := (publicBatchWrapper86_copies959 a h).2.1
  have c30690 := (publicBatchWrapper86_copies959 a h).2.2.1
  have c30691 := (publicBatchWrapper86_copies959 a h).2.2.2.1
  have c30692 := (publicBatchWrapper86_copies959 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_479_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_14
  simp only [← c30687, ← c30688, ← c30689] at e_479_14
  have e_479_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_15
  simp only [← c30690, ← c30691, k2, ← c30692] at e_479_15
  have hr := e_479_15
  simp only [e_479_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 479 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1836)) := by
  have c30693 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.1
  have c30694 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.1
  have c30695 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.1
  have c30696 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.1
  have c30697 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.1
  have c30698 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_479_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_16
  simp only [← c30693, ← c30694, ← c30695] at e_479_16
  have e_479_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_17
  simp only [← c30696, ← c30697, k2, ← c30698] at e_479_17
  have hr := e_479_17
  simp only [e_479_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 479 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1837)) := by
  have c30699 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30700 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30701 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30702 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30703 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30704 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_479_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_18
  simp only [← c30699, ← c30700, ← c30701] at e_479_18
  have e_479_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row479 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_479_19
  simp only [← c30702, ← c30703, k2, ← c30704] at e_479_19
  have hr := e_479_19
  simp only [e_479_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 480 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1838)) := by
  have c30705 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30706 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30707 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30708 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30709 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30710 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_480_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_0
  simp only [← c30705, ← c30706, ← c30707] at e_480_0
  have e_480_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_1
  simp only [← c30708, ← c30709, k2, ← c30710] at e_480_1
  have hr := e_480_1
  simp only [e_480_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 480 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1839)) := by
  have c30711 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30712 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30713 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30714 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30715 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30716 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_480_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_2
  simp only [← c30711, ← c30712, ← c30713] at e_480_2
  have e_480_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_3
  simp only [← c30714, ← c30715, k2, ← c30716] at e_480_3
  have hr := e_480_3
  simp only [e_480_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 480 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1840)) := by
  have c30717 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30718 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30719 := (publicBatchWrapper86_copies959 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30720 := (publicBatchWrapper86_copies960 a h).1
  have c30721 := (publicBatchWrapper86_copies960 a h).2.1
  have c30722 := (publicBatchWrapper86_copies960 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_480_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_4
  simp only [← c30717, ← c30718, ← c30719] at e_480_4
  have e_480_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_5
  simp only [← c30720, ← c30721, k2, ← c30722] at e_480_5
  have hr := e_480_5
  simp only [e_480_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 480 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1841)) := by
  have c30723 := (publicBatchWrapper86_copies960 a h).2.2.2.1
  have c30724 := (publicBatchWrapper86_copies960 a h).2.2.2.2.1
  have c30725 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.1
  have c30726 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.1
  have c30727 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.1
  have c30728 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_480_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_6
  simp only [← c30723, ← c30724, ← c30725] at e_480_6
  have e_480_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_7
  simp only [← c30726, ← c30727, k2, ← c30728] at e_480_7
  have hr := e_480_7
  simp only [e_480_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 480 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1842)) := by
  have c30729 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.1
  have c30730 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30731 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30732 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30733 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30734 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_480_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_8
  simp only [← c30729, ← c30730, ← c30731] at e_480_8
  have e_480_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_9
  simp only [← c30732, ← c30733, k2, ← c30734] at e_480_9
  have hr := e_480_9
  simp only [e_480_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 480 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1843)) := by
  have c30735 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30736 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30737 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30738 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30739 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30740 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_480_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_10
  simp only [← c30735, ← c30736, ← c30737] at e_480_10
  have e_480_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_11
  simp only [← c30738, ← c30739, k2, ← c30740] at e_480_11
  have hr := e_480_11
  simp only [e_480_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 480 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1844)) := by
  have c30741 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30742 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30743 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30744 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30745 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30746 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_480_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_12
  simp only [← c30741, ← c30742, ← c30743] at e_480_12
  have e_480_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_13
  simp only [← c30744, ← c30745, k2, ← c30746] at e_480_13
  have hr := e_480_13
  simp only [e_480_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 480 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1845)) := by
  have c30747 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30748 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30749 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30750 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30751 := (publicBatchWrapper86_copies960 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30752 := (publicBatchWrapper86_copies961 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_480_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_14
  simp only [← c30747, ← c30748, ← c30749] at e_480_14
  have e_480_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_15
  simp only [← c30750, ← c30751, k2, ← c30752] at e_480_15
  have hr := e_480_15
  simp only [e_480_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 480 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1846)) := by
  have c30753 := (publicBatchWrapper86_copies961 a h).2.1
  have c30754 := (publicBatchWrapper86_copies961 a h).2.2.1
  have c30755 := (publicBatchWrapper86_copies961 a h).2.2.2.1
  have c30756 := (publicBatchWrapper86_copies961 a h).2.2.2.2.1
  have c30757 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.1
  have c30758 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_480_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_16
  simp only [← c30753, ← c30754, ← c30755] at e_480_16
  have e_480_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_17
  simp only [← c30756, ← c30757, k2, ← c30758] at e_480_17
  have hr := e_480_17
  simp only [e_480_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 480 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1847)) := by
  have c30759 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.1
  have c30760 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.1
  have c30761 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.1
  have c30762 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30763 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30764 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_480_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_18
  simp only [← c30759, ← c30760, ← c30761] at e_480_18
  have e_480_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row480 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_480_19
  simp only [← c30762, ← c30763, k2, ← c30764] at e_480_19
  have hr := e_480_19
  simp only [e_480_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 481 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1848)) := by
  have c30765 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30766 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30767 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30768 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30769 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30770 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_481_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_0
  simp only [← c30765, ← c30766, ← c30767] at e_481_0
  have e_481_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_1
  simp only [← c30768, ← c30769, k2, ← c30770] at e_481_1
  have hr := e_481_1
  simp only [e_481_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 481 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1849)) := by
  have c30771 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30772 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30773 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30774 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30775 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30776 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_481_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_2
  simp only [← c30771, ← c30772, ← c30773] at e_481_2
  have e_481_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_3
  simp only [← c30774, ← c30775, k2, ← c30776] at e_481_3
  have hr := e_481_3
  simp only [e_481_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 481 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1850)) := by
  have c30777 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30778 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30779 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30780 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30781 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30782 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_481_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_4
  simp only [← c30777, ← c30778, ← c30779] at e_481_4
  have e_481_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_5
  simp only [← c30780, ← c30781, k2, ← c30782] at e_481_5
  have hr := e_481_5
  simp only [e_481_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 481 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1851)) := by
  have c30783 := (publicBatchWrapper86_copies961 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30784 := (publicBatchWrapper86_copies962 a h).1
  have c30785 := (publicBatchWrapper86_copies962 a h).2.1
  have c30786 := (publicBatchWrapper86_copies962 a h).2.2.1
  have c30787 := (publicBatchWrapper86_copies962 a h).2.2.2.1
  have c30788 := (publicBatchWrapper86_copies962 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_481_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_6
  simp only [← c30783, ← c30784, ← c30785] at e_481_6
  have e_481_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_7
  simp only [← c30786, ← c30787, k2, ← c30788] at e_481_7
  have hr := e_481_7
  simp only [e_481_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 481 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1852)) := by
  have c30789 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.1
  have c30790 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.1
  have c30791 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.1
  have c30792 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.1
  have c30793 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.1
  have c30794 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_481_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_8
  simp only [← c30789, ← c30790, ← c30791] at e_481_8
  have e_481_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_9
  simp only [← c30792, ← c30793, k2, ← c30794] at e_481_9
  have hr := e_481_9
  simp only [e_481_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 481 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1853)) := by
  have c30795 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30796 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30797 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30798 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30799 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30800 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_481_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_10
  simp only [← c30795, ← c30796, ← c30797] at e_481_10
  have e_481_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_11
  simp only [← c30798, ← c30799, k2, ← c30800] at e_481_11
  have hr := e_481_11
  simp only [e_481_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 481 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1854)) := by
  have c30801 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30802 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30803 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30804 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30805 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30806 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_481_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_12
  simp only [← c30801, ← c30802, ← c30803] at e_481_12
  have e_481_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_13
  simp only [← c30804, ← c30805, k2, ← c30806] at e_481_13
  have hr := e_481_13
  simp only [e_481_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 481 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1855)) := by
  have c30807 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30808 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30809 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30810 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30811 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30812 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_481_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_14
  simp only [← c30807, ← c30808, ← c30809] at e_481_14
  have e_481_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_15
  simp only [← c30810, ← c30811, k2, ← c30812] at e_481_15
  have hr := e_481_15
  simp only [e_481_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 481 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1856)) := by
  have c30813 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30814 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30815 := (publicBatchWrapper86_copies962 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30816 := (publicBatchWrapper86_copies963 a h).1
  have c30817 := (publicBatchWrapper86_copies963 a h).2.1
  have c30818 := (publicBatchWrapper86_copies963 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_481_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_16
  simp only [← c30813, ← c30814, ← c30815] at e_481_16
  have e_481_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_17
  simp only [← c30816, ← c30817, k2, ← c30818] at e_481_17
  have hr := e_481_17
  simp only [e_481_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 481 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1857)) := by
  have c30819 := (publicBatchWrapper86_copies963 a h).2.2.2.1
  have c30820 := (publicBatchWrapper86_copies963 a h).2.2.2.2.1
  have c30821 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.1
  have c30822 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.1
  have c30823 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.1
  have c30824 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_481_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_18
  simp only [← c30819, ← c30820, ← c30821] at e_481_18
  have e_481_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row481 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_481_19
  simp only [← c30822, ← c30823, k2, ← c30824] at e_481_19
  have hr := e_481_19
  simp only [e_481_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 482 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1858)) := by
  have c30825 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.1
  have c30826 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30827 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30828 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30829 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30830 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_482_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_0
  simp only [← c30825, ← c30826, ← c30827] at e_482_0
  have e_482_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_1
  simp only [← c30828, ← c30829, k2, ← c30830] at e_482_1
  have hr := e_482_1
  simp only [e_482_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 482 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1859)) := by
  have c30831 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30832 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30833 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30834 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30835 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30836 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_482_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_2
  simp only [← c30831, ← c30832, ← c30833] at e_482_2
  have e_482_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_3
  simp only [← c30834, ← c30835, k2, ← c30836] at e_482_3
  have hr := e_482_3
  simp only [e_482_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 482 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1860)) := by
  have c30837 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30838 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30839 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30840 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30841 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30842 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_482_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_4
  simp only [← c30837, ← c30838, ← c30839] at e_482_4
  have e_482_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_5
  simp only [← c30840, ← c30841, k2, ← c30842] at e_482_5
  have hr := e_482_5
  simp only [e_482_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 482 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1861)) := by
  have c30843 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30844 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30845 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30846 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30847 := (publicBatchWrapper86_copies963 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30848 := (publicBatchWrapper86_copies964 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_482_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_6
  simp only [← c30843, ← c30844, ← c30845] at e_482_6
  have e_482_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_7
  simp only [← c30846, ← c30847, k2, ← c30848] at e_482_7
  have hr := e_482_7
  simp only [e_482_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 482 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1862)) := by
  have c30849 := (publicBatchWrapper86_copies964 a h).2.1
  have c30850 := (publicBatchWrapper86_copies964 a h).2.2.1
  have c30851 := (publicBatchWrapper86_copies964 a h).2.2.2.1
  have c30852 := (publicBatchWrapper86_copies964 a h).2.2.2.2.1
  have c30853 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.1
  have c30854 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_482_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_8
  simp only [← c30849, ← c30850, ← c30851] at e_482_8
  have e_482_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_9
  simp only [← c30852, ← c30853, k2, ← c30854] at e_482_9
  have hr := e_482_9
  simp only [e_482_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 482 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1863)) := by
  have c30855 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.1
  have c30856 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.1
  have c30857 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.1
  have c30858 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30859 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30860 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_482_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_10
  simp only [← c30855, ← c30856, ← c30857] at e_482_10
  have e_482_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_11
  simp only [← c30858, ← c30859, k2, ← c30860] at e_482_11
  have hr := e_482_11
  simp only [e_482_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 482 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1864)) := by
  have c30861 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30862 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30863 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30864 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30865 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30866 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_482_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_12
  simp only [← c30861, ← c30862, ← c30863] at e_482_12
  have e_482_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_13
  simp only [← c30864, ← c30865, k2, ← c30866] at e_482_13
  have hr := e_482_13
  simp only [e_482_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 482 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1865)) := by
  have c30867 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30868 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30869 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30870 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30871 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30872 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_482_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_14
  simp only [← c30867, ← c30868, ← c30869] at e_482_14
  have e_482_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_15
  simp only [← c30870, ← c30871, k2, ← c30872] at e_482_15
  have hr := e_482_15
  simp only [e_482_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 482 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1866)) := by
  have c30873 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30874 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30875 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30876 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30877 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30878 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_482_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_16
  simp only [← c30873, ← c30874, ← c30875] at e_482_16
  have e_482_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_17
  simp only [← c30876, ← c30877, k2, ← c30878] at e_482_17
  have hr := e_482_17
  simp only [e_482_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 482 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1867)) := by
  have c30879 := (publicBatchWrapper86_copies964 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30880 := (publicBatchWrapper86_copies965 a h).1
  have c30881 := (publicBatchWrapper86_copies965 a h).2.1
  have c30882 := (publicBatchWrapper86_copies965 a h).2.2.1
  have c30883 := (publicBatchWrapper86_copies965 a h).2.2.2.1
  have c30884 := (publicBatchWrapper86_copies965 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_482_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_18
  simp only [← c30879, ← c30880, ← c30881] at e_482_18
  have e_482_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row482 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_482_19
  simp only [← c30882, ← c30883, k2, ← c30884] at e_482_19
  have hr := e_482_19
  simp only [e_482_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 483 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1868)) := by
  have c30885 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.1
  have c30886 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.1
  have c30887 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.1
  have c30888 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.1
  have c30889 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.1
  have c30890 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_483_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_0
  simp only [← c30885, ← c30886, ← c30887] at e_483_0
  have e_483_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_1
  simp only [← c30888, ← c30889, k2, ← c30890] at e_483_1
  have hr := e_483_1
  simp only [e_483_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 483 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1869)) := by
  have c30891 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30892 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30893 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30894 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30895 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30896 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_483_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_2
  simp only [← c30891, ← c30892, ← c30893] at e_483_2
  have e_483_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_3
  simp only [← c30894, ← c30895, k2, ← c30896] at e_483_3
  have hr := e_483_3
  simp only [e_483_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 483 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1870)) := by
  have c30897 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30898 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30899 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30900 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30901 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30902 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_483_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_4
  simp only [← c30897, ← c30898, ← c30899] at e_483_4
  have e_483_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_5
  simp only [← c30900, ← c30901, k2, ← c30902] at e_483_5
  have hr := e_483_5
  simp only [e_483_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 483 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1871)) := by
  have c30903 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30904 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30905 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30906 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30907 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30908 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_483_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_6
  simp only [← c30903, ← c30904, ← c30905] at e_483_6
  have e_483_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_7
  simp only [← c30906, ← c30907, k2, ← c30908] at e_483_7
  have hr := e_483_7
  simp only [e_483_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 483 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1872)) := by
  have c30909 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30910 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30911 := (publicBatchWrapper86_copies965 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30912 := (publicBatchWrapper86_copies966 a h).1
  have c30913 := (publicBatchWrapper86_copies966 a h).2.1
  have c30914 := (publicBatchWrapper86_copies966 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_483_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_8
  simp only [← c30909, ← c30910, ← c30911] at e_483_8
  have e_483_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_9
  simp only [← c30912, ← c30913, k2, ← c30914] at e_483_9
  have hr := e_483_9
  simp only [e_483_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 483 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1873)) := by
  have c30915 := (publicBatchWrapper86_copies966 a h).2.2.2.1
  have c30916 := (publicBatchWrapper86_copies966 a h).2.2.2.2.1
  have c30917 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.1
  have c30918 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.1
  have c30919 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.1
  have c30920 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_483_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_10
  simp only [← c30915, ← c30916, ← c30917] at e_483_10
  have e_483_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_11
  simp only [← c30918, ← c30919, k2, ← c30920] at e_483_11
  have hr := e_483_11
  simp only [e_483_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 483 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1874)) := by
  have c30921 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.1
  have c30922 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30923 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30924 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30925 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30926 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_483_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_12
  simp only [← c30921, ← c30922, ← c30923] at e_483_12
  have e_483_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_13
  simp only [← c30924, ← c30925, k2, ← c30926] at e_483_13
  have hr := e_483_13
  simp only [e_483_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 483 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1875)) := by
  have c30927 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30928 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30929 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30930 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30931 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30932 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_483_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_14
  simp only [← c30927, ← c30928, ← c30929] at e_483_14
  have e_483_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_15
  simp only [← c30930, ← c30931, k2, ← c30932] at e_483_15
  have hr := e_483_15
  simp only [e_483_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 483 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1876)) := by
  have c30933 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30934 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30935 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30936 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30937 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30938 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_483_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_16
  simp only [← c30933, ← c30934, ← c30935] at e_483_16
  have e_483_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_17
  simp only [← c30936, ← c30937, k2, ← c30938] at e_483_17
  have hr := e_483_17
  simp only [e_483_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 483 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1877)) := by
  have c30939 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30940 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30941 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30942 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30943 := (publicBatchWrapper86_copies966 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30944 := (publicBatchWrapper86_copies967 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_483_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_18
  simp only [← c30939, ← c30940, ← c30941] at e_483_18
  have e_483_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row483 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_483_19
  simp only [← c30942, ← c30943, k2, ← c30944] at e_483_19
  have hr := e_483_19
  simp only [e_483_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 484 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1878)) := by
  have c30945 := (publicBatchWrapper86_copies967 a h).2.1
  have c30946 := (publicBatchWrapper86_copies967 a h).2.2.1
  have c30947 := (publicBatchWrapper86_copies967 a h).2.2.2.1
  have c30948 := (publicBatchWrapper86_copies967 a h).2.2.2.2.1
  have c30949 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.1
  have c30950 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_484_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_0
  simp only [← c30945, ← c30946, ← c30947] at e_484_0
  have e_484_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_1
  simp only [← c30948, ← c30949, k2, ← c30950] at e_484_1
  have hr := e_484_1
  simp only [e_484_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 484 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1879)) := by
  have c30951 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.1
  have c30952 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.1
  have c30953 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.1
  have c30954 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.1
  have c30955 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30956 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_484_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_2
  simp only [← c30951, ← c30952, ← c30953] at e_484_2
  have e_484_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_3
  simp only [← c30954, ← c30955, k2, ← c30956] at e_484_3
  have hr := e_484_3
  simp only [e_484_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 484 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1880)) := by
  have c30957 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30958 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30959 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30960 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30961 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30962 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_484_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_4
  simp only [← c30957, ← c30958, ← c30959] at e_484_4
  have e_484_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_5
  simp only [← c30960, ← c30961, k2, ← c30962] at e_484_5
  have hr := e_484_5
  simp only [e_484_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 484 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1881)) := by
  have c30963 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30964 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30965 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30966 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30967 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30968 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_484_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_6
  simp only [← c30963, ← c30964, ← c30965] at e_484_6
  have e_484_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_7
  simp only [← c30966, ← c30967, k2, ← c30968] at e_484_7
  have hr := e_484_7
  simp only [e_484_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 484 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1882)) := by
  have c30969 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30970 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30971 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30972 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30973 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30974 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_484_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_8
  simp only [← c30969, ← c30970, ← c30971] at e_484_8
  have e_484_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_9
  simp only [← c30972, ← c30973, k2, ← c30974] at e_484_9
  have hr := e_484_9
  simp only [e_484_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 484 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1883)) := by
  have c30975 := (publicBatchWrapper86_copies967 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c30976 := (publicBatchWrapper86_copies968 a h).1
  have c30977 := (publicBatchWrapper86_copies968 a h).2.1
  have c30978 := (publicBatchWrapper86_copies968 a h).2.2.1
  have c30979 := (publicBatchWrapper86_copies968 a h).2.2.2.1
  have c30980 := (publicBatchWrapper86_copies968 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_484_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_10
  simp only [← c30975, ← c30976, ← c30977] at e_484_10
  have e_484_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_11
  simp only [← c30978, ← c30979, k2, ← c30980] at e_484_11
  have hr := e_484_11
  simp only [e_484_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 484 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1884)) := by
  have c30981 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.1
  have c30982 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.1
  have c30983 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.1
  have c30984 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.1
  have c30985 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.1
  have c30986 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_484_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_12
  simp only [← c30981, ← c30982, ← c30983] at e_484_12
  have e_484_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_13
  simp only [← c30984, ← c30985, k2, ← c30986] at e_484_13
  have hr := e_484_13
  simp only [e_484_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 484 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1885)) := by
  have c30987 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c30988 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30989 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30990 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30991 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30992 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_484_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_14
  simp only [← c30987, ← c30988, ← c30989] at e_484_14
  have e_484_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_15
  simp only [← c30990, ← c30991, k2, ← c30992] at e_484_15
  have hr := e_484_15
  simp only [e_484_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 484 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1886)) := by
  have c30993 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30994 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30995 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30996 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30997 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c30998 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_484_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_16
  simp only [← c30993, ← c30994, ← c30995] at e_484_16
  have e_484_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_17
  simp only [← c30996, ← c30997, k2, ← c30998] at e_484_17
  have hr := e_484_17
  simp only [e_484_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 484 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1887)) := by
  have c30999 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31000 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31001 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31002 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31003 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31004 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_484_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_18
  simp only [← c30999, ← c31000, ← c31001] at e_484_18
  have e_484_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row484 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_484_19
  simp only [← c31002, ← c31003, k2, ← c31004] at e_484_19
  have hr := e_484_19
  simp only [e_484_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 485 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1888)) := by
  have c31005 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31006 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31007 := (publicBatchWrapper86_copies968 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31008 := (publicBatchWrapper86_copies969 a h).1
  have c31009 := (publicBatchWrapper86_copies969 a h).2.1
  have c31010 := (publicBatchWrapper86_copies969 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_485_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_0
  simp only [← c31005, ← c31006, ← c31007] at e_485_0
  have e_485_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_1
  simp only [← c31008, ← c31009, k2, ← c31010] at e_485_1
  have hr := e_485_1
  simp only [e_485_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 485 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1889)) := by
  have c31011 := (publicBatchWrapper86_copies969 a h).2.2.2.1
  have c31012 := (publicBatchWrapper86_copies969 a h).2.2.2.2.1
  have c31013 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.1
  have c31014 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.1
  have c31015 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.1
  have c31016 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_485_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_2
  simp only [← c31011, ← c31012, ← c31013] at e_485_2
  have e_485_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_3
  simp only [← c31014, ← c31015, k2, ← c31016] at e_485_3
  have hr := e_485_3
  simp only [e_485_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 485 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1890)) := by
  have c31017 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.1
  have c31018 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31019 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31020 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31021 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31022 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_485_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_4
  simp only [← c31017, ← c31018, ← c31019] at e_485_4
  have e_485_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_5
  simp only [← c31020, ← c31021, k2, ← c31022] at e_485_5
  have hr := e_485_5
  simp only [e_485_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 485 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1891)) := by
  have c31023 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31024 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31025 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31026 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31027 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31028 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_485_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_6
  simp only [← c31023, ← c31024, ← c31025] at e_485_6
  have e_485_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_7
  simp only [← c31026, ← c31027, k2, ← c31028] at e_485_7
  have hr := e_485_7
  simp only [e_485_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 485 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1892)) := by
  have c31029 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31030 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31031 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31032 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31033 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31034 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_485_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_8
  simp only [← c31029, ← c31030, ← c31031] at e_485_8
  have e_485_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_9
  simp only [← c31032, ← c31033, k2, ← c31034] at e_485_9
  have hr := e_485_9
  simp only [e_485_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 485 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1893)) := by
  have c31035 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31036 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31037 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31038 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31039 := (publicBatchWrapper86_copies969 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31040 := (publicBatchWrapper86_copies970 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_485_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_10
  simp only [← c31035, ← c31036, ← c31037] at e_485_10
  have e_485_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_11
  simp only [← c31038, ← c31039, k2, ← c31040] at e_485_11
  have hr := e_485_11
  simp only [e_485_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 485 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1894)) := by
  have c31041 := (publicBatchWrapper86_copies970 a h).2.1
  have c31042 := (publicBatchWrapper86_copies970 a h).2.2.1
  have c31043 := (publicBatchWrapper86_copies970 a h).2.2.2.1
  have c31044 := (publicBatchWrapper86_copies970 a h).2.2.2.2.1
  have c31045 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.1
  have c31046 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_485_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_12
  simp only [← c31041, ← c31042, ← c31043] at e_485_12
  have e_485_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_13
  simp only [← c31044, ← c31045, k2, ← c31046] at e_485_13
  have hr := e_485_13
  simp only [e_485_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 485 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1895)) := by
  have c31047 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.1
  have c31048 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.1
  have c31049 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.1
  have c31050 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31051 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31052 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_485_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_14
  simp only [← c31047, ← c31048, ← c31049] at e_485_14
  have e_485_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_15
  simp only [← c31050, ← c31051, k2, ← c31052] at e_485_15
  have hr := e_485_15
  simp only [e_485_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 485 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1896)) := by
  have c31053 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31054 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31055 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31056 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31057 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31058 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_485_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_16
  simp only [← c31053, ← c31054, ← c31055] at e_485_16
  have e_485_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_17
  simp only [← c31056, ← c31057, k2, ← c31058] at e_485_17
  have hr := e_485_17
  simp only [e_485_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 485 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1897)) := by
  have c31059 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31060 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31061 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31062 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31063 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31064 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_485_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_18
  simp only [← c31059, ← c31060, ← c31061] at e_485_18
  have e_485_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row485 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_485_19
  simp only [← c31062, ← c31063, k2, ← c31064] at e_485_19
  have hr := e_485_19
  simp only [e_485_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 486 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1898)) := by
  have c31065 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31066 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31067 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31068 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31069 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31070 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_486_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_0
  simp only [← c31065, ← c31066, ← c31067] at e_486_0
  have e_486_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_1
  simp only [← c31068, ← c31069, k2, ← c31070] at e_486_1
  have hr := e_486_1
  simp only [e_486_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 486 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1899)) := by
  have c31071 := (publicBatchWrapper86_copies970 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31072 := (publicBatchWrapper86_copies971 a h).1
  have c31073 := (publicBatchWrapper86_copies971 a h).2.1
  have c31074 := (publicBatchWrapper86_copies971 a h).2.2.1
  have c31075 := (publicBatchWrapper86_copies971 a h).2.2.2.1
  have c31076 := (publicBatchWrapper86_copies971 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_486_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_2
  simp only [← c31071, ← c31072, ← c31073] at e_486_2
  have e_486_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_3
  simp only [← c31074, ← c31075, k2, ← c31076] at e_486_3
  have hr := e_486_3
  simp only [e_486_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 486 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1900)) := by
  have c31077 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.1
  have c31078 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.1
  have c31079 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.1
  have c31080 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.1
  have c31081 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.1
  have c31082 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_486_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_4
  simp only [← c31077, ← c31078, ← c31079] at e_486_4
  have e_486_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_5
  simp only [← c31080, ← c31081, k2, ← c31082] at e_486_5
  have hr := e_486_5
  simp only [e_486_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 486 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1901)) := by
  have c31083 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31084 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31085 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31086 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31087 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31088 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_486_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_6
  simp only [← c31083, ← c31084, ← c31085] at e_486_6
  have e_486_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_7
  simp only [← c31086, ← c31087, k2, ← c31088] at e_486_7
  have hr := e_486_7
  simp only [e_486_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 486 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1902)) := by
  have c31089 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31090 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31091 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31092 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31093 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31094 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_486_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_8
  simp only [← c31089, ← c31090, ← c31091] at e_486_8
  have e_486_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_9
  simp only [← c31092, ← c31093, k2, ← c31094] at e_486_9
  have hr := e_486_9
  simp only [e_486_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 486 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1903)) := by
  have c31095 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31096 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31097 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31098 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31099 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31100 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_486_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_10
  simp only [← c31095, ← c31096, ← c31097] at e_486_10
  have e_486_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_11
  simp only [← c31098, ← c31099, k2, ← c31100] at e_486_11
  have hr := e_486_11
  simp only [e_486_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 486 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1904)) := by
  have c31101 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31102 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31103 := (publicBatchWrapper86_copies971 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31104 := (publicBatchWrapper86_copies972 a h).1
  have c31105 := (publicBatchWrapper86_copies972 a h).2.1
  have c31106 := (publicBatchWrapper86_copies972 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_486_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_12
  simp only [← c31101, ← c31102, ← c31103] at e_486_12
  have e_486_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_13
  simp only [← c31104, ← c31105, k2, ← c31106] at e_486_13
  have hr := e_486_13
  simp only [e_486_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 486 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1905)) := by
  have c31107 := (publicBatchWrapper86_copies972 a h).2.2.2.1
  have c31108 := (publicBatchWrapper86_copies972 a h).2.2.2.2.1
  have c31109 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.1
  have c31110 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.1
  have c31111 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.1
  have c31112 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_486_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_14
  simp only [← c31107, ← c31108, ← c31109] at e_486_14
  have e_486_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_15
  simp only [← c31110, ← c31111, k2, ← c31112] at e_486_15
  have hr := e_486_15
  simp only [e_486_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 486 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1906)) := by
  have c31113 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.1
  have c31114 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31115 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31116 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31117 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31118 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_486_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_16
  simp only [← c31113, ← c31114, ← c31115] at e_486_16
  have e_486_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_17
  simp only [← c31116, ← c31117, k2, ← c31118] at e_486_17
  have hr := e_486_17
  simp only [e_486_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 486 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1907)) := by
  have c31119 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31120 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31121 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31122 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31123 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31124 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_486_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_18
  simp only [← c31119, ← c31120, ← c31121] at e_486_18
  have e_486_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row486 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_486_19
  simp only [← c31122, ← c31123, k2, ← c31124] at e_486_19
  have hr := e_486_19
  simp only [e_486_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 487 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1908)) := by
  have c31125 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31126 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31127 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31128 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31129 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31130 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_487_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_0
  simp only [← c31125, ← c31126, ← c31127] at e_487_0
  have e_487_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_1
  simp only [← c31128, ← c31129, k2, ← c31130] at e_487_1
  have hr := e_487_1
  simp only [e_487_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 487 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1909)) := by
  have c31131 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31132 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31133 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31134 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31135 := (publicBatchWrapper86_copies972 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31136 := (publicBatchWrapper86_copies973 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_487_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_2
  simp only [← c31131, ← c31132, ← c31133] at e_487_2
  have e_487_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_3
  simp only [← c31134, ← c31135, k2, ← c31136] at e_487_3
  have hr := e_487_3
  simp only [e_487_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 487 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1910)) := by
  have c31137 := (publicBatchWrapper86_copies973 a h).2.1
  have c31138 := (publicBatchWrapper86_copies973 a h).2.2.1
  have c31139 := (publicBatchWrapper86_copies973 a h).2.2.2.1
  have c31140 := (publicBatchWrapper86_copies973 a h).2.2.2.2.1
  have c31141 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.1
  have c31142 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_487_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_4
  simp only [← c31137, ← c31138, ← c31139] at e_487_4
  have e_487_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_5
  simp only [← c31140, ← c31141, k2, ← c31142] at e_487_5
  have hr := e_487_5
  simp only [e_487_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 487 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1911)) := by
  have c31143 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.1
  have c31144 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.1
  have c31145 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.1
  have c31146 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31147 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31148 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_487_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_6
  simp only [← c31143, ← c31144, ← c31145] at e_487_6
  have e_487_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_7
  simp only [← c31146, ← c31147, k2, ← c31148] at e_487_7
  have hr := e_487_7
  simp only [e_487_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 487 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1912)) := by
  have c31149 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31150 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31151 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31152 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31153 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31154 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_487_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_8
  simp only [← c31149, ← c31150, ← c31151] at e_487_8
  have e_487_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_9
  simp only [← c31152, ← c31153, k2, ← c31154] at e_487_9
  have hr := e_487_9
  simp only [e_487_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 487 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1913)) := by
  have c31155 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31156 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31157 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31158 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31159 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31160 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_487_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_10
  simp only [← c31155, ← c31156, ← c31157] at e_487_10
  have e_487_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_11
  simp only [← c31158, ← c31159, k2, ← c31160] at e_487_11
  have hr := e_487_11
  simp only [e_487_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 487 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1914)) := by
  have c31161 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31162 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31163 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31164 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31165 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31166 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_487_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_12
  simp only [← c31161, ← c31162, ← c31163] at e_487_12
  have e_487_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_13
  simp only [← c31164, ← c31165, k2, ← c31166] at e_487_13
  have hr := e_487_13
  simp only [e_487_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 487 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1915)) := by
  have c31167 := (publicBatchWrapper86_copies973 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31168 := (publicBatchWrapper86_copies974 a h).1
  have c31169 := (publicBatchWrapper86_copies974 a h).2.1
  have c31170 := (publicBatchWrapper86_copies974 a h).2.2.1
  have c31171 := (publicBatchWrapper86_copies974 a h).2.2.2.1
  have c31172 := (publicBatchWrapper86_copies974 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_487_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_14
  simp only [← c31167, ← c31168, ← c31169] at e_487_14
  have e_487_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_15
  simp only [← c31170, ← c31171, k2, ← c31172] at e_487_15
  have hr := e_487_15
  simp only [e_487_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 487 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1916)) := by
  have c31173 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.1
  have c31174 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.1
  have c31175 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.1
  have c31176 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.1
  have c31177 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.1
  have c31178 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_487_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_16
  simp only [← c31173, ← c31174, ← c31175] at e_487_16
  have e_487_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_17
  simp only [← c31176, ← c31177, k2, ← c31178] at e_487_17
  have hr := e_487_17
  simp only [e_487_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 487 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1917)) := by
  have c31179 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31180 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31181 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31182 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31183 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31184 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_487_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_18
  simp only [← c31179, ← c31180, ← c31181] at e_487_18
  have e_487_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row487 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_487_19
  simp only [← c31182, ← c31183, k2, ← c31184] at e_487_19
  have hr := e_487_19
  simp only [e_487_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 488 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1918)) := by
  have c31185 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31186 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31187 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31188 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31189 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31190 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_488_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_0
  simp only [← c31185, ← c31186, ← c31187] at e_488_0
  have e_488_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_1
  simp only [← c31188, ← c31189, k2, ← c31190] at e_488_1
  have hr := e_488_1
  simp only [e_488_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 488 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1919)) := by
  have c31191 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31192 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31193 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31194 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31195 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31196 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_488_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_2
  simp only [← c31191, ← c31192, ← c31193] at e_488_2
  have e_488_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_3
  simp only [← c31194, ← c31195, k2, ← c31196] at e_488_3
  have hr := e_488_3
  simp only [e_488_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 488 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1920)) := by
  have c31197 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31198 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31199 := (publicBatchWrapper86_copies974 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31200 := (publicBatchWrapper86_copies975 a h).1
  have c31201 := (publicBatchWrapper86_copies975 a h).2.1
  have c31202 := (publicBatchWrapper86_copies975 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_488_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_4
  simp only [← c31197, ← c31198, ← c31199] at e_488_4
  have e_488_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_5
  simp only [← c31200, ← c31201, k2, ← c31202] at e_488_5
  have hr := e_488_5
  simp only [e_488_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 488 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1921)) := by
  have c31203 := (publicBatchWrapper86_copies975 a h).2.2.2.1
  have c31204 := (publicBatchWrapper86_copies975 a h).2.2.2.2.1
  have c31205 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.1
  have c31206 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.1
  have c31207 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.1
  have c31208 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_488_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_6
  simp only [← c31203, ← c31204, ← c31205] at e_488_6
  have e_488_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_7
  simp only [← c31206, ← c31207, k2, ← c31208] at e_488_7
  have hr := e_488_7
  simp only [e_488_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 488 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1922)) := by
  have c31209 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.1
  have c31210 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31211 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31212 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31213 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31214 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_488_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_8
  simp only [← c31209, ← c31210, ← c31211] at e_488_8
  have e_488_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_9
  simp only [← c31212, ← c31213, k2, ← c31214] at e_488_9
  have hr := e_488_9
  simp only [e_488_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 488 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1923)) := by
  have c31215 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31216 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31217 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31218 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31219 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31220 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_488_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_10
  simp only [← c31215, ← c31216, ← c31217] at e_488_10
  have e_488_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_11
  simp only [← c31218, ← c31219, k2, ← c31220] at e_488_11
  have hr := e_488_11
  simp only [e_488_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 488 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1924)) := by
  have c31221 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31222 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31223 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31224 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31225 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31226 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_488_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_12
  simp only [← c31221, ← c31222, ← c31223] at e_488_12
  have e_488_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_13
  simp only [← c31224, ← c31225, k2, ← c31226] at e_488_13
  have hr := e_488_13
  simp only [e_488_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 488 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1925)) := by
  have c31227 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31228 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31229 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31230 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31231 := (publicBatchWrapper86_copies975 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31232 := (publicBatchWrapper86_copies976 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_488_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_14
  simp only [← c31227, ← c31228, ← c31229] at e_488_14
  have e_488_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_15
  simp only [← c31230, ← c31231, k2, ← c31232] at e_488_15
  have hr := e_488_15
  simp only [e_488_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 488 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1926)) := by
  have c31233 := (publicBatchWrapper86_copies976 a h).2.1
  have c31234 := (publicBatchWrapper86_copies976 a h).2.2.1
  have c31235 := (publicBatchWrapper86_copies976 a h).2.2.2.1
  have c31236 := (publicBatchWrapper86_copies976 a h).2.2.2.2.1
  have c31237 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.1
  have c31238 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_488_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_16
  simp only [← c31233, ← c31234, ← c31235] at e_488_16
  have e_488_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_17
  simp only [← c31236, ← c31237, k2, ← c31238] at e_488_17
  have hr := e_488_17
  simp only [e_488_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 488 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1927)) := by
  have c31239 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.1
  have c31240 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.1
  have c31241 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.1
  have c31242 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31243 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31244 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_488_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_18
  simp only [← c31239, ← c31240, ← c31241] at e_488_18
  have e_488_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row488 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_488_19
  simp only [← c31242, ← c31243, k2, ← c31244] at e_488_19
  have hr := e_488_19
  simp only [e_488_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 489 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1928)) := by
  have c31245 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31246 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31247 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31248 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31249 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31250 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_489_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_0
  simp only [← c31245, ← c31246, ← c31247] at e_489_0
  have e_489_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_1
  simp only [← c31248, ← c31249, k2, ← c31250] at e_489_1
  have hr := e_489_1
  simp only [e_489_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 489 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1929)) := by
  have c31251 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31252 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31253 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31254 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31255 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31256 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_489_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_2
  simp only [← c31251, ← c31252, ← c31253] at e_489_2
  have e_489_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_3
  simp only [← c31254, ← c31255, k2, ← c31256] at e_489_3
  have hr := e_489_3
  simp only [e_489_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 489 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1930)) := by
  have c31257 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31258 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31259 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31260 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31261 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31262 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_489_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_4
  simp only [← c31257, ← c31258, ← c31259] at e_489_4
  have e_489_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_5
  simp only [← c31260, ← c31261, k2, ← c31262] at e_489_5
  have hr := e_489_5
  simp only [e_489_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 489 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1931)) := by
  have c31263 := (publicBatchWrapper86_copies976 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31264 := (publicBatchWrapper86_copies977 a h).1
  have c31265 := (publicBatchWrapper86_copies977 a h).2.1
  have c31266 := (publicBatchWrapper86_copies977 a h).2.2.1
  have c31267 := (publicBatchWrapper86_copies977 a h).2.2.2.1
  have c31268 := (publicBatchWrapper86_copies977 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_489_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_6
  simp only [← c31263, ← c31264, ← c31265] at e_489_6
  have e_489_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_7
  simp only [← c31266, ← c31267, k2, ← c31268] at e_489_7
  have hr := e_489_7
  simp only [e_489_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 489 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1932)) := by
  have c31269 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.1
  have c31270 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.1
  have c31271 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.1
  have c31272 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.1
  have c31273 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.1
  have c31274 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_489_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_8
  simp only [← c31269, ← c31270, ← c31271] at e_489_8
  have e_489_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_9
  simp only [← c31272, ← c31273, k2, ← c31274] at e_489_9
  have hr := e_489_9
  simp only [e_489_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 489 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1933)) := by
  have c31275 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31276 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31277 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31278 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31279 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31280 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_489_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_10
  simp only [← c31275, ← c31276, ← c31277] at e_489_10
  have e_489_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_11
  simp only [← c31278, ← c31279, k2, ← c31280] at e_489_11
  have hr := e_489_11
  simp only [e_489_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 489 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1934)) := by
  have c31281 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31282 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31283 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31284 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31285 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31286 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_489_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_12
  simp only [← c31281, ← c31282, ← c31283] at e_489_12
  have e_489_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_13
  simp only [← c31284, ← c31285, k2, ← c31286] at e_489_13
  have hr := e_489_13
  simp only [e_489_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 489 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1935)) := by
  have c31287 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31288 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31289 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31290 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31291 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31292 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_489_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_14
  simp only [← c31287, ← c31288, ← c31289] at e_489_14
  have e_489_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_15
  simp only [← c31290, ← c31291, k2, ← c31292] at e_489_15
  have hr := e_489_15
  simp only [e_489_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 489 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1936)) := by
  have c31293 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31294 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31295 := (publicBatchWrapper86_copies977 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31296 := (publicBatchWrapper86_copies978 a h).1
  have c31297 := (publicBatchWrapper86_copies978 a h).2.1
  have c31298 := (publicBatchWrapper86_copies978 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_489_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_16
  simp only [← c31293, ← c31294, ← c31295] at e_489_16
  have e_489_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_17
  simp only [← c31296, ← c31297, k2, ← c31298] at e_489_17
  have hr := e_489_17
  simp only [e_489_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 489 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1937)) := by
  have c31299 := (publicBatchWrapper86_copies978 a h).2.2.2.1
  have c31300 := (publicBatchWrapper86_copies978 a h).2.2.2.2.1
  have c31301 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.1
  have c31302 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.1
  have c31303 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.1
  have c31304 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_489_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_18
  simp only [← c31299, ← c31300, ← c31301] at e_489_18
  have e_489_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row489 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_489_19
  simp only [← c31302, ← c31303, k2, ← c31304] at e_489_19
  have hr := e_489_19
  simp only [e_489_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 490 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1938)) := by
  have c31305 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.1
  have c31306 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31307 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31308 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31309 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31310 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_490_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_0
  simp only [← c31305, ← c31306, ← c31307] at e_490_0
  have e_490_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_1
  simp only [← c31308, ← c31309, k2, ← c31310] at e_490_1
  have hr := e_490_1
  simp only [e_490_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 490 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1939)) := by
  have c31311 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31312 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31313 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31314 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31315 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31316 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_490_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_2
  simp only [← c31311, ← c31312, ← c31313] at e_490_2
  have e_490_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_3
  simp only [← c31314, ← c31315, k2, ← c31316] at e_490_3
  have hr := e_490_3
  simp only [e_490_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 490 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1940)) := by
  have c31317 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31318 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31319 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31320 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31321 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31322 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_490_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_4
  simp only [← c31317, ← c31318, ← c31319] at e_490_4
  have e_490_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_5
  simp only [← c31320, ← c31321, k2, ← c31322] at e_490_5
  have hr := e_490_5
  simp only [e_490_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 490 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1941)) := by
  have c31323 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31324 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31325 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31326 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31327 := (publicBatchWrapper86_copies978 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31328 := (publicBatchWrapper86_copies979 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_490_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_6
  simp only [← c31323, ← c31324, ← c31325] at e_490_6
  have e_490_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_7
  simp only [← c31326, ← c31327, k2, ← c31328] at e_490_7
  have hr := e_490_7
  simp only [e_490_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 490 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1942)) := by
  have c31329 := (publicBatchWrapper86_copies979 a h).2.1
  have c31330 := (publicBatchWrapper86_copies979 a h).2.2.1
  have c31331 := (publicBatchWrapper86_copies979 a h).2.2.2.1
  have c31332 := (publicBatchWrapper86_copies979 a h).2.2.2.2.1
  have c31333 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.1
  have c31334 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_490_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_8
  simp only [← c31329, ← c31330, ← c31331] at e_490_8
  have e_490_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_9
  simp only [← c31332, ← c31333, k2, ← c31334] at e_490_9
  have hr := e_490_9
  simp only [e_490_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 490 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1943)) := by
  have c31335 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.1
  have c31336 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.1
  have c31337 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.1
  have c31338 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31339 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31340 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_490_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_10
  simp only [← c31335, ← c31336, ← c31337] at e_490_10
  have e_490_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_11
  simp only [← c31338, ← c31339, k2, ← c31340] at e_490_11
  have hr := e_490_11
  simp only [e_490_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 490 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1944)) := by
  have c31341 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31342 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31343 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31344 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31345 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31346 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_490_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_12
  simp only [← c31341, ← c31342, ← c31343] at e_490_12
  have e_490_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_13
  simp only [← c31344, ← c31345, k2, ← c31346] at e_490_13
  have hr := e_490_13
  simp only [e_490_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 490 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1945)) := by
  have c31347 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31348 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31349 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31350 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31351 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31352 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_490_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_14
  simp only [← c31347, ← c31348, ← c31349] at e_490_14
  have e_490_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_15
  simp only [← c31350, ← c31351, k2, ← c31352] at e_490_15
  have hr := e_490_15
  simp only [e_490_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 490 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1946)) := by
  have c31353 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31354 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31355 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31356 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31357 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31358 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_490_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_16
  simp only [← c31353, ← c31354, ← c31355] at e_490_16
  have e_490_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_17
  simp only [← c31356, ← c31357, k2, ← c31358] at e_490_17
  have hr := e_490_17
  simp only [e_490_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 490 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1947)) := by
  have c31359 := (publicBatchWrapper86_copies979 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31360 := (publicBatchWrapper86_copies980 a h).1
  have c31361 := (publicBatchWrapper86_copies980 a h).2.1
  have c31362 := (publicBatchWrapper86_copies980 a h).2.2.1
  have c31363 := (publicBatchWrapper86_copies980 a h).2.2.2.1
  have c31364 := (publicBatchWrapper86_copies980 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_490_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_18
  simp only [← c31359, ← c31360, ← c31361] at e_490_18
  have e_490_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row490 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_490_19
  simp only [← c31362, ← c31363, k2, ← c31364] at e_490_19
  have hr := e_490_19
  simp only [e_490_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 491 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1948)) := by
  have c31365 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.1
  have c31366 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.1
  have c31367 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.1
  have c31368 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.1
  have c31369 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.1
  have c31370 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_491_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_0
  simp only [← c31365, ← c31366, ← c31367] at e_491_0
  have e_491_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_1
  simp only [← c31368, ← c31369, k2, ← c31370] at e_491_1
  have hr := e_491_1
  simp only [e_491_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 491 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1949)) := by
  have c31371 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31372 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31373 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31374 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31375 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31376 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_491_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_2
  simp only [← c31371, ← c31372, ← c31373] at e_491_2
  have e_491_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_3
  simp only [← c31374, ← c31375, k2, ← c31376] at e_491_3
  have hr := e_491_3
  simp only [e_491_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 491 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1950)) := by
  have c31377 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31378 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31379 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31380 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31381 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31382 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_491_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_4
  simp only [← c31377, ← c31378, ← c31379] at e_491_4
  have e_491_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_5
  simp only [← c31380, ← c31381, k2, ← c31382] at e_491_5
  have hr := e_491_5
  simp only [e_491_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 491 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1951)) := by
  have c31383 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31384 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31385 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31386 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31387 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31388 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_491_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_6
  simp only [← c31383, ← c31384, ← c31385] at e_491_6
  have e_491_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_7
  simp only [← c31386, ← c31387, k2, ← c31388] at e_491_7
  have hr := e_491_7
  simp only [e_491_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 491 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1952)) := by
  have c31389 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31390 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31391 := (publicBatchWrapper86_copies980 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31392 := (publicBatchWrapper86_copies981 a h).1
  have c31393 := (publicBatchWrapper86_copies981 a h).2.1
  have c31394 := (publicBatchWrapper86_copies981 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_491_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_8
  simp only [← c31389, ← c31390, ← c31391] at e_491_8
  have e_491_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_9
  simp only [← c31392, ← c31393, k2, ← c31394] at e_491_9
  have hr := e_491_9
  simp only [e_491_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 491 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1953)) := by
  have c31395 := (publicBatchWrapper86_copies981 a h).2.2.2.1
  have c31396 := (publicBatchWrapper86_copies981 a h).2.2.2.2.1
  have c31397 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.1
  have c31398 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.1
  have c31399 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.1
  have c31400 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_491_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_10
  simp only [← c31395, ← c31396, ← c31397] at e_491_10
  have e_491_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_11
  simp only [← c31398, ← c31399, k2, ← c31400] at e_491_11
  have hr := e_491_11
  simp only [e_491_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 491 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1954)) := by
  have c31401 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.1
  have c31402 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31403 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31404 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31405 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31406 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_491_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_12
  simp only [← c31401, ← c31402, ← c31403] at e_491_12
  have e_491_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_13
  simp only [← c31404, ← c31405, k2, ← c31406] at e_491_13
  have hr := e_491_13
  simp only [e_491_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 491 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1955)) := by
  have c31407 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31408 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31409 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31410 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31411 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31412 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_491_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_14
  simp only [← c31407, ← c31408, ← c31409] at e_491_14
  have e_491_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_15
  simp only [← c31410, ← c31411, k2, ← c31412] at e_491_15
  have hr := e_491_15
  simp only [e_491_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 491 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1956)) := by
  have c31413 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31414 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31415 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31416 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31417 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31418 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_491_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_16
  simp only [← c31413, ← c31414, ← c31415] at e_491_16
  have e_491_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_17
  simp only [← c31416, ← c31417, k2, ← c31418] at e_491_17
  have hr := e_491_17
  simp only [e_491_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 491 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1957)) := by
  have c31419 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31420 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31421 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31422 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31423 := (publicBatchWrapper86_copies981 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31424 := (publicBatchWrapper86_copies982 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_491_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_18
  simp only [← c31419, ← c31420, ← c31421] at e_491_18
  have e_491_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row491 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_491_19
  simp only [← c31422, ← c31423, k2, ← c31424] at e_491_19
  have hr := e_491_19
  simp only [e_491_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 492 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1958)) := by
  have c31425 := (publicBatchWrapper86_copies982 a h).2.1
  have c31426 := (publicBatchWrapper86_copies982 a h).2.2.1
  have c31427 := (publicBatchWrapper86_copies982 a h).2.2.2.1
  have c31428 := (publicBatchWrapper86_copies982 a h).2.2.2.2.1
  have c31429 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.1
  have c31430 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_492_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_0
  simp only [← c31425, ← c31426, ← c31427] at e_492_0
  have e_492_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_1
  simp only [← c31428, ← c31429, k2, ← c31430] at e_492_1
  have hr := e_492_1
  simp only [e_492_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 492 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1959)) := by
  have c31431 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.1
  have c31432 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.1
  have c31433 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.1
  have c31434 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31435 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31436 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_492_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_2
  simp only [← c31431, ← c31432, ← c31433] at e_492_2
  have e_492_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_3
  simp only [← c31434, ← c31435, k2, ← c31436] at e_492_3
  have hr := e_492_3
  simp only [e_492_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 492 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1960)) := by
  have c31437 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31438 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31439 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31440 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31441 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31442 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_492_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_4
  simp only [← c31437, ← c31438, ← c31439] at e_492_4
  have e_492_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_5
  simp only [← c31440, ← c31441, k2, ← c31442] at e_492_5
  have hr := e_492_5
  simp only [e_492_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 492 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1961)) := by
  have c31443 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31444 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31445 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31446 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31447 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31448 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_492_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_6
  simp only [← c31443, ← c31444, ← c31445] at e_492_6
  have e_492_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_7
  simp only [← c31446, ← c31447, k2, ← c31448] at e_492_7
  have hr := e_492_7
  simp only [e_492_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 492 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1962)) := by
  have c31449 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31450 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31451 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31452 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31453 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31454 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_492_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_8
  simp only [← c31449, ← c31450, ← c31451] at e_492_8
  have e_492_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_9
  simp only [← c31452, ← c31453, k2, ← c31454] at e_492_9
  have hr := e_492_9
  simp only [e_492_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 492 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1963)) := by
  have c31455 := (publicBatchWrapper86_copies982 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31456 := (publicBatchWrapper86_copies983 a h).1
  have c31457 := (publicBatchWrapper86_copies983 a h).2.1
  have c31458 := (publicBatchWrapper86_copies983 a h).2.2.1
  have c31459 := (publicBatchWrapper86_copies983 a h).2.2.2.1
  have c31460 := (publicBatchWrapper86_copies983 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_492_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_10
  simp only [← c31455, ← c31456, ← c31457] at e_492_10
  have e_492_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_11
  simp only [← c31458, ← c31459, k2, ← c31460] at e_492_11
  have hr := e_492_11
  simp only [e_492_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 492 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1964)) := by
  have c31461 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.1
  have c31462 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.1
  have c31463 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.1
  have c31464 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.1
  have c31465 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.1
  have c31466 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_492_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_12
  simp only [← c31461, ← c31462, ← c31463] at e_492_12
  have e_492_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_13
  simp only [← c31464, ← c31465, k2, ← c31466] at e_492_13
  have hr := e_492_13
  simp only [e_492_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 492 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1965)) := by
  have c31467 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31468 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31469 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31470 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31471 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31472 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_492_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_14
  simp only [← c31467, ← c31468, ← c31469] at e_492_14
  have e_492_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_15
  simp only [← c31470, ← c31471, k2, ← c31472] at e_492_15
  have hr := e_492_15
  simp only [e_492_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 492 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1966)) := by
  have c31473 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31474 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31475 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31476 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31477 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31478 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_492_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_16
  simp only [← c31473, ← c31474, ← c31475] at e_492_16
  have e_492_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_17
  simp only [← c31476, ← c31477, k2, ← c31478] at e_492_17
  have hr := e_492_17
  simp only [e_492_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 492 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1967)) := by
  have c31479 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31480 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31481 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31482 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31483 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31484 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_492_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_18
  simp only [← c31479, ← c31480, ← c31481] at e_492_18
  have e_492_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row492 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_492_19
  simp only [← c31482, ← c31483, k2, ← c31484] at e_492_19
  have hr := e_492_19
  simp only [e_492_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 493 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1968)) := by
  have c31485 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31486 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31487 := (publicBatchWrapper86_copies983 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31488 := (publicBatchWrapper86_copies984 a h).1
  have c31489 := (publicBatchWrapper86_copies984 a h).2.1
  have c31490 := (publicBatchWrapper86_copies984 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_493_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_0
  simp only [← c31485, ← c31486, ← c31487] at e_493_0
  have e_493_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_1
  simp only [← c31488, ← c31489, k2, ← c31490] at e_493_1
  have hr := e_493_1
  simp only [e_493_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 493 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1969)) := by
  have c31491 := (publicBatchWrapper86_copies984 a h).2.2.2.1
  have c31492 := (publicBatchWrapper86_copies984 a h).2.2.2.2.1
  have c31493 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.1
  have c31494 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.1
  have c31495 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.1
  have c31496 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_493_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_2
  simp only [← c31491, ← c31492, ← c31493] at e_493_2
  have e_493_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_3
  simp only [← c31494, ← c31495, k2, ← c31496] at e_493_3
  have hr := e_493_3
  simp only [e_493_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 493 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2010)) := by
  have c31497 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.1
  have c31498 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31499 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31500 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31501 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31502 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_493_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_4
  simp only [← c31497, ← c31498, ← c31499] at e_493_4
  have e_493_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_5
  simp only [← c31500, ← c31501, k2, ← c31502] at e_493_5
  have hr := e_493_5
  simp only [e_493_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 493 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2011)) := by
  have c31503 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31504 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31505 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31506 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31507 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31508 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_493_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_6
  simp only [← c31503, ← c31504, ← c31505] at e_493_6
  have e_493_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_7
  simp only [← c31506, ← c31507, k2, ← c31508] at e_493_7
  have hr := e_493_7
  simp only [e_493_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 493 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2012)) := by
  have c31509 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31510 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31511 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31512 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31513 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31514 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_493_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_8
  simp only [← c31509, ← c31510, ← c31511] at e_493_8
  have e_493_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_9
  simp only [← c31512, ← c31513, k2, ← c31514] at e_493_9
  have hr := e_493_9
  simp only [e_493_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 493 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2013)) := by
  have c31515 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31516 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31517 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31518 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31519 := (publicBatchWrapper86_copies984 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31520 := (publicBatchWrapper86_copies985 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_493_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_10
  simp only [← c31515, ← c31516, ← c31517] at e_493_10
  have e_493_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_11
  simp only [← c31518, ← c31519, k2, ← c31520] at e_493_11
  have hr := e_493_11
  simp only [e_493_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 493 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2014)) := by
  have c31521 := (publicBatchWrapper86_copies985 a h).2.1
  have c31522 := (publicBatchWrapper86_copies985 a h).2.2.1
  have c31523 := (publicBatchWrapper86_copies985 a h).2.2.2.1
  have c31524 := (publicBatchWrapper86_copies985 a h).2.2.2.2.1
  have c31525 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.1
  have c31526 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_493_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_12
  simp only [← c31521, ← c31522, ← c31523] at e_493_12
  have e_493_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_13
  simp only [← c31524, ← c31525, k2, ← c31526] at e_493_13
  have hr := e_493_13
  simp only [e_493_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 493 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2015)) := by
  have c31527 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.1
  have c31528 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.1
  have c31529 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.1
  have c31530 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31531 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31532 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_493_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_14
  simp only [← c31527, ← c31528, ← c31529] at e_493_14
  have e_493_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_15
  simp only [← c31530, ← c31531, k2, ← c31532] at e_493_15
  have hr := e_493_15
  simp only [e_493_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 493 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2016)) := by
  have c31533 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31534 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31535 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31536 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31537 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31538 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_493_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_16
  simp only [← c31533, ← c31534, ← c31535] at e_493_16
  have e_493_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_17
  simp only [← c31536, ← c31537, k2, ← c31538] at e_493_17
  have hr := e_493_17
  simp only [e_493_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 493 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2017)) := by
  have c31539 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31540 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31541 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31542 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31543 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31544 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_493_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_18
  simp only [← c31539, ← c31540, ← c31541] at e_493_18
  have e_493_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row493 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_493_19
  simp only [← c31542, ← c31543, k2, ← c31544] at e_493_19
  have hr := e_493_19
  simp only [e_493_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 494 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2018)) := by
  have c31545 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31546 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31547 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31548 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31549 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31550 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_494_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_0
  simp only [← c31545, ← c31546, ← c31547] at e_494_0
  have e_494_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_1
  simp only [← c31548, ← c31549, k2, ← c31550] at e_494_1
  have hr := e_494_1
  simp only [e_494_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 494 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2019)) := by
  have c31551 := (publicBatchWrapper86_copies985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31552 := (publicBatchWrapper86_copies986 a h).1
  have c31553 := (publicBatchWrapper86_copies986 a h).2.1
  have c31554 := (publicBatchWrapper86_copies986 a h).2.2.1
  have c31555 := (publicBatchWrapper86_copies986 a h).2.2.2.1
  have c31556 := (publicBatchWrapper86_copies986 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_494_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_2
  simp only [← c31551, ← c31552, ← c31553] at e_494_2
  have e_494_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_3
  simp only [← c31554, ← c31555, k2, ← c31556] at e_494_3
  have hr := e_494_3
  simp only [e_494_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 494 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2020)) := by
  have c31557 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.1
  have c31558 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.1
  have c31559 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.1
  have c31560 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.1
  have c31561 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.1
  have c31562 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_494_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_4
  simp only [← c31557, ← c31558, ← c31559] at e_494_4
  have e_494_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_5
  simp only [← c31560, ← c31561, k2, ← c31562] at e_494_5
  have hr := e_494_5
  simp only [e_494_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 494 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2021)) := by
  have c31563 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31564 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31565 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31566 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31567 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31568 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_494_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_6
  simp only [← c31563, ← c31564, ← c31565] at e_494_6
  have e_494_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_7
  simp only [← c31566, ← c31567, k2, ← c31568] at e_494_7
  have hr := e_494_7
  simp only [e_494_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 494 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2022)) := by
  have c31569 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31570 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31571 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31572 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31573 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31574 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_494_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_8
  simp only [← c31569, ← c31570, ← c31571] at e_494_8
  have e_494_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_9
  simp only [← c31572, ← c31573, k2, ← c31574] at e_494_9
  have hr := e_494_9
  simp only [e_494_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 494 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2023)) := by
  have c31575 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31576 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31577 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31578 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31579 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31580 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_494_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_10
  simp only [← c31575, ← c31576, ← c31577] at e_494_10
  have e_494_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_11
  simp only [← c31578, ← c31579, k2, ← c31580] at e_494_11
  have hr := e_494_11
  simp only [e_494_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 494 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2024)) := by
  have c31581 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31582 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31583 := (publicBatchWrapper86_copies986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31584 := (publicBatchWrapper86_copies987 a h).1
  have c31585 := (publicBatchWrapper86_copies987 a h).2.1
  have c31586 := (publicBatchWrapper86_copies987 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_494_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_12
  simp only [← c31581, ← c31582, ← c31583] at e_494_12
  have e_494_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_13
  simp only [← c31584, ← c31585, k2, ← c31586] at e_494_13
  have hr := e_494_13
  simp only [e_494_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 494 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2025)) := by
  have c31587 := (publicBatchWrapper86_copies987 a h).2.2.2.1
  have c31588 := (publicBatchWrapper86_copies987 a h).2.2.2.2.1
  have c31589 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.1
  have c31590 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.1
  have c31591 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.1
  have c31592 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_494_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_14
  simp only [← c31587, ← c31588, ← c31589] at e_494_14
  have e_494_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_15
  simp only [← c31590, ← c31591, k2, ← c31592] at e_494_15
  have hr := e_494_15
  simp only [e_494_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 494 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2026)) := by
  have c31593 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.1
  have c31594 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31595 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31596 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31597 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31598 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_494_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_16
  simp only [← c31593, ← c31594, ← c31595] at e_494_16
  have e_494_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_17
  simp only [← c31596, ← c31597, k2, ← c31598] at e_494_17
  have hr := e_494_17
  simp only [e_494_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 494 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2027)) := by
  have c31599 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31600 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31601 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31602 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31603 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31604 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_494_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_18
  simp only [← c31599, ← c31600, ← c31601] at e_494_18
  have e_494_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row494 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_494_19
  simp only [← c31602, ← c31603, k2, ← c31604] at e_494_19
  have hr := e_494_19
  simp only [e_494_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 495 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2028)) := by
  have c31605 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31606 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31607 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31608 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31609 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31610 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_495_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_0
  simp only [← c31605, ← c31606, ← c31607] at e_495_0
  have e_495_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_1
  simp only [← c31608, ← c31609, k2, ← c31610] at e_495_1
  have hr := e_495_1
  simp only [e_495_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 495 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2029)) := by
  have c31611 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31612 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31613 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31614 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31615 := (publicBatchWrapper86_copies987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31616 := (publicBatchWrapper86_copies988 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_495_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_2
  simp only [← c31611, ← c31612, ← c31613] at e_495_2
  have e_495_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_3
  simp only [← c31614, ← c31615, k2, ← c31616] at e_495_3
  have hr := e_495_3
  simp only [e_495_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 495 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2030)) := by
  have c31617 := (publicBatchWrapper86_copies988 a h).2.1
  have c31618 := (publicBatchWrapper86_copies988 a h).2.2.1
  have c31619 := (publicBatchWrapper86_copies988 a h).2.2.2.1
  have c31620 := (publicBatchWrapper86_copies988 a h).2.2.2.2.1
  have c31621 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.1
  have c31622 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_495_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_4
  simp only [← c31617, ← c31618, ← c31619] at e_495_4
  have e_495_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_5
  simp only [← c31620, ← c31621, k2, ← c31622] at e_495_5
  have hr := e_495_5
  simp only [e_495_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 495 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2031)) := by
  have c31623 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.1
  have c31624 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.1
  have c31625 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.1
  have c31626 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31627 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31628 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_495_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_6
  simp only [← c31623, ← c31624, ← c31625] at e_495_6
  have e_495_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_7
  simp only [← c31626, ← c31627, k2, ← c31628] at e_495_7
  have hr := e_495_7
  simp only [e_495_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 495 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2032)) := by
  have c31629 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31630 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31631 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31632 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31633 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31634 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_495_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_8
  simp only [← c31629, ← c31630, ← c31631] at e_495_8
  have e_495_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_9
  simp only [← c31632, ← c31633, k2, ← c31634] at e_495_9
  have hr := e_495_9
  simp only [e_495_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 495 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2033)) := by
  have c31635 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31636 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31637 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31638 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31639 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31640 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_495_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_10
  simp only [← c31635, ← c31636, ← c31637] at e_495_10
  have e_495_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_11
  simp only [← c31638, ← c31639, k2, ← c31640] at e_495_11
  have hr := e_495_11
  simp only [e_495_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 495 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2034)) := by
  have c31641 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31642 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31643 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31644 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31645 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31646 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_495_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_12
  simp only [← c31641, ← c31642, ← c31643] at e_495_12
  have e_495_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_13
  simp only [← c31644, ← c31645, k2, ← c31646] at e_495_13
  have hr := e_495_13
  simp only [e_495_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 495 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2035)) := by
  have c31647 := (publicBatchWrapper86_copies988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31648 := (publicBatchWrapper86_copies989 a h).1
  have c31649 := (publicBatchWrapper86_copies989 a h).2.1
  have c31650 := (publicBatchWrapper86_copies989 a h).2.2.1
  have c31651 := (publicBatchWrapper86_copies989 a h).2.2.2.1
  have c31652 := (publicBatchWrapper86_copies989 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_495_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_14
  simp only [← c31647, ← c31648, ← c31649] at e_495_14
  have e_495_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_15
  simp only [← c31650, ← c31651, k2, ← c31652] at e_495_15
  have hr := e_495_15
  simp only [e_495_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 495 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2036)) := by
  have c31653 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.1
  have c31654 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.1
  have c31655 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.1
  have c31656 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.1
  have c31657 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.1
  have c31658 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_495_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_16
  simp only [← c31653, ← c31654, ← c31655] at e_495_16
  have e_495_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_17
  simp only [← c31656, ← c31657, k2, ← c31658] at e_495_17
  have hr := e_495_17
  simp only [e_495_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 495 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2037)) := by
  have c31659 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31660 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31661 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31662 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31663 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31664 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_495_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_18
  simp only [← c31659, ← c31660, ← c31661] at e_495_18
  have e_495_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row495 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_495_19
  simp only [← c31662, ← c31663, k2, ← c31664] at e_495_19
  have hr := e_495_19
  simp only [e_495_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 496 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2038)) := by
  have c31665 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31666 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31667 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31668 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31669 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31670 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_496_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_0
  simp only [← c31665, ← c31666, ← c31667] at e_496_0
  have e_496_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_1
  simp only [← c31668, ← c31669, k2, ← c31670] at e_496_1
  have hr := e_496_1
  simp only [e_496_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 496 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2039)) := by
  have c31671 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31672 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31673 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31674 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31675 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31676 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_496_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_2
  simp only [← c31671, ← c31672, ← c31673] at e_496_2
  have e_496_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_3
  simp only [← c31674, ← c31675, k2, ← c31676] at e_496_3
  have hr := e_496_3
  simp only [e_496_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 496 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2040)) := by
  have c31677 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31678 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31679 := (publicBatchWrapper86_copies989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31680 := (publicBatchWrapper86_copies990 a h).1
  have c31681 := (publicBatchWrapper86_copies990 a h).2.1
  have c31682 := (publicBatchWrapper86_copies990 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_496_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_4
  simp only [← c31677, ← c31678, ← c31679] at e_496_4
  have e_496_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_5
  simp only [← c31680, ← c31681, k2, ← c31682] at e_496_5
  have hr := e_496_5
  simp only [e_496_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 496 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2041)) := by
  have c31683 := (publicBatchWrapper86_copies990 a h).2.2.2.1
  have c31684 := (publicBatchWrapper86_copies990 a h).2.2.2.2.1
  have c31685 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.1
  have c31686 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.1
  have c31687 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.1
  have c31688 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_496_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_6
  simp only [← c31683, ← c31684, ← c31685] at e_496_6
  have e_496_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_7
  simp only [← c31686, ← c31687, k2, ← c31688] at e_496_7
  have hr := e_496_7
  simp only [e_496_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 496 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2042)) := by
  have c31689 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.1
  have c31690 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31691 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31692 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31693 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31694 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_496_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_8
  simp only [← c31689, ← c31690, ← c31691] at e_496_8
  have e_496_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_9
  simp only [← c31692, ← c31693, k2, ← c31694] at e_496_9
  have hr := e_496_9
  simp only [e_496_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 496 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2043)) := by
  have c31695 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31696 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31697 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31698 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31699 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31700 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_496_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_10
  simp only [← c31695, ← c31696, ← c31697] at e_496_10
  have e_496_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_11
  simp only [← c31698, ← c31699, k2, ← c31700] at e_496_11
  have hr := e_496_11
  simp only [e_496_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 496 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2044)) := by
  have c31701 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31702 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31703 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31704 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31705 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31706 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_496_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_12
  simp only [← c31701, ← c31702, ← c31703] at e_496_12
  have e_496_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_13
  simp only [← c31704, ← c31705, k2, ← c31706] at e_496_13
  have hr := e_496_13
  simp only [e_496_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 496 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2045)) := by
  have c31707 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31708 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31709 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31710 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31711 := (publicBatchWrapper86_copies990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31712 := (publicBatchWrapper86_copies991 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_496_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_14
  simp only [← c31707, ← c31708, ← c31709] at e_496_14
  have e_496_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_15
  simp only [← c31710, ← c31711, k2, ← c31712] at e_496_15
  have hr := e_496_15
  simp only [e_496_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 496 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2046)) := by
  have c31713 := (publicBatchWrapper86_copies991 a h).2.1
  have c31714 := (publicBatchWrapper86_copies991 a h).2.2.1
  have c31715 := (publicBatchWrapper86_copies991 a h).2.2.2.1
  have c31716 := (publicBatchWrapper86_copies991 a h).2.2.2.2.1
  have c31717 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.1
  have c31718 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_496_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_16
  simp only [← c31713, ← c31714, ← c31715] at e_496_16
  have e_496_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_17
  simp only [← c31716, ← c31717, k2, ← c31718] at e_496_17
  have hr := e_496_17
  simp only [e_496_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 496 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2047)) := by
  have c31719 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.1
  have c31720 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.1
  have c31721 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.1
  have c31722 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31723 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31724 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_496_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_18
  simp only [← c31719, ← c31720, ← c31721] at e_496_18
  have e_496_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row496 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_496_19
  simp only [← c31722, ← c31723, k2, ← c31724] at e_496_19
  have hr := e_496_19
  simp only [e_496_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 497 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2048)) := by
  have c31725 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31726 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31727 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31728 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31729 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31730 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_497_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_0
  simp only [← c31725, ← c31726, ← c31727] at e_497_0
  have e_497_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_1
  simp only [← c31728, ← c31729, k2, ← c31730] at e_497_1
  have hr := e_497_1
  simp only [e_497_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 497 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2049)) := by
  have c31731 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31732 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31733 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31734 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31735 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31736 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_497_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_2
  simp only [← c31731, ← c31732, ← c31733] at e_497_2
  have e_497_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_3
  simp only [← c31734, ← c31735, k2, ← c31736] at e_497_3
  have hr := e_497_3
  simp only [e_497_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 497 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2050)) := by
  have c31737 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31738 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31739 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31740 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31741 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31742 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_497_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_4
  simp only [← c31737, ← c31738, ← c31739] at e_497_4
  have e_497_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_5
  simp only [← c31740, ← c31741, k2, ← c31742] at e_497_5
  have hr := e_497_5
  simp only [e_497_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 497 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2051)) := by
  have c31743 := (publicBatchWrapper86_copies991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31744 := (publicBatchWrapper86_copies992 a h).1
  have c31745 := (publicBatchWrapper86_copies992 a h).2.1
  have c31746 := (publicBatchWrapper86_copies992 a h).2.2.1
  have c31747 := (publicBatchWrapper86_copies992 a h).2.2.2.1
  have c31748 := (publicBatchWrapper86_copies992 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_497_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_6
  simp only [← c31743, ← c31744, ← c31745] at e_497_6
  have e_497_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_7
  simp only [← c31746, ← c31747, k2, ← c31748] at e_497_7
  have hr := e_497_7
  simp only [e_497_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 497 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2052)) := by
  have c31749 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.1
  have c31750 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.1
  have c31751 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.1
  have c31752 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.1
  have c31753 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.1
  have c31754 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_497_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_8
  simp only [← c31749, ← c31750, ← c31751] at e_497_8
  have e_497_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_9
  simp only [← c31752, ← c31753, k2, ← c31754] at e_497_9
  have hr := e_497_9
  simp only [e_497_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 497 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2053)) := by
  have c31755 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31756 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31757 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31758 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31759 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31760 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_497_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_10
  simp only [← c31755, ← c31756, ← c31757] at e_497_10
  have e_497_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_11
  simp only [← c31758, ← c31759, k2, ← c31760] at e_497_11
  have hr := e_497_11
  simp only [e_497_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 497 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2054)) := by
  have c31761 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31762 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31763 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31764 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31765 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31766 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_497_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_12
  simp only [← c31761, ← c31762, ← c31763] at e_497_12
  have e_497_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_13
  simp only [← c31764, ← c31765, k2, ← c31766] at e_497_13
  have hr := e_497_13
  simp only [e_497_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 497 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2055)) := by
  have c31767 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31768 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31769 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31770 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31771 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31772 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_497_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_14
  simp only [← c31767, ← c31768, ← c31769] at e_497_14
  have e_497_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_15
  simp only [← c31770, ← c31771, k2, ← c31772] at e_497_15
  have hr := e_497_15
  simp only [e_497_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 497 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2056)) := by
  have c31773 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31774 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31775 := (publicBatchWrapper86_copies992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31776 := (publicBatchWrapper86_copies993 a h).1
  have c31777 := (publicBatchWrapper86_copies993 a h).2.1
  have c31778 := (publicBatchWrapper86_copies993 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_497_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_16
  simp only [← c31773, ← c31774, ← c31775] at e_497_16
  have e_497_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_17
  simp only [← c31776, ← c31777, k2, ← c31778] at e_497_17
  have hr := e_497_17
  simp only [e_497_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 497 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2057)) := by
  have c31779 := (publicBatchWrapper86_copies993 a h).2.2.2.1
  have c31780 := (publicBatchWrapper86_copies993 a h).2.2.2.2.1
  have c31781 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.1
  have c31782 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.1
  have c31783 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.1
  have c31784 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_497_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_18
  simp only [← c31779, ← c31780, ← c31781] at e_497_18
  have e_497_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row497 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_497_19
  simp only [← c31782, ← c31783, k2, ← c31784] at e_497_19
  have hr := e_497_19
  simp only [e_497_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 498 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2058)) := by
  have c31785 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.1
  have c31786 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31787 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31788 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31789 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31790 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_498_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_0
  simp only [← c31785, ← c31786, ← c31787] at e_498_0
  have e_498_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_1
  simp only [← c31788, ← c31789, k2, ← c31790] at e_498_1
  have hr := e_498_1
  simp only [e_498_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 498 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2059)) := by
  have c31791 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31792 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31793 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31794 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31795 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31796 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_498_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_2
  simp only [← c31791, ← c31792, ← c31793] at e_498_2
  have e_498_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_3
  simp only [← c31794, ← c31795, k2, ← c31796] at e_498_3
  have hr := e_498_3
  simp only [e_498_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 498 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2060)) := by
  have c31797 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31798 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31799 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31800 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31801 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31802 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_498_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_4
  simp only [← c31797, ← c31798, ← c31799] at e_498_4
  have e_498_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_5
  simp only [← c31800, ← c31801, k2, ← c31802] at e_498_5
  have hr := e_498_5
  simp only [e_498_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 498 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2061)) := by
  have c31803 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31804 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31805 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31806 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31807 := (publicBatchWrapper86_copies993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31808 := (publicBatchWrapper86_copies994 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_498_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_6
  simp only [← c31803, ← c31804, ← c31805] at e_498_6
  have e_498_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_7
  simp only [← c31806, ← c31807, k2, ← c31808] at e_498_7
  have hr := e_498_7
  simp only [e_498_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 498 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2062)) := by
  have c31809 := (publicBatchWrapper86_copies994 a h).2.1
  have c31810 := (publicBatchWrapper86_copies994 a h).2.2.1
  have c31811 := (publicBatchWrapper86_copies994 a h).2.2.2.1
  have c31812 := (publicBatchWrapper86_copies994 a h).2.2.2.2.1
  have c31813 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.1
  have c31814 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_498_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_8
  simp only [← c31809, ← c31810, ← c31811] at e_498_8
  have e_498_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_9
  simp only [← c31812, ← c31813, k2, ← c31814] at e_498_9
  have hr := e_498_9
  simp only [e_498_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 498 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2063)) := by
  have c31815 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.1
  have c31816 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.1
  have c31817 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.1
  have c31818 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31819 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31820 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_498_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_10
  simp only [← c31815, ← c31816, ← c31817] at e_498_10
  have e_498_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_11
  simp only [← c31818, ← c31819, k2, ← c31820] at e_498_11
  have hr := e_498_11
  simp only [e_498_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 498 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2064)) := by
  have c31821 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31822 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31823 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31824 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31825 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31826 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_498_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_12
  simp only [← c31821, ← c31822, ← c31823] at e_498_12
  have e_498_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_13
  simp only [← c31824, ← c31825, k2, ← c31826] at e_498_13
  have hr := e_498_13
  simp only [e_498_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 498 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2065)) := by
  have c31827 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31828 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31829 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31830 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31831 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31832 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_498_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_14
  simp only [← c31827, ← c31828, ← c31829] at e_498_14
  have e_498_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_15
  simp only [← c31830, ← c31831, k2, ← c31832] at e_498_15
  have hr := e_498_15
  simp only [e_498_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 498 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2066)) := by
  have c31833 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31834 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31835 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31836 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31837 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31838 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_498_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_16
  simp only [← c31833, ← c31834, ← c31835] at e_498_16
  have e_498_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_17
  simp only [← c31836, ← c31837, k2, ← c31838] at e_498_17
  have hr := e_498_17
  simp only [e_498_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 498 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2067)) := by
  have c31839 := (publicBatchWrapper86_copies994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31840 := (publicBatchWrapper86_copies995 a h).1
  have c31841 := (publicBatchWrapper86_copies995 a h).2.1
  have c31842 := (publicBatchWrapper86_copies995 a h).2.2.1
  have c31843 := (publicBatchWrapper86_copies995 a h).2.2.2.1
  have c31844 := (publicBatchWrapper86_copies995 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_498_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_18
  simp only [← c31839, ← c31840, ← c31841] at e_498_18
  have e_498_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row498 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_498_19
  simp only [← c31842, ← c31843, k2, ← c31844] at e_498_19
  have hr := e_498_19
  simp only [e_498_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 499 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2068)) := by
  have c31845 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.1
  have c31846 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.1
  have c31847 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.1
  have c31848 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.1
  have c31849 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.1
  have c31850 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_499_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_0
  simp only [← c31845, ← c31846, ← c31847] at e_499_0
  have e_499_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_1
  simp only [← c31848, ← c31849, k2, ← c31850] at e_499_1
  have hr := e_499_1
  simp only [e_499_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 499 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2069)) := by
  have c31851 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31852 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31853 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31854 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31855 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31856 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_499_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_2
  simp only [← c31851, ← c31852, ← c31853] at e_499_2
  have e_499_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_3
  simp only [← c31854, ← c31855, k2, ← c31856] at e_499_3
  have hr := e_499_3
  simp only [e_499_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 499 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2070)) := by
  have c31857 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31858 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31859 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31860 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31861 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31862 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_499_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_4
  simp only [← c31857, ← c31858, ← c31859] at e_499_4
  have e_499_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_5
  simp only [← c31860, ← c31861, k2, ← c31862] at e_499_5
  have hr := e_499_5
  simp only [e_499_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 499 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2071)) := by
  have c31863 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31864 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31865 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31866 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31867 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31868 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_499_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_6
  simp only [← c31863, ← c31864, ← c31865] at e_499_6
  have e_499_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_7
  simp only [← c31866, ← c31867, k2, ← c31868] at e_499_7
  have hr := e_499_7
  simp only [e_499_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 499 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2072)) := by
  have c31869 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31870 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31871 := (publicBatchWrapper86_copies995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31872 := (publicBatchWrapper86_copies996 a h).1
  have c31873 := (publicBatchWrapper86_copies996 a h).2.1
  have c31874 := (publicBatchWrapper86_copies996 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_499_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_8
  simp only [← c31869, ← c31870, ← c31871] at e_499_8
  have e_499_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_9
  simp only [← c31872, ← c31873, k2, ← c31874] at e_499_9
  have hr := e_499_9
  simp only [e_499_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 499 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2073)) := by
  have c31875 := (publicBatchWrapper86_copies996 a h).2.2.2.1
  have c31876 := (publicBatchWrapper86_copies996 a h).2.2.2.2.1
  have c31877 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.1
  have c31878 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.1
  have c31879 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.1
  have c31880 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_499_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_10
  simp only [← c31875, ← c31876, ← c31877] at e_499_10
  have e_499_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_11
  simp only [← c31878, ← c31879, k2, ← c31880] at e_499_11
  have hr := e_499_11
  simp only [e_499_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 499 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2074)) := by
  have c31881 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.1
  have c31882 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31883 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31884 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31885 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31886 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_499_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_12
  simp only [← c31881, ← c31882, ← c31883] at e_499_12
  have e_499_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_13
  simp only [← c31884, ← c31885, k2, ← c31886] at e_499_13
  have hr := e_499_13
  simp only [e_499_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 499 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2075)) := by
  have c31887 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31888 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31889 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31890 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31891 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31892 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_499_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_14
  simp only [← c31887, ← c31888, ← c31889] at e_499_14
  have e_499_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_15
  simp only [← c31890, ← c31891, k2, ← c31892] at e_499_15
  have hr := e_499_15
  simp only [e_499_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 499 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2076)) := by
  have c31893 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31894 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31895 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31896 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31897 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31898 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_499_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_16
  simp only [← c31893, ← c31894, ← c31895] at e_499_16
  have e_499_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_17
  simp only [← c31896, ← c31897, k2, ← c31898] at e_499_17
  have hr := e_499_17
  simp only [e_499_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 499 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2077)) := by
  have c31899 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31900 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31901 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31902 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31903 := (publicBatchWrapper86_copies996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31904 := (publicBatchWrapper86_copies997 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_499_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_18
  simp only [← c31899, ← c31900, ← c31901] at e_499_18
  have e_499_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row499 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_499_19
  simp only [← c31902, ← c31903, k2, ← c31904] at e_499_19
  have hr := e_499_19
  simp only [e_499_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 500 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2078)) := by
  have c31905 := (publicBatchWrapper86_copies997 a h).2.1
  have c31906 := (publicBatchWrapper86_copies997 a h).2.2.1
  have c31907 := (publicBatchWrapper86_copies997 a h).2.2.2.1
  have c31908 := (publicBatchWrapper86_copies997 a h).2.2.2.2.1
  have c31909 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.1
  have c31910 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_500_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_0
  simp only [← c31905, ← c31906, ← c31907] at e_500_0
  have e_500_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_1
  simp only [← c31908, ← c31909, k2, ← c31910] at e_500_1
  have hr := e_500_1
  simp only [e_500_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 500 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2079)) := by
  have c31911 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.1
  have c31912 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.1
  have c31913 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.1
  have c31914 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31915 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31916 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_500_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_2
  simp only [← c31911, ← c31912, ← c31913] at e_500_2
  have e_500_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_3
  simp only [← c31914, ← c31915, k2, ← c31916] at e_500_3
  have hr := e_500_3
  simp only [e_500_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 500 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2080)) := by
  have c31917 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31918 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31919 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31920 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31921 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31922 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_500_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_4
  simp only [← c31917, ← c31918, ← c31919] at e_500_4
  have e_500_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_5
  simp only [← c31920, ← c31921, k2, ← c31922] at e_500_5
  have hr := e_500_5
  simp only [e_500_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 500 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2081)) := by
  have c31923 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31924 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31925 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31926 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31927 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31928 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_500_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_6
  simp only [← c31923, ← c31924, ← c31925] at e_500_6
  have e_500_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_7
  simp only [← c31926, ← c31927, k2, ← c31928] at e_500_7
  have hr := e_500_7
  simp only [e_500_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 500 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2082)) := by
  have c31929 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31930 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31931 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31932 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31933 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31934 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_500_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_8
  simp only [← c31929, ← c31930, ← c31931] at e_500_8
  have e_500_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_9
  simp only [← c31932, ← c31933, k2, ← c31934] at e_500_9
  have hr := e_500_9
  simp only [e_500_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 500 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2083)) := by
  have c31935 := (publicBatchWrapper86_copies997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31936 := (publicBatchWrapper86_copies998 a h).1
  have c31937 := (publicBatchWrapper86_copies998 a h).2.1
  have c31938 := (publicBatchWrapper86_copies998 a h).2.2.1
  have c31939 := (publicBatchWrapper86_copies998 a h).2.2.2.1
  have c31940 := (publicBatchWrapper86_copies998 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_500_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_10
  simp only [← c31935, ← c31936, ← c31937] at e_500_10
  have e_500_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_11
  simp only [← c31938, ← c31939, k2, ← c31940] at e_500_11
  have hr := e_500_11
  simp only [e_500_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 500 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2084)) := by
  have c31941 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.1
  have c31942 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.1
  have c31943 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.1
  have c31944 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.1
  have c31945 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.1
  have c31946 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_500_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_12
  simp only [← c31941, ← c31942, ← c31943] at e_500_12
  have e_500_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_13
  simp only [← c31944, ← c31945, k2, ← c31946] at e_500_13
  have hr := e_500_13
  simp only [e_500_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 500 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2085)) := by
  have c31947 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31948 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31949 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31950 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31951 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31952 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_500_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_14
  simp only [← c31947, ← c31948, ← c31949] at e_500_14
  have e_500_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_15
  simp only [← c31950, ← c31951, k2, ← c31952] at e_500_15
  have hr := e_500_15
  simp only [e_500_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 500 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2086)) := by
  have c31953 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31954 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31955 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31956 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31957 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31958 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_500_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_16
  simp only [← c31953, ← c31954, ← c31955] at e_500_16
  have e_500_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_17
  simp only [← c31956, ← c31957, k2, ← c31958] at e_500_17
  have hr := e_500_17
  simp only [e_500_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 500 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2087)) := by
  have c31959 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31960 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31961 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31962 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31963 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31964 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_500_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_18
  simp only [← c31959, ← c31960, ← c31961] at e_500_18
  have e_500_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row500 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_500_19
  simp only [← c31962, ← c31963, k2, ← c31964] at e_500_19
  have hr := e_500_19
  simp only [e_500_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 501 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2088)) := by
  have c31965 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31966 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31967 := (publicBatchWrapper86_copies998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c31968 := (publicBatchWrapper86_copies999 a h).1
  have c31969 := (publicBatchWrapper86_copies999 a h).2.1
  have c31970 := (publicBatchWrapper86_copies999 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_501_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_0
  simp only [← c31965, ← c31966, ← c31967] at e_501_0
  have e_501_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_1
  simp only [← c31968, ← c31969, k2, ← c31970] at e_501_1
  have hr := e_501_1
  simp only [e_501_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 501 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2089)) := by
  have c31971 := (publicBatchWrapper86_copies999 a h).2.2.2.1
  have c31972 := (publicBatchWrapper86_copies999 a h).2.2.2.2.1
  have c31973 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.1
  have c31974 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.1
  have c31975 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.1
  have c31976 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_501_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_2
  simp only [← c31971, ← c31972, ← c31973] at e_501_2
  have e_501_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_3
  simp only [← c31974, ← c31975, k2, ← c31976] at e_501_3
  have hr := e_501_3
  simp only [e_501_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 501 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2090)) := by
  have c31977 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.1
  have c31978 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.1
  have c31979 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c31980 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31981 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31982 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_501_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_4
  simp only [← c31977, ← c31978, ← c31979] at e_501_4
  have e_501_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_5
  simp only [← c31980, ← c31981, k2, ← c31982] at e_501_5
  have hr := e_501_5
  simp only [e_501_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 501 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2091)) := by
  have c31983 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31984 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31985 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31986 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31987 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31988 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_501_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_6
  simp only [← c31983, ← c31984, ← c31985] at e_501_6
  have e_501_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_7
  simp only [← c31986, ← c31987, k2, ← c31988] at e_501_7
  have hr := e_501_7
  simp only [e_501_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 501 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2092)) := by
  have c31989 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31990 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31991 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31992 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31993 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31994 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_501_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_8
  simp only [← c31989, ← c31990, ← c31991] at e_501_8
  have e_501_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_9
  simp only [← c31992, ← c31993, k2, ← c31994] at e_501_9
  have hr := e_501_9
  simp only [e_501_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 501 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2093)) := by
  have c31995 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31996 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31997 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31998 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c31999 := (publicBatchWrapper86_copies999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32000 := (publicBatchWrapper86_copies1000 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_501_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_10
  simp only [← c31995, ← c31996, ← c31997] at e_501_10
  have e_501_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_11
  simp only [← c31998, ← c31999, k2, ← c32000] at e_501_11
  have hr := e_501_11
  simp only [e_501_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 501 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2094)) := by
  have c32001 := (publicBatchWrapper86_copies1000 a h).2.1
  have c32002 := (publicBatchWrapper86_copies1000 a h).2.2.1
  have c32003 := (publicBatchWrapper86_copies1000 a h).2.2.2.1
  have c32004 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.1
  have c32005 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.1
  have c32006 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_501_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_12
  simp only [← c32001, ← c32002, ← c32003] at e_501_12
  have e_501_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_13
  simp only [← c32004, ← c32005, k2, ← c32006] at e_501_13
  have hr := e_501_13
  simp only [e_501_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 501 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2095)) := by
  have c32007 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.1
  have c32008 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.1
  have c32009 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.1
  have c32010 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32011 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32012 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_501_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_14
  simp only [← c32007, ← c32008, ← c32009] at e_501_14
  have e_501_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_15
  simp only [← c32010, ← c32011, k2, ← c32012] at e_501_15
  have hr := e_501_15
  simp only [e_501_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 501 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2096)) := by
  have c32013 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32014 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32015 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32016 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32017 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32018 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_501_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_16
  simp only [← c32013, ← c32014, ← c32015] at e_501_16
  have e_501_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_17
  simp only [← c32016, ← c32017, k2, ← c32018] at e_501_17
  have hr := e_501_17
  simp only [e_501_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 501 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2097)) := by
  have c32019 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32020 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32021 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32022 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32023 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32024 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_501_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_18
  simp only [← c32019, ← c32020, ← c32021] at e_501_18
  have e_501_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row501 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_501_19
  simp only [← c32022, ← c32023, k2, ← c32024] at e_501_19
  have hr := e_501_19
  simp only [e_501_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 502 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2098)) := by
  have c32025 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32026 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32027 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32028 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32029 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32030 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_502_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_0
  simp only [← c32025, ← c32026, ← c32027] at e_502_0
  have e_502_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_1
  simp only [← c32028, ← c32029, k2, ← c32030] at e_502_1
  have hr := e_502_1
  simp only [e_502_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 502 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2099)) := by
  have c32031 := (publicBatchWrapper86_copies1000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32032 := (publicBatchWrapper86_copies1001 a h).1
  have c32033 := (publicBatchWrapper86_copies1001 a h).2.1
  have c32034 := (publicBatchWrapper86_copies1001 a h).2.2.1
  have c32035 := (publicBatchWrapper86_copies1001 a h).2.2.2.1
  have c32036 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_502_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_2
  simp only [← c32031, ← c32032, ← c32033] at e_502_2
  have e_502_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_3
  simp only [← c32034, ← c32035, k2, ← c32036] at e_502_3
  have hr := e_502_3
  simp only [e_502_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 502 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2100)) := by
  have c32037 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.1
  have c32038 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.1
  have c32039 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.1
  have c32040 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.1
  have c32041 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.1
  have c32042 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_502_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_4
  simp only [← c32037, ← c32038, ← c32039] at e_502_4
  have e_502_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_5
  simp only [← c32040, ← c32041, k2, ← c32042] at e_502_5
  have hr := e_502_5
  simp only [e_502_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 502 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2101)) := by
  have c32043 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32044 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32045 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32046 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32047 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32048 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_502_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_6
  simp only [← c32043, ← c32044, ← c32045] at e_502_6
  have e_502_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_7
  simp only [← c32046, ← c32047, k2, ← c32048] at e_502_7
  have hr := e_502_7
  simp only [e_502_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 502 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2102)) := by
  have c32049 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32050 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32051 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32052 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32053 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32054 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_502_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_8
  simp only [← c32049, ← c32050, ← c32051] at e_502_8
  have e_502_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_9
  simp only [← c32052, ← c32053, k2, ← c32054] at e_502_9
  have hr := e_502_9
  simp only [e_502_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 502 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2103)) := by
  have c32055 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32056 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32057 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32058 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32059 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32060 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_502_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_10
  simp only [← c32055, ← c32056, ← c32057] at e_502_10
  have e_502_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_11
  simp only [← c32058, ← c32059, k2, ← c32060] at e_502_11
  have hr := e_502_11
  simp only [e_502_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 502 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2104)) := by
  have c32061 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32062 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32063 := (publicBatchWrapper86_copies1001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32064 := (publicBatchWrapper86_copies1002 a h).1
  have c32065 := (publicBatchWrapper86_copies1002 a h).2.1
  have c32066 := (publicBatchWrapper86_copies1002 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_502_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_12
  simp only [← c32061, ← c32062, ← c32063] at e_502_12
  have e_502_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_13
  simp only [← c32064, ← c32065, k2, ← c32066] at e_502_13
  have hr := e_502_13
  simp only [e_502_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 502 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2105)) := by
  have c32067 := (publicBatchWrapper86_copies1002 a h).2.2.2.1
  have c32068 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.1
  have c32069 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.1
  have c32070 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.1
  have c32071 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.1
  have c32072 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_502_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_14
  simp only [← c32067, ← c32068, ← c32069] at e_502_14
  have e_502_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_15
  simp only [← c32070, ← c32071, k2, ← c32072] at e_502_15
  have hr := e_502_15
  simp only [e_502_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 502 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2106)) := by
  have c32073 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.1
  have c32074 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32075 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32076 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32077 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32078 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_502_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_16
  simp only [← c32073, ← c32074, ← c32075] at e_502_16
  have e_502_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_17
  simp only [← c32076, ← c32077, k2, ← c32078] at e_502_17
  have hr := e_502_17
  simp only [e_502_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 502 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2107)) := by
  have c32079 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32080 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32081 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32082 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32083 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32084 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_502_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_18
  simp only [← c32079, ← c32080, ← c32081] at e_502_18
  have e_502_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row502 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_502_19
  simp only [← c32082, ← c32083, k2, ← c32084] at e_502_19
  have hr := e_502_19
  simp only [e_502_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 503 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2108)) := by
  have c32085 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32086 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32087 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32088 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32089 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32090 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_503_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_0
  simp only [← c32085, ← c32086, ← c32087] at e_503_0
  have e_503_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_1
  simp only [← c32088, ← c32089, k2, ← c32090] at e_503_1
  have hr := e_503_1
  simp only [e_503_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 503 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2109)) := by
  have c32091 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32092 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32093 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32094 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32095 := (publicBatchWrapper86_copies1002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32096 := (publicBatchWrapper86_copies1003 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_503_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_2
  simp only [← c32091, ← c32092, ← c32093] at e_503_2
  have e_503_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_3
  simp only [← c32094, ← c32095, k2, ← c32096] at e_503_3
  have hr := e_503_3
  simp only [e_503_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 503 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2110)) := by
  have c32097 := (publicBatchWrapper86_copies1003 a h).2.1
  have c32098 := (publicBatchWrapper86_copies1003 a h).2.2.1
  have c32099 := (publicBatchWrapper86_copies1003 a h).2.2.2.1
  have c32100 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.1
  have c32101 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.1
  have c32102 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_503_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_4
  simp only [← c32097, ← c32098, ← c32099] at e_503_4
  have e_503_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_5
  simp only [← c32100, ← c32101, k2, ← c32102] at e_503_5
  have hr := e_503_5
  simp only [e_503_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 503 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2111)) := by
  have c32103 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.1
  have c32104 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.1
  have c32105 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.1
  have c32106 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32107 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32108 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_503_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_6
  simp only [← c32103, ← c32104, ← c32105] at e_503_6
  have e_503_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_7
  simp only [← c32106, ← c32107, k2, ← c32108] at e_503_7
  have hr := e_503_7
  simp only [e_503_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 503 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2112)) := by
  have c32109 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32110 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32111 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32112 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32113 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32114 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_503_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_8
  simp only [← c32109, ← c32110, ← c32111] at e_503_8
  have e_503_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_9
  simp only [← c32112, ← c32113, k2, ← c32114] at e_503_9
  have hr := e_503_9
  simp only [e_503_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 503 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2113)) := by
  have c32115 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32116 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32117 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32118 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32119 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32120 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_503_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_10
  simp only [← c32115, ← c32116, ← c32117] at e_503_10
  have e_503_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_11
  simp only [← c32118, ← c32119, k2, ← c32120] at e_503_11
  have hr := e_503_11
  simp only [e_503_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 503 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2114)) := by
  have c32121 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32122 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32123 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32124 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32125 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32126 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_503_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_12
  simp only [← c32121, ← c32122, ← c32123] at e_503_12
  have e_503_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_13
  simp only [← c32124, ← c32125, k2, ← c32126] at e_503_13
  have hr := e_503_13
  simp only [e_503_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 503 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2115)) := by
  have c32127 := (publicBatchWrapper86_copies1003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32128 := (publicBatchWrapper86_copies1004 a h).1
  have c32129 := (publicBatchWrapper86_copies1004 a h).2.1
  have c32130 := (publicBatchWrapper86_copies1004 a h).2.2.1
  have c32131 := (publicBatchWrapper86_copies1004 a h).2.2.2.1
  have c32132 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_503_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_14
  simp only [← c32127, ← c32128, ← c32129] at e_503_14
  have e_503_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_15
  simp only [← c32130, ← c32131, k2, ← c32132] at e_503_15
  have hr := e_503_15
  simp only [e_503_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 503 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2116)) := by
  have c32133 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.1
  have c32134 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.1
  have c32135 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.1
  have c32136 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.1
  have c32137 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.1
  have c32138 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_503_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_16
  simp only [← c32133, ← c32134, ← c32135] at e_503_16
  have e_503_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_17
  simp only [← c32136, ← c32137, k2, ← c32138] at e_503_17
  have hr := e_503_17
  simp only [e_503_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 503 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2117)) := by
  have c32139 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32140 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32141 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32142 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32143 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32144 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_503_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_18
  simp only [← c32139, ← c32140, ← c32141] at e_503_18
  have e_503_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row503 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_503_19
  simp only [← c32142, ← c32143, k2, ← c32144] at e_503_19
  have hr := e_503_19
  simp only [e_503_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 504 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2118)) := by
  have c32145 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32146 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32147 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32148 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32149 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32150 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_504_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_0
  simp only [← c32145, ← c32146, ← c32147] at e_504_0
  have e_504_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_1
  simp only [← c32148, ← c32149, k2, ← c32150] at e_504_1
  have hr := e_504_1
  simp only [e_504_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 504 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2119)) := by
  have c32151 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32152 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32153 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32154 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32155 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32156 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_504_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_2
  simp only [← c32151, ← c32152, ← c32153] at e_504_2
  have e_504_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_3
  simp only [← c32154, ← c32155, k2, ← c32156] at e_504_3
  have hr := e_504_3
  simp only [e_504_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 504 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2120)) := by
  have c32157 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32158 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32159 := (publicBatchWrapper86_copies1004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32160 := (publicBatchWrapper86_copies1005 a h).1
  have c32161 := (publicBatchWrapper86_copies1005 a h).2.1
  have c32162 := (publicBatchWrapper86_copies1005 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_504_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_4
  simp only [← c32157, ← c32158, ← c32159] at e_504_4
  have e_504_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_5
  simp only [← c32160, ← c32161, k2, ← c32162] at e_504_5
  have hr := e_504_5
  simp only [e_504_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 504 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2121)) := by
  have c32163 := (publicBatchWrapper86_copies1005 a h).2.2.2.1
  have c32164 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.1
  have c32165 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.1
  have c32166 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.1
  have c32167 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.1
  have c32168 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_504_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_6
  simp only [← c32163, ← c32164, ← c32165] at e_504_6
  have e_504_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_7
  simp only [← c32166, ← c32167, k2, ← c32168] at e_504_7
  have hr := e_504_7
  simp only [e_504_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 504 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2122)) := by
  have c32169 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.1
  have c32170 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32171 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32172 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32173 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32174 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_504_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_8
  simp only [← c32169, ← c32170, ← c32171] at e_504_8
  have e_504_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_9
  simp only [← c32172, ← c32173, k2, ← c32174] at e_504_9
  have hr := e_504_9
  simp only [e_504_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 504 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2123)) := by
  have c32175 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32176 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32177 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32178 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32179 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32180 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_504_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_10
  simp only [← c32175, ← c32176, ← c32177] at e_504_10
  have e_504_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_11
  simp only [← c32178, ← c32179, k2, ← c32180] at e_504_11
  have hr := e_504_11
  simp only [e_504_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 504 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2124)) := by
  have c32181 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32182 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32183 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32184 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32185 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32186 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_504_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_12
  simp only [← c32181, ← c32182, ← c32183] at e_504_12
  have e_504_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_13
  simp only [← c32184, ← c32185, k2, ← c32186] at e_504_13
  have hr := e_504_13
  simp only [e_504_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 504 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2125)) := by
  have c32187 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32188 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32189 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32190 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32191 := (publicBatchWrapper86_copies1005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32192 := (publicBatchWrapper86_copies1006 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_504_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_14
  simp only [← c32187, ← c32188, ← c32189] at e_504_14
  have e_504_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_15
  simp only [← c32190, ← c32191, k2, ← c32192] at e_504_15
  have hr := e_504_15
  simp only [e_504_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 504 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2126)) := by
  have c32193 := (publicBatchWrapper86_copies1006 a h).2.1
  have c32194 := (publicBatchWrapper86_copies1006 a h).2.2.1
  have c32195 := (publicBatchWrapper86_copies1006 a h).2.2.2.1
  have c32196 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.1
  have c32197 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.1
  have c32198 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_504_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_16
  simp only [← c32193, ← c32194, ← c32195] at e_504_16
  have e_504_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_17
  simp only [← c32196, ← c32197, k2, ← c32198] at e_504_17
  have hr := e_504_17
  simp only [e_504_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 504 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2127)) := by
  have c32199 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.1
  have c32200 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.1
  have c32201 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.1
  have c32202 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32203 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32204 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_504_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_18
  simp only [← c32199, ← c32200, ← c32201] at e_504_18
  have e_504_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row504 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_504_19
  simp only [← c32202, ← c32203, k2, ← c32204] at e_504_19
  have hr := e_504_19
  simp only [e_504_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 505 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2128)) := by
  have c32205 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32206 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32207 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32208 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32209 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32210 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_505_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_0
  simp only [← c32205, ← c32206, ← c32207] at e_505_0
  have e_505_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_1
  simp only [← c32208, ← c32209, k2, ← c32210] at e_505_1
  have hr := e_505_1
  simp only [e_505_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 505 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2129)) := by
  have c32211 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32212 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32213 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32214 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32215 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32216 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_505_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_2
  simp only [← c32211, ← c32212, ← c32213] at e_505_2
  have e_505_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_3
  simp only [← c32214, ← c32215, k2, ← c32216] at e_505_3
  have hr := e_505_3
  simp only [e_505_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 505 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2130)) := by
  have c32217 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32218 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32219 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32220 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32221 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32222 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_505_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_4
  simp only [← c32217, ← c32218, ← c32219] at e_505_4
  have e_505_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_5
  simp only [← c32220, ← c32221, k2, ← c32222] at e_505_5
  have hr := e_505_5
  simp only [e_505_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 505 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2131)) := by
  have c32223 := (publicBatchWrapper86_copies1006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32224 := (publicBatchWrapper86_copies1007 a h).1
  have c32225 := (publicBatchWrapper86_copies1007 a h).2.1
  have c32226 := (publicBatchWrapper86_copies1007 a h).2.2.1
  have c32227 := (publicBatchWrapper86_copies1007 a h).2.2.2.1
  have c32228 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_505_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_6
  simp only [← c32223, ← c32224, ← c32225] at e_505_6
  have e_505_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_7
  simp only [← c32226, ← c32227, k2, ← c32228] at e_505_7
  have hr := e_505_7
  simp only [e_505_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 505 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2132)) := by
  have c32229 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.1
  have c32230 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.1
  have c32231 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.1
  have c32232 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.1
  have c32233 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.1
  have c32234 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_505_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_8
  simp only [← c32229, ← c32230, ← c32231] at e_505_8
  have e_505_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_9
  simp only [← c32232, ← c32233, k2, ← c32234] at e_505_9
  have hr := e_505_9
  simp only [e_505_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 505 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2133)) := by
  have c32235 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32236 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32237 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32238 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32239 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32240 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_505_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_10
  simp only [← c32235, ← c32236, ← c32237] at e_505_10
  have e_505_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_11
  simp only [← c32238, ← c32239, k2, ← c32240] at e_505_11
  have hr := e_505_11
  simp only [e_505_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 505 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2134)) := by
  have c32241 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32242 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32243 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32244 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32245 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32246 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_505_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_12
  simp only [← c32241, ← c32242, ← c32243] at e_505_12
  have e_505_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_13
  simp only [← c32244, ← c32245, k2, ← c32246] at e_505_13
  have hr := e_505_13
  simp only [e_505_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 505 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2135)) := by
  have c32247 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32248 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32249 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32250 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32251 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32252 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_505_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_14
  simp only [← c32247, ← c32248, ← c32249] at e_505_14
  have e_505_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_15
  simp only [← c32250, ← c32251, k2, ← c32252] at e_505_15
  have hr := e_505_15
  simp only [e_505_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 505 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2136)) := by
  have c32253 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32254 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32255 := (publicBatchWrapper86_copies1007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32256 := (publicBatchWrapper86_copies1008 a h).1
  have c32257 := (publicBatchWrapper86_copies1008 a h).2.1
  have c32258 := (publicBatchWrapper86_copies1008 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_505_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_16
  simp only [← c32253, ← c32254, ← c32255] at e_505_16
  have e_505_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_17
  simp only [← c32256, ← c32257, k2, ← c32258] at e_505_17
  have hr := e_505_17
  simp only [e_505_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 505 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2137)) := by
  have c32259 := (publicBatchWrapper86_copies1008 a h).2.2.2.1
  have c32260 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.1
  have c32261 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.1
  have c32262 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.1
  have c32263 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.1
  have c32264 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_505_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_18
  simp only [← c32259, ← c32260, ← c32261] at e_505_18
  have e_505_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row505 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_505_19
  simp only [← c32262, ← c32263, k2, ← c32264] at e_505_19
  have hr := e_505_19
  simp only [e_505_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 506 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2138)) := by
  have c32265 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.1
  have c32266 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32267 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32268 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32269 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32270 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_506_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_0
  simp only [← c32265, ← c32266, ← c32267] at e_506_0
  have e_506_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_1
  simp only [← c32268, ← c32269, k2, ← c32270] at e_506_1
  have hr := e_506_1
  simp only [e_506_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 506 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2139)) := by
  have c32271 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32272 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32273 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32274 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32275 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32276 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_506_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_2
  simp only [← c32271, ← c32272, ← c32273] at e_506_2
  have e_506_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_3
  simp only [← c32274, ← c32275, k2, ← c32276] at e_506_3
  have hr := e_506_3
  simp only [e_506_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 506 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2140)) := by
  have c32277 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32278 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32279 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32280 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32281 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32282 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_506_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_4
  simp only [← c32277, ← c32278, ← c32279] at e_506_4
  have e_506_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_5
  simp only [← c32280, ← c32281, k2, ← c32282] at e_506_5
  have hr := e_506_5
  simp only [e_506_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 506 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2141)) := by
  have c32283 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32284 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32285 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32286 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32287 := (publicBatchWrapper86_copies1008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32288 := (publicBatchWrapper86_copies1009 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_506_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_6
  simp only [← c32283, ← c32284, ← c32285] at e_506_6
  have e_506_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_7
  simp only [← c32286, ← c32287, k2, ← c32288] at e_506_7
  have hr := e_506_7
  simp only [e_506_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 506 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2142)) := by
  have c32289 := (publicBatchWrapper86_copies1009 a h).2.1
  have c32290 := (publicBatchWrapper86_copies1009 a h).2.2.1
  have c32291 := (publicBatchWrapper86_copies1009 a h).2.2.2.1
  have c32292 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.1
  have c32293 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.1
  have c32294 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_506_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_8
  simp only [← c32289, ← c32290, ← c32291] at e_506_8
  have e_506_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_9
  simp only [← c32292, ← c32293, k2, ← c32294] at e_506_9
  have hr := e_506_9
  simp only [e_506_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 506 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2143)) := by
  have c32295 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.1
  have c32296 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.1
  have c32297 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.1
  have c32298 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32299 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32300 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_506_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_10
  simp only [← c32295, ← c32296, ← c32297] at e_506_10
  have e_506_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_11
  simp only [← c32298, ← c32299, k2, ← c32300] at e_506_11
  have hr := e_506_11
  simp only [e_506_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 506 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2144)) := by
  have c32301 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32302 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32303 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32304 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32305 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32306 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_506_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_12
  simp only [← c32301, ← c32302, ← c32303] at e_506_12
  have e_506_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_13
  simp only [← c32304, ← c32305, k2, ← c32306] at e_506_13
  have hr := e_506_13
  simp only [e_506_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 506 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2145)) := by
  have c32307 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32308 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32309 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32310 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32311 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32312 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_506_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_14
  simp only [← c32307, ← c32308, ← c32309] at e_506_14
  have e_506_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_15
  simp only [← c32310, ← c32311, k2, ← c32312] at e_506_15
  have hr := e_506_15
  simp only [e_506_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 506 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2146)) := by
  have c32313 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32314 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32315 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32316 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32317 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32318 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_506_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_16
  simp only [← c32313, ← c32314, ← c32315] at e_506_16
  have e_506_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_17
  simp only [← c32316, ← c32317, k2, ← c32318] at e_506_17
  have hr := e_506_17
  simp only [e_506_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 506 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2147)) := by
  have c32319 := (publicBatchWrapper86_copies1009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32320 := (publicBatchWrapper86_copies1010 a h).1
  have c32321 := (publicBatchWrapper86_copies1010 a h).2.1
  have c32322 := (publicBatchWrapper86_copies1010 a h).2.2.1
  have c32323 := (publicBatchWrapper86_copies1010 a h).2.2.2.1
  have c32324 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_506_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_18
  simp only [← c32319, ← c32320, ← c32321] at e_506_18
  have e_506_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row506 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_506_19
  simp only [← c32322, ← c32323, k2, ← c32324] at e_506_19
  have hr := e_506_19
  simp only [e_506_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 507 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2148)) := by
  have c32325 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.1
  have c32326 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.1
  have c32327 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.1
  have c32328 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.1
  have c32329 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.1
  have c32330 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_507_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_0
  simp only [← c32325, ← c32326, ← c32327] at e_507_0
  have e_507_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_1
  simp only [← c32328, ← c32329, k2, ← c32330] at e_507_1
  have hr := e_507_1
  simp only [e_507_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 507 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2149)) := by
  have c32331 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32332 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32333 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32334 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32335 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32336 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_507_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_2
  simp only [← c32331, ← c32332, ← c32333] at e_507_2
  have e_507_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_3
  simp only [← c32334, ← c32335, k2, ← c32336] at e_507_3
  have hr := e_507_3
  simp only [e_507_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 507 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2150)) := by
  have c32337 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32338 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32339 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32340 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32341 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32342 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_507_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_4
  simp only [← c32337, ← c32338, ← c32339] at e_507_4
  have e_507_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_5
  simp only [← c32340, ← c32341, k2, ← c32342] at e_507_5
  have hr := e_507_5
  simp only [e_507_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 507 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2151)) := by
  have c32343 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32344 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32345 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32346 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32347 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32348 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_507_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_6
  simp only [← c32343, ← c32344, ← c32345] at e_507_6
  have e_507_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_7
  simp only [← c32346, ← c32347, k2, ← c32348] at e_507_7
  have hr := e_507_7
  simp only [e_507_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 507 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2152)) := by
  have c32349 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32350 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32351 := (publicBatchWrapper86_copies1010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32352 := (publicBatchWrapper86_copies1011 a h).1
  have c32353 := (publicBatchWrapper86_copies1011 a h).2.1
  have c32354 := (publicBatchWrapper86_copies1011 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_507_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_8
  simp only [← c32349, ← c32350, ← c32351] at e_507_8
  have e_507_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_9
  simp only [← c32352, ← c32353, k2, ← c32354] at e_507_9
  have hr := e_507_9
  simp only [e_507_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 507 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2153)) := by
  have c32355 := (publicBatchWrapper86_copies1011 a h).2.2.2.1
  have c32356 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.1
  have c32357 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.1
  have c32358 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.1
  have c32359 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.1
  have c32360 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_507_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_10
  simp only [← c32355, ← c32356, ← c32357] at e_507_10
  have e_507_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_11
  simp only [← c32358, ← c32359, k2, ← c32360] at e_507_11
  have hr := e_507_11
  simp only [e_507_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 507 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2154)) := by
  have c32361 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.1
  have c32362 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32363 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32364 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32365 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32366 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_507_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_12
  simp only [← c32361, ← c32362, ← c32363] at e_507_12
  have e_507_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_13
  simp only [← c32364, ← c32365, k2, ← c32366] at e_507_13
  have hr := e_507_13
  simp only [e_507_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 507 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2155)) := by
  have c32367 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32368 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32369 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32370 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32371 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32372 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_507_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_14
  simp only [← c32367, ← c32368, ← c32369] at e_507_14
  have e_507_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_15
  simp only [← c32370, ← c32371, k2, ← c32372] at e_507_15
  have hr := e_507_15
  simp only [e_507_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 507 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2156)) := by
  have c32373 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32374 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32375 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32376 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32377 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32378 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_507_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_16
  simp only [← c32373, ← c32374, ← c32375] at e_507_16
  have e_507_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_17
  simp only [← c32376, ← c32377, k2, ← c32378] at e_507_17
  have hr := e_507_17
  simp only [e_507_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 507 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2157)) := by
  have c32379 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32380 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32381 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32382 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32383 := (publicBatchWrapper86_copies1011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32384 := (publicBatchWrapper86_copies1012 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_507_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_18
  simp only [← c32379, ← c32380, ← c32381] at e_507_18
  have e_507_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row507 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_507_19
  simp only [← c32382, ← c32383, k2, ← c32384] at e_507_19
  have hr := e_507_19
  simp only [e_507_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 508 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2158)) := by
  have c32385 := (publicBatchWrapper86_copies1012 a h).2.1
  have c32386 := (publicBatchWrapper86_copies1012 a h).2.2.1
  have c32387 := (publicBatchWrapper86_copies1012 a h).2.2.2.1
  have c32388 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.1
  have c32389 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.1
  have c32390 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_508_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_0
  simp only [← c32385, ← c32386, ← c32387] at e_508_0
  have e_508_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_1
  simp only [← c32388, ← c32389, k2, ← c32390] at e_508_1
  have hr := e_508_1
  simp only [e_508_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 508 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2159)) := by
  have c32391 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.1
  have c32392 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.1
  have c32393 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.1
  have c32394 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32395 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32396 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_508_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_2
  simp only [← c32391, ← c32392, ← c32393] at e_508_2
  have e_508_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_3
  simp only [← c32394, ← c32395, k2, ← c32396] at e_508_3
  have hr := e_508_3
  simp only [e_508_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 508 23) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2160)) := by
  have c32397 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32398 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32399 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32400 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32401 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32402 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_508_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_4
  simp only [← c32397, ← c32398, ← c32399] at e_508_4
  have e_508_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_5
  simp only [← c32400, ← c32401, k2, ← c32402] at e_508_5
  have hr := e_508_5
  simp only [e_508_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 508 31) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2161)) := by
  have c32403 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32404 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32405 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32406 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32407 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32408 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_508_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_6
  simp only [← c32403, ← c32404, ← c32405] at e_508_6
  have e_508_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_7
  simp only [← c32406, ← c32407, k2, ← c32408] at e_508_7
  have hr := e_508_7
  simp only [e_508_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 508 39) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2162)) := by
  have c32409 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32410 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32411 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32412 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32413 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32414 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_508_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_8
  simp only [← c32409, ← c32410, ← c32411] at e_508_8
  have e_508_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_9
  simp only [← c32412, ← c32413, k2, ← c32414] at e_508_9
  have hr := e_508_9
  simp only [e_508_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 508 47) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2163)) := by
  have c32415 := (publicBatchWrapper86_copies1012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32416 := (publicBatchWrapper86_copies1013 a h).1
  have c32417 := (publicBatchWrapper86_copies1013 a h).2.1
  have c32418 := (publicBatchWrapper86_copies1013 a h).2.2.1
  have c32419 := (publicBatchWrapper86_copies1013 a h).2.2.2.1
  have c32420 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_508_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_10
  simp only [← c32415, ← c32416, ← c32417] at e_508_10
  have e_508_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_11
  simp only [← c32418, ← c32419, k2, ← c32420] at e_508_11
  have hr := e_508_11
  simp only [e_508_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 508 55) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2164)) := by
  have c32421 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.1
  have c32422 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.1
  have c32423 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.1
  have c32424 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.1
  have c32425 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.1
  have c32426 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_508_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_12
  simp only [← c32421, ← c32422, ← c32423] at e_508_12
  have e_508_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_13
  simp only [← c32424, ← c32425, k2, ← c32426] at e_508_13
  have hr := e_508_13
  simp only [e_508_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 508 63) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2165)) := by
  have c32427 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32428 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32429 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32430 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32431 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32432 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_508_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_14
  simp only [← c32427, ← c32428, ← c32429] at e_508_14
  have e_508_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_15
  simp only [← c32430, ← c32431, k2, ← c32432] at e_508_15
  have hr := e_508_15
  simp only [e_508_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 508 71) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2166)) := by
  have c32433 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32434 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32435 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32436 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32437 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32438 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_508_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_16
  simp only [← c32433, ← c32434, ← c32435] at e_508_16
  have e_508_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_17
  simp only [← c32436, ← c32437, k2, ← c32438] at e_508_17
  have hr := e_508_17
  simp only [e_508_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 508 79) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2167)) := by
  have c32439 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32440 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32441 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32442 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32443 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32444 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_508_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_18
  simp only [← c32439, ← c32440, ← c32441] at e_508_18
  have e_508_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row508 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_508_19
  simp only [← c32442, ← c32443, k2, ← c32444] at e_508_19
  have hr := e_508_19
  simp only [e_508_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 509 7) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2168)) := by
  have c32445 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32446 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32447 := (publicBatchWrapper86_copies1013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32448 := (publicBatchWrapper86_copies1014 a h).1
  have c32449 := (publicBatchWrapper86_copies1014 a h).2.1
  have c32450 := (publicBatchWrapper86_copies1014 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_509_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_0
  simp only [← c32445, ← c32446, ← c32447] at e_509_0
  have e_509_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_1
  simp only [← c32448, ← c32449, k2, ← c32450] at e_509_1
  have hr := e_509_1
  simp only [e_509_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 509 15) = bselect (a (.wire 11 67)) (a (.virt 200)) (a (.virt 2169)) := by
  have c32451 := (publicBatchWrapper86_copies1014 a h).2.2.2.1
  have c32452 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.1
  have c32453 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.1
  have c32454 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.1
  have c32455 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.1
  have c32456 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_509_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_2
  simp only [← c32451, ← c32452, ← c32453] at e_509_2
  have e_509_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_3
  simp only [← c32454, ← c32455, k2, ← c32456] at e_509_3
  have hr := e_509_3
  simp only [e_509_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 509 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2210)) := by
  have c32457 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.1
  have c32458 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32459 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32460 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32461 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32462 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_509_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_4
  simp only [← c32457, ← c32458, ← c32459] at e_509_4
  have e_509_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_5
  simp only [← c32460, ← c32461, k2, ← c32462] at e_509_5
  have hr := e_509_5
  simp only [e_509_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 509 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2211)) := by
  have c32463 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32464 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32465 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32466 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32467 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32468 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_509_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_6
  simp only [← c32463, ← c32464, ← c32465] at e_509_6
  have e_509_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_7
  simp only [← c32466, ← c32467, k2, ← c32468] at e_509_7
  have hr := e_509_7
  simp only [e_509_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 509 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2212)) := by
  have c32469 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32470 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32471 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32472 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32473 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32474 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_509_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_8
  simp only [← c32469, ← c32470, ← c32471] at e_509_8
  have e_509_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_9
  simp only [← c32472, ← c32473, k2, ← c32474] at e_509_9
  have hr := e_509_9
  simp only [e_509_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 509 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2213)) := by
  have c32475 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32476 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32477 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32478 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32479 := (publicBatchWrapper86_copies1014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32480 := (publicBatchWrapper86_copies1015 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_509_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_10
  simp only [← c32475, ← c32476, ← c32477] at e_509_10
  have e_509_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_11
  simp only [← c32478, ← c32479, k2, ← c32480] at e_509_11
  have hr := e_509_11
  simp only [e_509_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 509 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2214)) := by
  have c32481 := (publicBatchWrapper86_copies1015 a h).2.1
  have c32482 := (publicBatchWrapper86_copies1015 a h).2.2.1
  have c32483 := (publicBatchWrapper86_copies1015 a h).2.2.2.1
  have c32484 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.1
  have c32485 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.1
  have c32486 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_509_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_12
  simp only [← c32481, ← c32482, ← c32483] at e_509_12
  have e_509_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_13
  simp only [← c32484, ← c32485, k2, ← c32486] at e_509_13
  have hr := e_509_13
  simp only [e_509_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 509 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2215)) := by
  have c32487 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.1
  have c32488 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.1
  have c32489 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.1
  have c32490 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32491 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32492 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_509_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_14
  simp only [← c32487, ← c32488, ← c32489] at e_509_14
  have e_509_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_15
  simp only [← c32490, ← c32491, k2, ← c32492] at e_509_15
  have hr := e_509_15
  simp only [e_509_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 509 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2216)) := by
  have c32493 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32494 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32495 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32496 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32497 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32498 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_509_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_16
  simp only [← c32493, ← c32494, ← c32495] at e_509_16
  have e_509_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_17
  simp only [← c32496, ← c32497, k2, ← c32498] at e_509_17
  have hr := e_509_17
  simp only [e_509_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 509 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2217)) := by
  have c32499 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32500 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32501 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32502 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32503 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32504 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_509_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_18
  simp only [← c32499, ← c32500, ← c32501] at e_509_18
  have e_509_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row509 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_509_19
  simp only [← c32502, ← c32503, k2, ← c32504] at e_509_19
  have hr := e_509_19
  simp only [e_509_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 510 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2218)) := by
  have c32505 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32506 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32507 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32508 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32509 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32510 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_510_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_0
  simp only [← c32505, ← c32506, ← c32507] at e_510_0
  have e_510_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_1
  simp only [← c32508, ← c32509, k2, ← c32510] at e_510_1
  have hr := e_510_1
  simp only [e_510_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 510 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2219)) := by
  have c32511 := (publicBatchWrapper86_copies1015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32512 := (publicBatchWrapper86_copies1016 a h).1
  have c32513 := (publicBatchWrapper86_copies1016 a h).2.1
  have c32514 := (publicBatchWrapper86_copies1016 a h).2.2.1
  have c32515 := (publicBatchWrapper86_copies1016 a h).2.2.2.1
  have c32516 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_510_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_2
  simp only [← c32511, ← c32512, ← c32513] at e_510_2
  have e_510_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_3
  simp only [← c32514, ← c32515, k2, ← c32516] at e_510_3
  have hr := e_510_3
  simp only [e_510_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 510 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2220)) := by
  have c32517 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.1
  have c32518 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.1
  have c32519 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.1
  have c32520 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.1
  have c32521 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.1
  have c32522 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_510_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_4
  simp only [← c32517, ← c32518, ← c32519] at e_510_4
  have e_510_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_5
  simp only [← c32520, ← c32521, k2, ← c32522] at e_510_5
  have hr := e_510_5
  simp only [e_510_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 510 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2221)) := by
  have c32523 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32524 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32525 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32526 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32527 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32528 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_510_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_6
  simp only [← c32523, ← c32524, ← c32525] at e_510_6
  have e_510_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_7
  simp only [← c32526, ← c32527, k2, ← c32528] at e_510_7
  have hr := e_510_7
  simp only [e_510_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 510 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2222)) := by
  have c32529 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32530 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32531 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32532 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32533 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32534 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_510_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_8
  simp only [← c32529, ← c32530, ← c32531] at e_510_8
  have e_510_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_9
  simp only [← c32532, ← c32533, k2, ← c32534] at e_510_9
  have hr := e_510_9
  simp only [e_510_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 510 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2223)) := by
  have c32535 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32536 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32537 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32538 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32539 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32540 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_510_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_10
  simp only [← c32535, ← c32536, ← c32537] at e_510_10
  have e_510_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_11
  simp only [← c32538, ← c32539, k2, ← c32540] at e_510_11
  have hr := e_510_11
  simp only [e_510_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 510 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2224)) := by
  have c32541 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32542 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32543 := (publicBatchWrapper86_copies1016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32544 := (publicBatchWrapper86_copies1017 a h).1
  have c32545 := (publicBatchWrapper86_copies1017 a h).2.1
  have c32546 := (publicBatchWrapper86_copies1017 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_510_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_12
  simp only [← c32541, ← c32542, ← c32543] at e_510_12
  have e_510_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_13
  simp only [← c32544, ← c32545, k2, ← c32546] at e_510_13
  have hr := e_510_13
  simp only [e_510_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 510 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2225)) := by
  have c32547 := (publicBatchWrapper86_copies1017 a h).2.2.2.1
  have c32548 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.1
  have c32549 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.1
  have c32550 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.1
  have c32551 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.1
  have c32552 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_510_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_14
  simp only [← c32547, ← c32548, ← c32549] at e_510_14
  have e_510_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_15
  simp only [← c32550, ← c32551, k2, ← c32552] at e_510_15
  have hr := e_510_15
  simp only [e_510_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 510 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2226)) := by
  have c32553 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.1
  have c32554 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32555 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32556 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32557 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32558 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_510_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_16
  simp only [← c32553, ← c32554, ← c32555] at e_510_16
  have e_510_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_17
  simp only [← c32556, ← c32557, k2, ← c32558] at e_510_17
  have hr := e_510_17
  simp only [e_510_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 510 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2227)) := by
  have c32559 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32560 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32561 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32562 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32563 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32564 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_510_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_18
  simp only [← c32559, ← c32560, ← c32561] at e_510_18
  have e_510_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row510 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_510_19
  simp only [← c32562, ← c32563, k2, ← c32564] at e_510_19
  have hr := e_510_19
  simp only [e_510_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 511 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2228)) := by
  have c32565 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32566 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32567 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32568 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32569 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32570 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_511_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_0
  simp only [← c32565, ← c32566, ← c32567] at e_511_0
  have e_511_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_1
  simp only [← c32568, ← c32569, k2, ← c32570] at e_511_1
  have hr := e_511_1
  simp only [e_511_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 511 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2229)) := by
  have c32571 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32572 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32573 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32574 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32575 := (publicBatchWrapper86_copies1017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32576 := (publicBatchWrapper86_copies1018 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_511_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_2
  simp only [← c32571, ← c32572, ← c32573] at e_511_2
  have e_511_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_3
  simp only [← c32574, ← c32575, k2, ← c32576] at e_511_3
  have hr := e_511_3
  simp only [e_511_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 511 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2230)) := by
  have c32577 := (publicBatchWrapper86_copies1018 a h).2.1
  have c32578 := (publicBatchWrapper86_copies1018 a h).2.2.1
  have c32579 := (publicBatchWrapper86_copies1018 a h).2.2.2.1
  have c32580 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.1
  have c32581 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.1
  have c32582 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_511_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_4
  simp only [← c32577, ← c32578, ← c32579] at e_511_4
  have e_511_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_5
  simp only [← c32580, ← c32581, k2, ← c32582] at e_511_5
  have hr := e_511_5
  simp only [e_511_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 511 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2231)) := by
  have c32583 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.1
  have c32584 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.1
  have c32585 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.1
  have c32586 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32587 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32588 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_511_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_6
  simp only [← c32583, ← c32584, ← c32585] at e_511_6
  have e_511_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_7
  simp only [← c32586, ← c32587, k2, ← c32588] at e_511_7
  have hr := e_511_7
  simp only [e_511_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 511 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2232)) := by
  have c32589 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32590 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32591 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32592 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32593 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32594 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_511_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_8
  simp only [← c32589, ← c32590, ← c32591] at e_511_8
  have e_511_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_9
  simp only [← c32592, ← c32593, k2, ← c32594] at e_511_9
  have hr := e_511_9
  simp only [e_511_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 511 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2233)) := by
  have c32595 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32596 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32597 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32598 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32599 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32600 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_511_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_10
  simp only [← c32595, ← c32596, ← c32597] at e_511_10
  have e_511_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_11
  simp only [← c32598, ← c32599, k2, ← c32600] at e_511_11
  have hr := e_511_11
  simp only [e_511_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 511 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2234)) := by
  have c32601 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32602 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32603 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32604 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32605 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32606 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_511_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_12
  simp only [← c32601, ← c32602, ← c32603] at e_511_12
  have e_511_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_13
  simp only [← c32604, ← c32605, k2, ← c32606] at e_511_13
  have hr := e_511_13
  simp only [e_511_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 511 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2235)) := by
  have c32607 := (publicBatchWrapper86_copies1018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32608 := (publicBatchWrapper86_copies1019 a h).1
  have c32609 := (publicBatchWrapper86_copies1019 a h).2.1
  have c32610 := (publicBatchWrapper86_copies1019 a h).2.2.1
  have c32611 := (publicBatchWrapper86_copies1019 a h).2.2.2.1
  have c32612 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_511_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_14
  simp only [← c32607, ← c32608, ← c32609] at e_511_14
  have e_511_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_15
  simp only [← c32610, ← c32611, k2, ← c32612] at e_511_15
  have hr := e_511_15
  simp only [e_511_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 511 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2236)) := by
  have c32613 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.1
  have c32614 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.1
  have c32615 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.1
  have c32616 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.1
  have c32617 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.1
  have c32618 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_511_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_16
  simp only [← c32613, ← c32614, ← c32615] at e_511_16
  have e_511_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_17
  simp only [← c32616, ← c32617, k2, ← c32618] at e_511_17
  have hr := e_511_17
  simp only [e_511_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 511 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2237)) := by
  have c32619 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32620 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32621 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32622 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32623 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32624 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_511_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_18
  simp only [← c32619, ← c32620, ← c32621] at e_511_18
  have e_511_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row511 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_511_19
  simp only [← c32622, ← c32623, k2, ← c32624] at e_511_19
  have hr := e_511_19
  simp only [e_511_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 512 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2238)) := by
  have c32625 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32626 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32627 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32628 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32629 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32630 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_512_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_0
  simp only [← c32625, ← c32626, ← c32627] at e_512_0
  have e_512_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_1
  simp only [← c32628, ← c32629, k2, ← c32630] at e_512_1
  have hr := e_512_1
  simp only [e_512_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 512 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2239)) := by
  have c32631 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32632 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32633 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32634 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32635 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32636 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_512_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_2
  simp only [← c32631, ← c32632, ← c32633] at e_512_2
  have e_512_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_3
  simp only [← c32634, ← c32635, k2, ← c32636] at e_512_3
  have hr := e_512_3
  simp only [e_512_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 512 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2240)) := by
  have c32637 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32638 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32639 := (publicBatchWrapper86_copies1019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32640 := (publicBatchWrapper86_copies1020 a h).1
  have c32641 := (publicBatchWrapper86_copies1020 a h).2.1
  have c32642 := (publicBatchWrapper86_copies1020 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_512_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_4
  simp only [← c32637, ← c32638, ← c32639] at e_512_4
  have e_512_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_5
  simp only [← c32640, ← c32641, k2, ← c32642] at e_512_5
  have hr := e_512_5
  simp only [e_512_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 512 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2241)) := by
  have c32643 := (publicBatchWrapper86_copies1020 a h).2.2.2.1
  have c32644 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.1
  have c32645 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.1
  have c32646 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.1
  have c32647 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.1
  have c32648 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_512_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_6
  simp only [← c32643, ← c32644, ← c32645] at e_512_6
  have e_512_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_7
  simp only [← c32646, ← c32647, k2, ← c32648] at e_512_7
  have hr := e_512_7
  simp only [e_512_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 512 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2242)) := by
  have c32649 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.1
  have c32650 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32651 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32652 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32653 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32654 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_512_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_8
  simp only [← c32649, ← c32650, ← c32651] at e_512_8
  have e_512_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_9
  simp only [← c32652, ← c32653, k2, ← c32654] at e_512_9
  have hr := e_512_9
  simp only [e_512_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 512 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2243)) := by
  have c32655 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32656 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32657 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32658 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32659 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32660 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_512_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_10
  simp only [← c32655, ← c32656, ← c32657] at e_512_10
  have e_512_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_11
  simp only [← c32658, ← c32659, k2, ← c32660] at e_512_11
  have hr := e_512_11
  simp only [e_512_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 512 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2244)) := by
  have c32661 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32662 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32663 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32664 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32665 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32666 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_512_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_12
  simp only [← c32661, ← c32662, ← c32663] at e_512_12
  have e_512_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_13
  simp only [← c32664, ← c32665, k2, ← c32666] at e_512_13
  have hr := e_512_13
  simp only [e_512_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 512 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2245)) := by
  have c32667 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32668 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32669 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32670 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32671 := (publicBatchWrapper86_copies1020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32672 := (publicBatchWrapper86_copies1021 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_512_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_14
  simp only [← c32667, ← c32668, ← c32669] at e_512_14
  have e_512_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_15
  simp only [← c32670, ← c32671, k2, ← c32672] at e_512_15
  have hr := e_512_15
  simp only [e_512_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 512 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2246)) := by
  have c32673 := (publicBatchWrapper86_copies1021 a h).2.1
  have c32674 := (publicBatchWrapper86_copies1021 a h).2.2.1
  have c32675 := (publicBatchWrapper86_copies1021 a h).2.2.2.1
  have c32676 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.1
  have c32677 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.1
  have c32678 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_512_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_16
  simp only [← c32673, ← c32674, ← c32675] at e_512_16
  have e_512_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_17
  simp only [← c32676, ← c32677, k2, ← c32678] at e_512_17
  have hr := e_512_17
  simp only [e_512_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 512 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2247)) := by
  have c32679 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.1
  have c32680 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.1
  have c32681 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.1
  have c32682 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32683 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32684 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_512_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_18
  simp only [← c32679, ← c32680, ← c32681] at e_512_18
  have e_512_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row512 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_512_19
  simp only [← c32682, ← c32683, k2, ← c32684] at e_512_19
  have hr := e_512_19
  simp only [e_512_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 513 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2248)) := by
  have c32685 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32686 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32687 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32688 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32689 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32690 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_513_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_0
  simp only [← c32685, ← c32686, ← c32687] at e_513_0
  have e_513_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_1
  simp only [← c32688, ← c32689, k2, ← c32690] at e_513_1
  have hr := e_513_1
  simp only [e_513_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 513 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2249)) := by
  have c32691 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32692 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32693 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32694 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32695 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32696 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_513_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_2
  simp only [← c32691, ← c32692, ← c32693] at e_513_2
  have e_513_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_3
  simp only [← c32694, ← c32695, k2, ← c32696] at e_513_3
  have hr := e_513_3
  simp only [e_513_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 513 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2250)) := by
  have c32697 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32698 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32699 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32700 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32701 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32702 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_513_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_4
  simp only [← c32697, ← c32698, ← c32699] at e_513_4
  have e_513_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_5
  simp only [← c32700, ← c32701, k2, ← c32702] at e_513_5
  have hr := e_513_5
  simp only [e_513_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 513 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2251)) := by
  have c32703 := (publicBatchWrapper86_copies1021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32704 := (publicBatchWrapper86_copies1022 a h).1
  have c32705 := (publicBatchWrapper86_copies1022 a h).2.1
  have c32706 := (publicBatchWrapper86_copies1022 a h).2.2.1
  have c32707 := (publicBatchWrapper86_copies1022 a h).2.2.2.1
  have c32708 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_513_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_6
  simp only [← c32703, ← c32704, ← c32705] at e_513_6
  have e_513_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_7
  simp only [← c32706, ← c32707, k2, ← c32708] at e_513_7
  have hr := e_513_7
  simp only [e_513_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 513 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2252)) := by
  have c32709 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.1
  have c32710 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.1
  have c32711 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.1
  have c32712 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.1
  have c32713 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.1
  have c32714 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_513_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_8
  simp only [← c32709, ← c32710, ← c32711] at e_513_8
  have e_513_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_9
  simp only [← c32712, ← c32713, k2, ← c32714] at e_513_9
  have hr := e_513_9
  simp only [e_513_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 513 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2253)) := by
  have c32715 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32716 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32717 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32718 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32719 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32720 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_513_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_10
  simp only [← c32715, ← c32716, ← c32717] at e_513_10
  have e_513_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_11
  simp only [← c32718, ← c32719, k2, ← c32720] at e_513_11
  have hr := e_513_11
  simp only [e_513_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 513 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2254)) := by
  have c32721 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32722 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32723 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32724 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32725 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32726 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_513_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_12
  simp only [← c32721, ← c32722, ← c32723] at e_513_12
  have e_513_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_13
  simp only [← c32724, ← c32725, k2, ← c32726] at e_513_13
  have hr := e_513_13
  simp only [e_513_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 513 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2255)) := by
  have c32727 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32728 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32729 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32730 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32731 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32732 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_513_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_14
  simp only [← c32727, ← c32728, ← c32729] at e_513_14
  have e_513_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_15
  simp only [← c32730, ← c32731, k2, ← c32732] at e_513_15
  have hr := e_513_15
  simp only [e_513_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 513 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2256)) := by
  have c32733 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32734 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32735 := (publicBatchWrapper86_copies1022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32736 := (publicBatchWrapper86_copies1023 a h).1
  have c32737 := (publicBatchWrapper86_copies1023 a h).2.1
  have c32738 := (publicBatchWrapper86_copies1023 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_513_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_16
  simp only [← c32733, ← c32734, ← c32735] at e_513_16
  have e_513_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_17
  simp only [← c32736, ← c32737, k2, ← c32738] at e_513_17
  have hr := e_513_17
  simp only [e_513_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 513 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2257)) := by
  have c32739 := (publicBatchWrapper86_copies1023 a h).2.2.2.1
  have c32740 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.1
  have c32741 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.1
  have c32742 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.1
  have c32743 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.1
  have c32744 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_513_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_18
  simp only [← c32739, ← c32740, ← c32741] at e_513_18
  have e_513_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row513 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_513_19
  simp only [← c32742, ← c32743, k2, ← c32744] at e_513_19
  have hr := e_513_19
  simp only [e_513_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 514 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2258)) := by
  have c32745 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.1
  have c32746 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32747 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32748 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32749 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32750 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_514_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_0
  simp only [← c32745, ← c32746, ← c32747] at e_514_0
  have e_514_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_1
  simp only [← c32748, ← c32749, k2, ← c32750] at e_514_1
  have hr := e_514_1
  simp only [e_514_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 514 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2259)) := by
  have c32751 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32752 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32753 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32754 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32755 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32756 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_514_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_2
  simp only [← c32751, ← c32752, ← c32753] at e_514_2
  have e_514_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_3
  simp only [← c32754, ← c32755, k2, ← c32756] at e_514_3
  have hr := e_514_3
  simp only [e_514_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 514 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2260)) := by
  have c32757 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32758 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32759 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32760 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32761 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32762 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_514_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_4
  simp only [← c32757, ← c32758, ← c32759] at e_514_4
  have e_514_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_5
  simp only [← c32760, ← c32761, k2, ← c32762] at e_514_5
  have hr := e_514_5
  simp only [e_514_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 514 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2261)) := by
  have c32763 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32764 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32765 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32766 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32767 := (publicBatchWrapper86_copies1023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32768 := (publicBatchWrapper86_copies1024 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_514_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_6
  simp only [← c32763, ← c32764, ← c32765] at e_514_6
  have e_514_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_7
  simp only [← c32766, ← c32767, k2, ← c32768] at e_514_7
  have hr := e_514_7
  simp only [e_514_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 514 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2262)) := by
  have c32769 := (publicBatchWrapper86_copies1024 a h).2.1
  have c32770 := (publicBatchWrapper86_copies1024 a h).2.2.1
  have c32771 := (publicBatchWrapper86_copies1024 a h).2.2.2.1
  have c32772 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.1
  have c32773 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.1
  have c32774 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_514_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_8
  simp only [← c32769, ← c32770, ← c32771] at e_514_8
  have e_514_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_9
  simp only [← c32772, ← c32773, k2, ← c32774] at e_514_9
  have hr := e_514_9
  simp only [e_514_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 514 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2263)) := by
  have c32775 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.1
  have c32776 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.1
  have c32777 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.1
  have c32778 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32779 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32780 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_514_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_10
  simp only [← c32775, ← c32776, ← c32777] at e_514_10
  have e_514_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_11
  simp only [← c32778, ← c32779, k2, ← c32780] at e_514_11
  have hr := e_514_11
  simp only [e_514_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 514 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2264)) := by
  have c32781 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32782 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32783 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32784 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32785 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32786 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_514_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_12
  simp only [← c32781, ← c32782, ← c32783] at e_514_12
  have e_514_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_13
  simp only [← c32784, ← c32785, k2, ← c32786] at e_514_13
  have hr := e_514_13
  simp only [e_514_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 514 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2265)) := by
  have c32787 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32788 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32789 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32790 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32791 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32792 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_514_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_14
  simp only [← c32787, ← c32788, ← c32789] at e_514_14
  have e_514_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_15
  simp only [← c32790, ← c32791, k2, ← c32792] at e_514_15
  have hr := e_514_15
  simp only [e_514_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 514 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2266)) := by
  have c32793 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32794 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32795 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32796 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32797 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32798 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_514_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_16
  simp only [← c32793, ← c32794, ← c32795] at e_514_16
  have e_514_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_17
  simp only [← c32796, ← c32797, k2, ← c32798] at e_514_17
  have hr := e_514_17
  simp only [e_514_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 514 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2267)) := by
  have c32799 := (publicBatchWrapper86_copies1024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32800 := (publicBatchWrapper86_copies1025 a h).1
  have c32801 := (publicBatchWrapper86_copies1025 a h).2.1
  have c32802 := (publicBatchWrapper86_copies1025 a h).2.2.1
  have c32803 := (publicBatchWrapper86_copies1025 a h).2.2.2.1
  have c32804 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_514_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_18
  simp only [← c32799, ← c32800, ← c32801] at e_514_18
  have e_514_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row514 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_514_19
  simp only [← c32802, ← c32803, k2, ← c32804] at e_514_19
  have hr := e_514_19
  simp only [e_514_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 515 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2268)) := by
  have c32805 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.1
  have c32806 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.1
  have c32807 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.1
  have c32808 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.1
  have c32809 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.1
  have c32810 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_515_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_0
  simp only [← c32805, ← c32806, ← c32807] at e_515_0
  have e_515_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_1
  simp only [← c32808, ← c32809, k2, ← c32810] at e_515_1
  have hr := e_515_1
  simp only [e_515_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 515 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2269)) := by
  have c32811 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32812 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32813 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32814 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32815 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32816 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_515_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_2
  simp only [← c32811, ← c32812, ← c32813] at e_515_2
  have e_515_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_3
  simp only [← c32814, ← c32815, k2, ← c32816] at e_515_3
  have hr := e_515_3
  simp only [e_515_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 515 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2270)) := by
  have c32817 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32818 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32819 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32820 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32821 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32822 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_515_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_4
  simp only [← c32817, ← c32818, ← c32819] at e_515_4
  have e_515_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_5
  simp only [← c32820, ← c32821, k2, ← c32822] at e_515_5
  have hr := e_515_5
  simp only [e_515_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 515 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2271)) := by
  have c32823 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32824 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32825 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32826 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32827 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32828 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_515_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_6
  simp only [← c32823, ← c32824, ← c32825] at e_515_6
  have e_515_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_7
  simp only [← c32826, ← c32827, k2, ← c32828] at e_515_7
  have hr := e_515_7
  simp only [e_515_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 515 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2272)) := by
  have c32829 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32830 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32831 := (publicBatchWrapper86_copies1025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32832 := (publicBatchWrapper86_copies1026 a h).1
  have c32833 := (publicBatchWrapper86_copies1026 a h).2.1
  have c32834 := (publicBatchWrapper86_copies1026 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_515_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_8
  simp only [← c32829, ← c32830, ← c32831] at e_515_8
  have e_515_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_9
  simp only [← c32832, ← c32833, k2, ← c32834] at e_515_9
  have hr := e_515_9
  simp only [e_515_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 515 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2273)) := by
  have c32835 := (publicBatchWrapper86_copies1026 a h).2.2.2.1
  have c32836 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.1
  have c32837 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.1
  have c32838 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.1
  have c32839 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.1
  have c32840 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_515_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_10
  simp only [← c32835, ← c32836, ← c32837] at e_515_10
  have e_515_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_11
  simp only [← c32838, ← c32839, k2, ← c32840] at e_515_11
  have hr := e_515_11
  simp only [e_515_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 515 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2274)) := by
  have c32841 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.1
  have c32842 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32843 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32844 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32845 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32846 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_515_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_12
  simp only [← c32841, ← c32842, ← c32843] at e_515_12
  have e_515_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_13
  simp only [← c32844, ← c32845, k2, ← c32846] at e_515_13
  have hr := e_515_13
  simp only [e_515_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 515 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2275)) := by
  have c32847 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32848 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32849 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32850 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32851 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32852 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_515_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_14
  simp only [← c32847, ← c32848, ← c32849] at e_515_14
  have e_515_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_15
  simp only [← c32850, ← c32851, k2, ← c32852] at e_515_15
  have hr := e_515_15
  simp only [e_515_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 515 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2276)) := by
  have c32853 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32854 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32855 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32856 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32857 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32858 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_515_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_16
  simp only [← c32853, ← c32854, ← c32855] at e_515_16
  have e_515_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_17
  simp only [← c32856, ← c32857, k2, ← c32858] at e_515_17
  have hr := e_515_17
  simp only [e_515_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 515 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2277)) := by
  have c32859 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32860 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32861 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32862 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32863 := (publicBatchWrapper86_copies1026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32864 := (publicBatchWrapper86_copies1027 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_515_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_18
  simp only [← c32859, ← c32860, ← c32861] at e_515_18
  have e_515_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row515 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_515_19
  simp only [← c32862, ← c32863, k2, ← c32864] at e_515_19
  have hr := e_515_19
  simp only [e_515_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 516 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2278)) := by
  have c32865 := (publicBatchWrapper86_copies1027 a h).2.1
  have c32866 := (publicBatchWrapper86_copies1027 a h).2.2.1
  have c32867 := (publicBatchWrapper86_copies1027 a h).2.2.2.1
  have c32868 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.1
  have c32869 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.1
  have c32870 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_516_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_0
  simp only [← c32865, ← c32866, ← c32867] at e_516_0
  have e_516_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_1
  simp only [← c32868, ← c32869, k2, ← c32870] at e_516_1
  have hr := e_516_1
  simp only [e_516_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 516 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2279)) := by
  have c32871 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.1
  have c32872 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.1
  have c32873 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.1
  have c32874 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32875 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32876 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_516_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_2
  simp only [← c32871, ← c32872, ← c32873] at e_516_2
  have e_516_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_3
  simp only [← c32874, ← c32875, k2, ← c32876] at e_516_3
  have hr := e_516_3
  simp only [e_516_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 516 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2280)) := by
  have c32877 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32878 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32879 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32880 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32881 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32882 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_516_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_4
  simp only [← c32877, ← c32878, ← c32879] at e_516_4
  have e_516_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_5
  simp only [← c32880, ← c32881, k2, ← c32882] at e_516_5
  have hr := e_516_5
  simp only [e_516_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 516 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2281)) := by
  have c32883 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32884 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32885 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32886 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32887 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32888 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_516_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_6
  simp only [← c32883, ← c32884, ← c32885] at e_516_6
  have e_516_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_7
  simp only [← c32886, ← c32887, k2, ← c32888] at e_516_7
  have hr := e_516_7
  simp only [e_516_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 516 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2282)) := by
  have c32889 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32890 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32891 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32892 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32893 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32894 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_516_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_8
  simp only [← c32889, ← c32890, ← c32891] at e_516_8
  have e_516_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_9
  simp only [← c32892, ← c32893, k2, ← c32894] at e_516_9
  have hr := e_516_9
  simp only [e_516_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 516 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2283)) := by
  have c32895 := (publicBatchWrapper86_copies1027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32896 := (publicBatchWrapper86_copies1028 a h).1
  have c32897 := (publicBatchWrapper86_copies1028 a h).2.1
  have c32898 := (publicBatchWrapper86_copies1028 a h).2.2.1
  have c32899 := (publicBatchWrapper86_copies1028 a h).2.2.2.1
  have c32900 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_516_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_10
  simp only [← c32895, ← c32896, ← c32897] at e_516_10
  have e_516_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_11
  simp only [← c32898, ← c32899, k2, ← c32900] at e_516_11
  have hr := e_516_11
  simp only [e_516_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 516 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2284)) := by
  have c32901 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.1
  have c32902 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.1
  have c32903 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.1
  have c32904 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.1
  have c32905 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.1
  have c32906 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_516_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_12
  simp only [← c32901, ← c32902, ← c32903] at e_516_12
  have e_516_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_13
  simp only [← c32904, ← c32905, k2, ← c32906] at e_516_13
  have hr := e_516_13
  simp only [e_516_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 516 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2285)) := by
  have c32907 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32908 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32909 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32910 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32911 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32912 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_516_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_14
  simp only [← c32907, ← c32908, ← c32909] at e_516_14
  have e_516_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_15
  simp only [← c32910, ← c32911, k2, ← c32912] at e_516_15
  have hr := e_516_15
  simp only [e_516_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 516 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2286)) := by
  have c32913 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32914 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32915 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32916 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32917 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32918 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_516_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_16
  simp only [← c32913, ← c32914, ← c32915] at e_516_16
  have e_516_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_17
  simp only [← c32916, ← c32917, k2, ← c32918] at e_516_17
  have hr := e_516_17
  simp only [e_516_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 516 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2287)) := by
  have c32919 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32920 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32921 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32922 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32923 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32924 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_516_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_18
  simp only [← c32919, ← c32920, ← c32921] at e_516_18
  have e_516_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row516 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_516_19
  simp only [← c32922, ← c32923, k2, ← c32924] at e_516_19
  have hr := e_516_19
  simp only [e_516_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 517 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2288)) := by
  have c32925 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32926 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32927 := (publicBatchWrapper86_copies1028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32928 := (publicBatchWrapper86_copies1029 a h).1
  have c32929 := (publicBatchWrapper86_copies1029 a h).2.1
  have c32930 := (publicBatchWrapper86_copies1029 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_517_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_0
  simp only [← c32925, ← c32926, ← c32927] at e_517_0
  have e_517_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_1
  simp only [← c32928, ← c32929, k2, ← c32930] at e_517_1
  have hr := e_517_1
  simp only [e_517_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 517 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2289)) := by
  have c32931 := (publicBatchWrapper86_copies1029 a h).2.2.2.1
  have c32932 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.1
  have c32933 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.1
  have c32934 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.1
  have c32935 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.1
  have c32936 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_517_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_2
  simp only [← c32931, ← c32932, ← c32933] at e_517_2
  have e_517_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_3
  simp only [← c32934, ← c32935, k2, ← c32936] at e_517_3
  have hr := e_517_3
  simp only [e_517_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 517 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2290)) := by
  have c32937 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.1
  have c32938 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32939 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32940 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32941 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32942 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_517_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_4
  simp only [← c32937, ← c32938, ← c32939] at e_517_4
  have e_517_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_5
  simp only [← c32940, ← c32941, k2, ← c32942] at e_517_5
  have hr := e_517_5
  simp only [e_517_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 517 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2291)) := by
  have c32943 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32944 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32945 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32946 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32947 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32948 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_517_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_6
  simp only [← c32943, ← c32944, ← c32945] at e_517_6
  have e_517_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_7
  simp only [← c32946, ← c32947, k2, ← c32948] at e_517_7
  have hr := e_517_7
  simp only [e_517_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 517 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2292)) := by
  have c32949 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32950 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32951 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32952 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32953 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32954 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_517_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_8
  simp only [← c32949, ← c32950, ← c32951] at e_517_8
  have e_517_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_9
  simp only [← c32952, ← c32953, k2, ← c32954] at e_517_9
  have hr := e_517_9
  simp only [e_517_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 517 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2293)) := by
  have c32955 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32956 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32957 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32958 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32959 := (publicBatchWrapper86_copies1029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32960 := (publicBatchWrapper86_copies1030 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_517_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_10
  simp only [← c32955, ← c32956, ← c32957] at e_517_10
  have e_517_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_11
  simp only [← c32958, ← c32959, k2, ← c32960] at e_517_11
  have hr := e_517_11
  simp only [e_517_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 517 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2294)) := by
  have c32961 := (publicBatchWrapper86_copies1030 a h).2.1
  have c32962 := (publicBatchWrapper86_copies1030 a h).2.2.1
  have c32963 := (publicBatchWrapper86_copies1030 a h).2.2.2.1
  have c32964 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.1
  have c32965 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.1
  have c32966 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_517_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_12
  simp only [← c32961, ← c32962, ← c32963] at e_517_12
  have e_517_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_13
  simp only [← c32964, ← c32965, k2, ← c32966] at e_517_13
  have hr := e_517_13
  simp only [e_517_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 517 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2295)) := by
  have c32967 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.1
  have c32968 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.1
  have c32969 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.1
  have c32970 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.1
  have c32971 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c32972 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_517_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_14
  simp only [← c32967, ← c32968, ← c32969] at e_517_14
  have e_517_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_15
  simp only [← c32970, ← c32971, k2, ← c32972] at e_517_15
  have hr := e_517_15
  simp only [e_517_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 517 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2296)) := by
  have c32973 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32974 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32975 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32976 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32977 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32978 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_517_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_16
  simp only [← c32973, ← c32974, ← c32975] at e_517_16
  have e_517_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_17
  simp only [← c32976, ← c32977, k2, ← c32978] at e_517_17
  have hr := e_517_17
  simp only [e_517_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 517 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2297)) := by
  have c32979 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32980 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32981 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32982 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32983 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32984 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_517_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_18
  simp only [← c32979, ← c32980, ← c32981] at e_517_18
  have e_517_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row517 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_517_19
  simp only [← c32982, ← c32983, k2, ← c32984] at e_517_19
  have hr := e_517_19
  simp only [e_517_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 518 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2298)) := by
  have c32985 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32986 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32987 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32988 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32989 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c32990 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_518_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_0
  simp only [← c32985, ← c32986, ← c32987] at e_518_0
  have e_518_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_1
  simp only [← c32988, ← c32989, k2, ← c32990] at e_518_1
  have hr := e_518_1
  simp only [e_518_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 518 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2299)) := by
  have c32991 := (publicBatchWrapper86_copies1030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c32992 := (publicBatchWrapper86_copies1031 a h).1
  have c32993 := (publicBatchWrapper86_copies1031 a h).2.1
  have c32994 := (publicBatchWrapper86_copies1031 a h).2.2.1
  have c32995 := (publicBatchWrapper86_copies1031 a h).2.2.2.1
  have c32996 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_518_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_2
  simp only [← c32991, ← c32992, ← c32993] at e_518_2
  have e_518_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_3
  simp only [← c32994, ← c32995, k2, ← c32996] at e_518_3
  have hr := e_518_3
  simp only [e_518_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 518 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2300)) := by
  have c32997 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.1
  have c32998 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.1
  have c32999 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.1
  have c33000 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.1
  have c33001 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.1
  have c33002 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_518_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_4
  simp only [← c32997, ← c32998, ← c32999] at e_518_4
  have e_518_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_5
  simp only [← c33000, ← c33001, k2, ← c33002] at e_518_5
  have hr := e_518_5
  simp only [e_518_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 518 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2301)) := by
  have c33003 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33004 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33005 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33006 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33007 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33008 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_518_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_6
  simp only [← c33003, ← c33004, ← c33005] at e_518_6
  have e_518_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_7
  simp only [← c33006, ← c33007, k2, ← c33008] at e_518_7
  have hr := e_518_7
  simp only [e_518_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 518 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2302)) := by
  have c33009 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33010 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33011 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33012 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33013 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33014 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_518_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_8
  simp only [← c33009, ← c33010, ← c33011] at e_518_8
  have e_518_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_9
  simp only [← c33012, ← c33013, k2, ← c33014] at e_518_9
  have hr := e_518_9
  simp only [e_518_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 518 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2303)) := by
  have c33015 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33016 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33017 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33018 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33019 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33020 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_518_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_10
  simp only [← c33015, ← c33016, ← c33017] at e_518_10
  have e_518_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_11
  simp only [← c33018, ← c33019, k2, ← c33020] at e_518_11
  have hr := e_518_11
  simp only [e_518_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 518 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2304)) := by
  have c33021 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33022 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33023 := (publicBatchWrapper86_copies1031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33024 := (publicBatchWrapper86_copies1032 a h).1
  have c33025 := (publicBatchWrapper86_copies1032 a h).2.1
  have c33026 := (publicBatchWrapper86_copies1032 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_518_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_12
  simp only [← c33021, ← c33022, ← c33023] at e_518_12
  have e_518_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_13
  simp only [← c33024, ← c33025, k2, ← c33026] at e_518_13
  have hr := e_518_13
  simp only [e_518_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 518 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2305)) := by
  have c33027 := (publicBatchWrapper86_copies1032 a h).2.2.2.1
  have c33028 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.1
  have c33029 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.1
  have c33030 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.1
  have c33031 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.1
  have c33032 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_518_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_14
  simp only [← c33027, ← c33028, ← c33029] at e_518_14
  have e_518_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_15
  simp only [← c33030, ← c33031, k2, ← c33032] at e_518_15
  have hr := e_518_15
  simp only [e_518_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 518 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2306)) := by
  have c33033 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.1
  have c33034 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33035 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33036 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33037 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33038 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_518_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_16
  simp only [← c33033, ← c33034, ← c33035] at e_518_16
  have e_518_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_17
  simp only [← c33036, ← c33037, k2, ← c33038] at e_518_17
  have hr := e_518_17
  simp only [e_518_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4781 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 518 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2307)) := by
  have c33039 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33040 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33041 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33042 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33043 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33044 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_518_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_18
  simp only [← c33039, ← c33040, ← c33041] at e_518_18
  have e_518_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row518 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_518_19
  simp only [← c33042, ← c33043, k2, ← c33044] at e_518_19
  have hr := e_518_19
  simp only [e_518_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4782 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 519 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2308)) := by
  have c33045 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33046 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33047 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33048 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33049 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33050 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_519_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_0
  simp only [← c33045, ← c33046, ← c33047] at e_519_0
  have e_519_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_1
  simp only [← c33048, ← c33049, k2, ← c33050] at e_519_1
  have hr := e_519_1
  simp only [e_519_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4783 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 519 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2309)) := by
  have c33051 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33052 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33053 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33054 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33055 := (publicBatchWrapper86_copies1032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33056 := (publicBatchWrapper86_copies1033 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_519_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_2
  simp only [← c33051, ← c33052, ← c33053] at e_519_2
  have e_519_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_3
  simp only [← c33054, ← c33055, k2, ← c33056] at e_519_3
  have hr := e_519_3
  simp only [e_519_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4784 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 519 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2310)) := by
  have c33057 := (publicBatchWrapper86_copies1033 a h).2.1
  have c33058 := (publicBatchWrapper86_copies1033 a h).2.2.1
  have c33059 := (publicBatchWrapper86_copies1033 a h).2.2.2.1
  have c33060 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.1
  have c33061 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.1
  have c33062 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_519_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_4
  simp only [← c33057, ← c33058, ← c33059] at e_519_4
  have e_519_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_5
  simp only [← c33060, ← c33061, k2, ← c33062] at e_519_5
  have hr := e_519_5
  simp only [e_519_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4785 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 519 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2311)) := by
  have c33063 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.1
  have c33064 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.1
  have c33065 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.1
  have c33066 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33067 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33068 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_519_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_6
  simp only [← c33063, ← c33064, ← c33065] at e_519_6
  have e_519_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_7
  simp only [← c33066, ← c33067, k2, ← c33068] at e_519_7
  have hr := e_519_7
  simp only [e_519_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4786 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 519 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2312)) := by
  have c33069 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33070 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33071 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33072 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33073 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33074 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_519_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_8
  simp only [← c33069, ← c33070, ← c33071] at e_519_8
  have e_519_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_9
  simp only [← c33072, ← c33073, k2, ← c33074] at e_519_9
  have hr := e_519_9
  simp only [e_519_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4787 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 519 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2313)) := by
  have c33075 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33076 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33077 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33078 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33079 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33080 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_519_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_10
  simp only [← c33075, ← c33076, ← c33077] at e_519_10
  have e_519_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_11
  simp only [← c33078, ← c33079, k2, ← c33080] at e_519_11
  have hr := e_519_11
  simp only [e_519_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4788 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 519 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2314)) := by
  have c33081 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33082 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33083 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33084 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33085 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33086 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_519_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_12
  simp only [← c33081, ← c33082, ← c33083] at e_519_12
  have e_519_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_13
  simp only [← c33084, ← c33085, k2, ← c33086] at e_519_13
  have hr := e_519_13
  simp only [e_519_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4789 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 519 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2315)) := by
  have c33087 := (publicBatchWrapper86_copies1033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33088 := (publicBatchWrapper86_copies1034 a h).1
  have c33089 := (publicBatchWrapper86_copies1034 a h).2.1
  have c33090 := (publicBatchWrapper86_copies1034 a h).2.2.1
  have c33091 := (publicBatchWrapper86_copies1034 a h).2.2.2.1
  have c33092 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_519_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_14
  simp only [← c33087, ← c33088, ← c33089] at e_519_14
  have e_519_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_15
  simp only [← c33090, ← c33091, k2, ← c33092] at e_519_15
  have hr := e_519_15
  simp only [e_519_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4790 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 519 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2316)) := by
  have c33093 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.1
  have c33094 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.1
  have c33095 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.1
  have c33096 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.1
  have c33097 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.1
  have c33098 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_519_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_16
  simp only [← c33093, ← c33094, ← c33095] at e_519_16
  have e_519_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_17
  simp only [← c33096, ← c33097, k2, ← c33098] at e_519_17
  have hr := e_519_17
  simp only [e_519_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4791 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 519 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2317)) := by
  have c33099 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33100 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33101 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33102 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33103 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33104 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_519_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_18
  simp only [← c33099, ← c33100, ← c33101] at e_519_18
  have e_519_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row519 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_519_19
  simp only [← c33102, ← c33103, k2, ← c33104] at e_519_19
  have hr := e_519_19
  simp only [e_519_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4792 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 520 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2318)) := by
  have c33105 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33106 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33107 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33108 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33109 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33110 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_520_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_0
  simp only [← c33105, ← c33106, ← c33107] at e_520_0
  have e_520_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_1
  simp only [← c33108, ← c33109, k2, ← c33110] at e_520_1
  have hr := e_520_1
  simp only [e_520_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4793 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 520 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2319)) := by
  have c33111 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33112 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33113 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33114 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33115 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33116 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_520_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_2
  simp only [← c33111, ← c33112, ← c33113] at e_520_2
  have e_520_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_3
  simp only [← c33114, ← c33115, k2, ← c33116] at e_520_3
  have hr := e_520_3
  simp only [e_520_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4794 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 520 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2320)) := by
  have c33117 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33118 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33119 := (publicBatchWrapper86_copies1034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33120 := (publicBatchWrapper86_copies1035 a h).1
  have c33121 := (publicBatchWrapper86_copies1035 a h).2.1
  have c33122 := (publicBatchWrapper86_copies1035 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_520_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_4
  simp only [← c33117, ← c33118, ← c33119] at e_520_4
  have e_520_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_5
  simp only [← c33120, ← c33121, k2, ← c33122] at e_520_5
  have hr := e_520_5
  simp only [e_520_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4795 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 520 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2321)) := by
  have c33123 := (publicBatchWrapper86_copies1035 a h).2.2.2.1
  have c33124 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.1
  have c33125 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.1
  have c33126 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.1
  have c33127 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.1
  have c33128 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_520_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_6
  simp only [← c33123, ← c33124, ← c33125] at e_520_6
  have e_520_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_7
  simp only [← c33126, ← c33127, k2, ← c33128] at e_520_7
  have hr := e_520_7
  simp only [e_520_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4796 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 520 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2322)) := by
  have c33129 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.1
  have c33130 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33131 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33132 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33133 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33134 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_520_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_8
  simp only [← c33129, ← c33130, ← c33131] at e_520_8
  have e_520_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_9
  simp only [← c33132, ← c33133, k2, ← c33134] at e_520_9
  have hr := e_520_9
  simp only [e_520_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4797 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 520 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2323)) := by
  have c33135 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33136 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33137 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33138 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33139 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33140 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_520_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_10
  simp only [← c33135, ← c33136, ← c33137] at e_520_10
  have e_520_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_11
  simp only [← c33138, ← c33139, k2, ← c33140] at e_520_11
  have hr := e_520_11
  simp only [e_520_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4798 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 520 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2324)) := by
  have c33141 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33142 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33143 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33144 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33145 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33146 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_520_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_12
  simp only [← c33141, ← c33142, ← c33143] at e_520_12
  have e_520_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_13
  simp only [← c33144, ← c33145, k2, ← c33146] at e_520_13
  have hr := e_520_13
  simp only [e_520_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4799 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 520 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2325)) := by
  have c33147 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33148 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33149 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33150 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33151 := (publicBatchWrapper86_copies1035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33152 := (publicBatchWrapper86_copies1036 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_520_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_14
  simp only [← c33147, ← c33148, ← c33149] at e_520_14
  have e_520_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_15
  simp only [← c33150, ← c33151, k2, ← c33152] at e_520_15
  have hr := e_520_15
  simp only [e_520_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 520 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2326)) := by
  have c33153 := (publicBatchWrapper86_copies1036 a h).2.1
  have c33154 := (publicBatchWrapper86_copies1036 a h).2.2.1
  have c33155 := (publicBatchWrapper86_copies1036 a h).2.2.2.1
  have c33156 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.1
  have c33157 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.1
  have c33158 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_520_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_16
  simp only [← c33153, ← c33154, ← c33155] at e_520_16
  have e_520_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_17
  simp only [← c33156, ← c33157, k2, ← c33158] at e_520_17
  have hr := e_520_17
  simp only [e_520_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 520 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2327)) := by
  have c33159 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.1
  have c33160 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.1
  have c33161 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.1
  have c33162 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33163 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33164 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_520_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_18
  simp only [← c33159, ← c33160, ← c33161] at e_520_18
  have e_520_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row520 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_520_19
  simp only [← c33162, ← c33163, k2, ← c33164] at e_520_19
  have hr := e_520_19
  simp only [e_520_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 521 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2328)) := by
  have c33165 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33166 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33167 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33168 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33169 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33170 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_521_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_0
  simp only [← c33165, ← c33166, ← c33167] at e_521_0
  have e_521_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_1
  simp only [← c33168, ← c33169, k2, ← c33170] at e_521_1
  have hr := e_521_1
  simp only [e_521_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 521 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2329)) := by
  have c33171 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33172 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33173 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33174 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33175 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33176 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_521_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_2
  simp only [← c33171, ← c33172, ← c33173] at e_521_2
  have e_521_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_3
  simp only [← c33174, ← c33175, k2, ← c33176] at e_521_3
  have hr := e_521_3
  simp only [e_521_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 521 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2330)) := by
  have c33177 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33178 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33179 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33180 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33181 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33182 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_521_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_4
  simp only [← c33177, ← c33178, ← c33179] at e_521_4
  have e_521_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_5
  simp only [← c33180, ← c33181, k2, ← c33182] at e_521_5
  have hr := e_521_5
  simp only [e_521_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 521 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2331)) := by
  have c33183 := (publicBatchWrapper86_copies1036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33184 := (publicBatchWrapper86_copies1037 a h).1
  have c33185 := (publicBatchWrapper86_copies1037 a h).2.1
  have c33186 := (publicBatchWrapper86_copies1037 a h).2.2.1
  have c33187 := (publicBatchWrapper86_copies1037 a h).2.2.2.1
  have c33188 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_521_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_6
  simp only [← c33183, ← c33184, ← c33185] at e_521_6
  have e_521_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_7
  simp only [← c33186, ← c33187, k2, ← c33188] at e_521_7
  have hr := e_521_7
  simp only [e_521_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 521 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2332)) := by
  have c33189 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.1
  have c33190 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.1
  have c33191 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.1
  have c33192 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.1
  have c33193 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.1
  have c33194 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_521_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_8
  simp only [← c33189, ← c33190, ← c33191] at e_521_8
  have e_521_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_9
  simp only [← c33192, ← c33193, k2, ← c33194] at e_521_9
  have hr := e_521_9
  simp only [e_521_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 521 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2333)) := by
  have c33195 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33196 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33197 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33198 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33199 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33200 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_521_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_10
  simp only [← c33195, ← c33196, ← c33197] at e_521_10
  have e_521_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_11
  simp only [← c33198, ← c33199, k2, ← c33200] at e_521_11
  have hr := e_521_11
  simp only [e_521_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 521 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2334)) := by
  have c33201 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33202 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33203 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33204 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33205 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33206 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_521_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_12
  simp only [← c33201, ← c33202, ← c33203] at e_521_12
  have e_521_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_13
  simp only [← c33204, ← c33205, k2, ← c33206] at e_521_13
  have hr := e_521_13
  simp only [e_521_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 521 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2335)) := by
  have c33207 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33208 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33209 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33210 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33211 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33212 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_521_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_14
  simp only [← c33207, ← c33208, ← c33209] at e_521_14
  have e_521_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_15
  simp only [← c33210, ← c33211, k2, ← c33212] at e_521_15
  have hr := e_521_15
  simp only [e_521_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 521 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2336)) := by
  have c33213 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33214 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33215 := (publicBatchWrapper86_copies1037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33216 := (publicBatchWrapper86_copies1038 a h).1
  have c33217 := (publicBatchWrapper86_copies1038 a h).2.1
  have c33218 := (publicBatchWrapper86_copies1038 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_521_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_16
  simp only [← c33213, ← c33214, ← c33215] at e_521_16
  have e_521_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_17
  simp only [← c33216, ← c33217, k2, ← c33218] at e_521_17
  have hr := e_521_17
  simp only [e_521_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 521 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2337)) := by
  have c33219 := (publicBatchWrapper86_copies1038 a h).2.2.2.1
  have c33220 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.1
  have c33221 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.1
  have c33222 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.1
  have c33223 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.1
  have c33224 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_521_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_18
  simp only [← c33219, ← c33220, ← c33221] at e_521_18
  have e_521_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row521 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_521_19
  simp only [← c33222, ← c33223, k2, ← c33224] at e_521_19
  have hr := e_521_19
  simp only [e_521_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 522 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2338)) := by
  have c33225 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.1
  have c33226 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33227 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33228 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33229 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33230 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_522_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_0
  simp only [← c33225, ← c33226, ← c33227] at e_522_0
  have e_522_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_1
  simp only [← c33228, ← c33229, k2, ← c33230] at e_522_1
  have hr := e_522_1
  simp only [e_522_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 522 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2339)) := by
  have c33231 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33232 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33233 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33234 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33235 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33236 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_522_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_2
  simp only [← c33231, ← c33232, ← c33233] at e_522_2
  have e_522_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_3
  simp only [← c33234, ← c33235, k2, ← c33236] at e_522_3
  have hr := e_522_3
  simp only [e_522_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 522 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2340)) := by
  have c33237 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33238 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33239 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33240 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33241 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33242 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_522_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_4
  simp only [← c33237, ← c33238, ← c33239] at e_522_4
  have e_522_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_5
  simp only [← c33240, ← c33241, k2, ← c33242] at e_522_5
  have hr := e_522_5
  simp only [e_522_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 522 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2341)) := by
  have c33243 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33244 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33245 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33246 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33247 := (publicBatchWrapper86_copies1038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33248 := (publicBatchWrapper86_copies1039 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_522_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_6
  simp only [← c33243, ← c33244, ← c33245] at e_522_6
  have e_522_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_7
  simp only [← c33246, ← c33247, k2, ← c33248] at e_522_7
  have hr := e_522_7
  simp only [e_522_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 522 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2342)) := by
  have c33249 := (publicBatchWrapper86_copies1039 a h).2.1
  have c33250 := (publicBatchWrapper86_copies1039 a h).2.2.1
  have c33251 := (publicBatchWrapper86_copies1039 a h).2.2.2.1
  have c33252 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.1
  have c33253 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.1
  have c33254 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_522_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_8
  simp only [← c33249, ← c33250, ← c33251] at e_522_8
  have e_522_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_9
  simp only [← c33252, ← c33253, k2, ← c33254] at e_522_9
  have hr := e_522_9
  simp only [e_522_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 522 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2343)) := by
  have c33255 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.1
  have c33256 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.1
  have c33257 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.1
  have c33258 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33259 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33260 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_522_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_10
  simp only [← c33255, ← c33256, ← c33257] at e_522_10
  have e_522_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_11
  simp only [← c33258, ← c33259, k2, ← c33260] at e_522_11
  have hr := e_522_11
  simp only [e_522_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 522 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2344)) := by
  have c33261 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33262 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33263 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33264 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33265 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33266 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_522_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_12
  simp only [← c33261, ← c33262, ← c33263] at e_522_12
  have e_522_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_13
  simp only [← c33264, ← c33265, k2, ← c33266] at e_522_13
  have hr := e_522_13
  simp only [e_522_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 522 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2345)) := by
  have c33267 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33268 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33269 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33270 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33271 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33272 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_522_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_14
  simp only [← c33267, ← c33268, ← c33269] at e_522_14
  have e_522_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_15
  simp only [← c33270, ← c33271, k2, ← c33272] at e_522_15
  have hr := e_522_15
  simp only [e_522_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 522 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2346)) := by
  have c33273 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33274 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33275 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33276 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33277 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33278 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_522_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_16
  simp only [← c33273, ← c33274, ← c33275] at e_522_16
  have e_522_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_17
  simp only [← c33276, ← c33277, k2, ← c33278] at e_522_17
  have hr := e_522_17
  simp only [e_522_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 522 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2347)) := by
  have c33279 := (publicBatchWrapper86_copies1039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33280 := (publicBatchWrapper86_copies1040 a h).1
  have c33281 := (publicBatchWrapper86_copies1040 a h).2.1
  have c33282 := (publicBatchWrapper86_copies1040 a h).2.2.1
  have c33283 := (publicBatchWrapper86_copies1040 a h).2.2.2.1
  have c33284 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_522_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_18
  simp only [← c33279, ← c33280, ← c33281] at e_522_18
  have e_522_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row522 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_522_19
  simp only [← c33282, ← c33283, k2, ← c33284] at e_522_19
  have hr := e_522_19
  simp only [e_522_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 523 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2348)) := by
  have c33285 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.1
  have c33286 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.1
  have c33287 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.1
  have c33288 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.1
  have c33289 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.1
  have c33290 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_523_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_0
  simp only [← c33285, ← c33286, ← c33287] at e_523_0
  have e_523_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_1
  simp only [← c33288, ← c33289, k2, ← c33290] at e_523_1
  have hr := e_523_1
  simp only [e_523_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 523 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2349)) := by
  have c33291 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33292 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33293 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33294 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33295 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33296 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_523_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_2
  simp only [← c33291, ← c33292, ← c33293] at e_523_2
  have e_523_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_3
  simp only [← c33294, ← c33295, k2, ← c33296] at e_523_3
  have hr := e_523_3
  simp only [e_523_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 523 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2350)) := by
  have c33297 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33298 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33299 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33300 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33301 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33302 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_523_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_4
  simp only [← c33297, ← c33298, ← c33299] at e_523_4
  have e_523_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_5
  simp only [← c33300, ← c33301, k2, ← c33302] at e_523_5
  have hr := e_523_5
  simp only [e_523_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 523 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2351)) := by
  have c33303 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33304 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33305 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33306 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33307 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33308 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_523_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_6
  simp only [← c33303, ← c33304, ← c33305] at e_523_6
  have e_523_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_7
  simp only [← c33306, ← c33307, k2, ← c33308] at e_523_7
  have hr := e_523_7
  simp only [e_523_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4826 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 523 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2352)) := by
  have c33309 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33310 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33311 := (publicBatchWrapper86_copies1040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33312 := (publicBatchWrapper86_copies1041 a h).1
  have c33313 := (publicBatchWrapper86_copies1041 a h).2.1
  have c33314 := (publicBatchWrapper86_copies1041 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_523_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_8
  simp only [← c33309, ← c33310, ← c33311] at e_523_8
  have e_523_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_9
  simp only [← c33312, ← c33313, k2, ← c33314] at e_523_9
  have hr := e_523_9
  simp only [e_523_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4827 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 523 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2353)) := by
  have c33315 := (publicBatchWrapper86_copies1041 a h).2.2.2.1
  have c33316 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.1
  have c33317 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.1
  have c33318 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.1
  have c33319 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.1
  have c33320 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_523_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_10
  simp only [← c33315, ← c33316, ← c33317] at e_523_10
  have e_523_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_11
  simp only [← c33318, ← c33319, k2, ← c33320] at e_523_11
  have hr := e_523_11
  simp only [e_523_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4828 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 523 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2354)) := by
  have c33321 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.1
  have c33322 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33323 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33324 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33325 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33326 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_523_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_12
  simp only [← c33321, ← c33322, ← c33323] at e_523_12
  have e_523_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_13
  simp only [← c33324, ← c33325, k2, ← c33326] at e_523_13
  have hr := e_523_13
  simp only [e_523_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4829 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 523 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2355)) := by
  have c33327 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33328 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33329 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33330 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33331 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33332 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_523_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_14
  simp only [← c33327, ← c33328, ← c33329] at e_523_14
  have e_523_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_15
  simp only [← c33330, ← c33331, k2, ← c33332] at e_523_15
  have hr := e_523_15
  simp only [e_523_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4830 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 523 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2356)) := by
  have c33333 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33334 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33335 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33336 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33337 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33338 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_523_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_16
  simp only [← c33333, ← c33334, ← c33335] at e_523_16
  have e_523_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_17
  simp only [← c33336, ← c33337, k2, ← c33338] at e_523_17
  have hr := e_523_17
  simp only [e_523_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4831 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 523 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2357)) := by
  have c33339 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33340 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33341 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33342 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33343 := (publicBatchWrapper86_copies1041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33344 := (publicBatchWrapper86_copies1042 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_523_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_18
  simp only [← c33339, ← c33340, ← c33341] at e_523_18
  have e_523_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row523 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_523_19
  simp only [← c33342, ← c33343, k2, ← c33344] at e_523_19
  have hr := e_523_19
  simp only [e_523_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4832 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 524 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2358)) := by
  have c33345 := (publicBatchWrapper86_copies1042 a h).2.1
  have c33346 := (publicBatchWrapper86_copies1042 a h).2.2.1
  have c33347 := (publicBatchWrapper86_copies1042 a h).2.2.2.1
  have c33348 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.1
  have c33349 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.1
  have c33350 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_524_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_0
  simp only [← c33345, ← c33346, ← c33347] at e_524_0
  have e_524_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_1
  simp only [← c33348, ← c33349, k2, ← c33350] at e_524_1
  have hr := e_524_1
  simp only [e_524_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4833 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 524 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2359)) := by
  have c33351 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.1
  have c33352 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.1
  have c33353 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.1
  have c33354 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33355 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33356 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_524_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_2
  simp only [← c33351, ← c33352, ← c33353] at e_524_2
  have e_524_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_3
  simp only [← c33354, ← c33355, k2, ← c33356] at e_524_3
  have hr := e_524_3
  simp only [e_524_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4834 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 524 23) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2360)) := by
  have c33357 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33358 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33359 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33360 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33361 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33362 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_524_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_4
  simp only [← c33357, ← c33358, ← c33359] at e_524_4
  have e_524_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_5
  simp only [← c33360, ← c33361, k2, ← c33362] at e_524_5
  have hr := e_524_5
  simp only [e_524_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4835 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 524 31) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2361)) := by
  have c33363 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33364 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33365 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33366 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33367 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33368 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_524_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_6
  simp only [← c33363, ← c33364, ← c33365] at e_524_6
  have e_524_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_7
  simp only [← c33366, ← c33367, k2, ← c33368] at e_524_7
  have hr := e_524_7
  simp only [e_524_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4836 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 524 39) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2362)) := by
  have c33369 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33370 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33371 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33372 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33373 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33374 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_524_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_8
  simp only [← c33369, ← c33370, ← c33371] at e_524_8
  have e_524_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_9
  simp only [← c33372, ← c33373, k2, ← c33374] at e_524_9
  have hr := e_524_9
  simp only [e_524_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4837 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 524 47) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2363)) := by
  have c33375 := (publicBatchWrapper86_copies1042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33376 := (publicBatchWrapper86_copies1043 a h).1
  have c33377 := (publicBatchWrapper86_copies1043 a h).2.1
  have c33378 := (publicBatchWrapper86_copies1043 a h).2.2.1
  have c33379 := (publicBatchWrapper86_copies1043 a h).2.2.2.1
  have c33380 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_524_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_10
  simp only [← c33375, ← c33376, ← c33377] at e_524_10
  have e_524_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_11
  simp only [← c33378, ← c33379, k2, ← c33380] at e_524_11
  have hr := e_524_11
  simp only [e_524_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4838 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 524 55) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2364)) := by
  have c33381 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.1
  have c33382 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.1
  have c33383 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.1
  have c33384 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.1
  have c33385 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.1
  have c33386 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_524_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_12
  simp only [← c33381, ← c33382, ← c33383] at e_524_12
  have e_524_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_13
  simp only [← c33384, ← c33385, k2, ← c33386] at e_524_13
  have hr := e_524_13
  simp only [e_524_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4839 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 524 63) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2365)) := by
  have c33387 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33388 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33389 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33390 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33391 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33392 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_524_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_14
  simp only [← c33387, ← c33388, ← c33389] at e_524_14
  have e_524_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_15
  simp only [← c33390, ← c33391, k2, ← c33392] at e_524_15
  have hr := e_524_15
  simp only [e_524_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4840 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 524 71) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2366)) := by
  have c33393 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33394 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33395 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33396 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33397 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33398 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_524_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_16
  simp only [← c33393, ← c33394, ← c33395] at e_524_16
  have e_524_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_17
  simp only [← c33396, ← c33397, k2, ← c33398] at e_524_17
  have hr := e_524_17
  simp only [e_524_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4841 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 524 79) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2367)) := by
  have c33399 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33400 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33401 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33402 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33403 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33404 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_524_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_18
  simp only [← c33399, ← c33400, ← c33401] at e_524_18
  have e_524_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row524 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_524_19
  simp only [← c33402, ← c33403, k2, ← c33404] at e_524_19
  have hr := e_524_19
  simp only [e_524_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4842 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 525 7) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2368)) := by
  have c33405 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33406 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33407 := (publicBatchWrapper86_copies1043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33408 := (publicBatchWrapper86_copies1044 a h).1
  have c33409 := (publicBatchWrapper86_copies1044 a h).2.1
  have c33410 := (publicBatchWrapper86_copies1044 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_525_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_0
  simp only [← c33405, ← c33406, ← c33407] at e_525_0
  have e_525_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_1
  simp only [← c33408, ← c33409, k2, ← c33410] at e_525_1
  have hr := e_525_1
  simp only [e_525_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4843 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 525 15) = bselect (a (.wire 13 31)) (a (.virt 200)) (a (.virt 2369)) := by
  have c33411 := (publicBatchWrapper86_copies1044 a h).2.2.2.1
  have c33412 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.1
  have c33413 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.1
  have c33414 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.1
  have c33415 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.1
  have c33416 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_525_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_2
  simp only [← c33411, ← c33412, ← c33413] at e_525_2
  have e_525_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_3
  simp only [← c33414, ← c33415, k2, ← c33416] at e_525_3
  have hr := e_525_3
  simp only [e_525_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4844 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 525 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2410)) := by
  have c33417 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.1
  have c33418 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33419 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33420 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33421 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33422 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_525_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_4
  simp only [← c33417, ← c33418, ← c33419] at e_525_4
  have e_525_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_5
  simp only [← c33420, ← c33421, k2, ← c33422] at e_525_5
  have hr := e_525_5
  simp only [e_525_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4845 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 525 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2411)) := by
  have c33423 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33424 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33425 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33426 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33427 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33428 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_525_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_6
  simp only [← c33423, ← c33424, ← c33425] at e_525_6
  have e_525_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_7
  simp only [← c33426, ← c33427, k2, ← c33428] at e_525_7
  have hr := e_525_7
  simp only [e_525_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4846 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 525 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2412)) := by
  have c33429 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33430 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33431 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33432 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33433 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33434 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_525_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_8
  simp only [← c33429, ← c33430, ← c33431] at e_525_8
  have e_525_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_9
  simp only [← c33432, ← c33433, k2, ← c33434] at e_525_9
  have hr := e_525_9
  simp only [e_525_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4847 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 525 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2413)) := by
  have c33435 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33436 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33437 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33438 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33439 := (publicBatchWrapper86_copies1044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33440 := (publicBatchWrapper86_copies1045 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_525_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_10
  simp only [← c33435, ← c33436, ← c33437] at e_525_10
  have e_525_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_11
  simp only [← c33438, ← c33439, k2, ← c33440] at e_525_11
  have hr := e_525_11
  simp only [e_525_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4848 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 525 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2414)) := by
  have c33441 := (publicBatchWrapper86_copies1045 a h).2.1
  have c33442 := (publicBatchWrapper86_copies1045 a h).2.2.1
  have c33443 := (publicBatchWrapper86_copies1045 a h).2.2.2.1
  have c33444 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.1
  have c33445 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.1
  have c33446 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_525_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_12
  simp only [← c33441, ← c33442, ← c33443] at e_525_12
  have e_525_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_13
  simp only [← c33444, ← c33445, k2, ← c33446] at e_525_13
  have hr := e_525_13
  simp only [e_525_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4849 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 525 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2415)) := by
  have c33447 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.1
  have c33448 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.1
  have c33449 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.1
  have c33450 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33451 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33452 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_525_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_14
  simp only [← c33447, ← c33448, ← c33449] at e_525_14
  have e_525_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_15
  simp only [← c33450, ← c33451, k2, ← c33452] at e_525_15
  have hr := e_525_15
  simp only [e_525_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4850 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 525 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2416)) := by
  have c33453 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33454 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33455 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33456 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33457 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33458 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_525_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_16
  simp only [← c33453, ← c33454, ← c33455] at e_525_16
  have e_525_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_17
  simp only [← c33456, ← c33457, k2, ← c33458] at e_525_17
  have hr := e_525_17
  simp only [e_525_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4851 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 525 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2417)) := by
  have c33459 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33460 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33461 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33462 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33463 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33464 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_525_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_18
  simp only [← c33459, ← c33460, ← c33461] at e_525_18
  have e_525_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row525 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_525_19
  simp only [← c33462, ← c33463, k2, ← c33464] at e_525_19
  have hr := e_525_19
  simp only [e_525_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4852 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 526 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2418)) := by
  have c33465 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33466 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33467 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33468 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33469 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33470 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_526_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_0
  simp only [← c33465, ← c33466, ← c33467] at e_526_0
  have e_526_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_1
  simp only [← c33468, ← c33469, k2, ← c33470] at e_526_1
  have hr := e_526_1
  simp only [e_526_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4853 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 526 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2419)) := by
  have c33471 := (publicBatchWrapper86_copies1045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33472 := (publicBatchWrapper86_copies1046 a h).1
  have c33473 := (publicBatchWrapper86_copies1046 a h).2.1
  have c33474 := (publicBatchWrapper86_copies1046 a h).2.2.1
  have c33475 := (publicBatchWrapper86_copies1046 a h).2.2.2.1
  have c33476 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_526_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_2
  simp only [← c33471, ← c33472, ← c33473] at e_526_2
  have e_526_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_3
  simp only [← c33474, ← c33475, k2, ← c33476] at e_526_3
  have hr := e_526_3
  simp only [e_526_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4854 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 526 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2420)) := by
  have c33477 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.1
  have c33478 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.1
  have c33479 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.1
  have c33480 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.1
  have c33481 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.1
  have c33482 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_526_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_4
  simp only [← c33477, ← c33478, ← c33479] at e_526_4
  have e_526_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_5
  simp only [← c33480, ← c33481, k2, ← c33482] at e_526_5
  have hr := e_526_5
  simp only [e_526_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4855 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 526 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2421)) := by
  have c33483 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33484 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33485 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33486 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33487 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33488 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_526_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_6
  simp only [← c33483, ← c33484, ← c33485] at e_526_6
  have e_526_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_7
  simp only [← c33486, ← c33487, k2, ← c33488] at e_526_7
  have hr := e_526_7
  simp only [e_526_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4856 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 526 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2422)) := by
  have c33489 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33490 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33491 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33492 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33493 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33494 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_526_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_8
  simp only [← c33489, ← c33490, ← c33491] at e_526_8
  have e_526_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_9
  simp only [← c33492, ← c33493, k2, ← c33494] at e_526_9
  have hr := e_526_9
  simp only [e_526_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4857 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 526 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2423)) := by
  have c33495 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33496 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33497 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33498 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33499 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33500 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_526_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_10
  simp only [← c33495, ← c33496, ← c33497] at e_526_10
  have e_526_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_11
  simp only [← c33498, ← c33499, k2, ← c33500] at e_526_11
  have hr := e_526_11
  simp only [e_526_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4858 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 526 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2424)) := by
  have c33501 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33502 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33503 := (publicBatchWrapper86_copies1046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33504 := (publicBatchWrapper86_copies1047 a h).1
  have c33505 := (publicBatchWrapper86_copies1047 a h).2.1
  have c33506 := (publicBatchWrapper86_copies1047 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_526_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_12
  simp only [← c33501, ← c33502, ← c33503] at e_526_12
  have e_526_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_13
  simp only [← c33504, ← c33505, k2, ← c33506] at e_526_13
  have hr := e_526_13
  simp only [e_526_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4859 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 526 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2425)) := by
  have c33507 := (publicBatchWrapper86_copies1047 a h).2.2.2.1
  have c33508 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.1
  have c33509 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.1
  have c33510 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.1
  have c33511 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.1
  have c33512 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_526_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_14
  simp only [← c33507, ← c33508, ← c33509] at e_526_14
  have e_526_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_15
  simp only [← c33510, ← c33511, k2, ← c33512] at e_526_15
  have hr := e_526_15
  simp only [e_526_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4860 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 526 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2426)) := by
  have c33513 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.1
  have c33514 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33515 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33516 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33517 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33518 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_526_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_16
  simp only [← c33513, ← c33514, ← c33515] at e_526_16
  have e_526_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_17
  simp only [← c33516, ← c33517, k2, ← c33518] at e_526_17
  have hr := e_526_17
  simp only [e_526_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4861 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 526 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2427)) := by
  have c33519 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33520 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33521 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33522 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33523 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33524 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_526_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_18
  simp only [← c33519, ← c33520, ← c33521] at e_526_18
  have e_526_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row526 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_526_19
  simp only [← c33522, ← c33523, k2, ← c33524] at e_526_19
  have hr := e_526_19
  simp only [e_526_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4862 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 527 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2428)) := by
  have c33525 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33526 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33527 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33528 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33529 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33530 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_527_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_0
  simp only [← c33525, ← c33526, ← c33527] at e_527_0
  have e_527_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_1
  simp only [← c33528, ← c33529, k2, ← c33530] at e_527_1
  have hr := e_527_1
  simp only [e_527_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4863 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 527 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2429)) := by
  have c33531 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33532 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33533 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33534 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33535 := (publicBatchWrapper86_copies1047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33536 := (publicBatchWrapper86_copies1048 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_527_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_2
  simp only [← c33531, ← c33532, ← c33533] at e_527_2
  have e_527_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_3
  simp only [← c33534, ← c33535, k2, ← c33536] at e_527_3
  have hr := e_527_3
  simp only [e_527_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4864 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 527 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2430)) := by
  have c33537 := (publicBatchWrapper86_copies1048 a h).2.1
  have c33538 := (publicBatchWrapper86_copies1048 a h).2.2.1
  have c33539 := (publicBatchWrapper86_copies1048 a h).2.2.2.1
  have c33540 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.1
  have c33541 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.1
  have c33542 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_527_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_4
  simp only [← c33537, ← c33538, ← c33539] at e_527_4
  have e_527_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_5
  simp only [← c33540, ← c33541, k2, ← c33542] at e_527_5
  have hr := e_527_5
  simp only [e_527_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4865 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 527 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2431)) := by
  have c33543 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.1
  have c33544 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.1
  have c33545 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.1
  have c33546 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33547 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33548 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_527_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_6
  simp only [← c33543, ← c33544, ← c33545] at e_527_6
  have e_527_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_7
  simp only [← c33546, ← c33547, k2, ← c33548] at e_527_7
  have hr := e_527_7
  simp only [e_527_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4866 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 527 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2432)) := by
  have c33549 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33550 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33551 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33552 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33553 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33554 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_527_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_8
  simp only [← c33549, ← c33550, ← c33551] at e_527_8
  have e_527_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_9
  simp only [← c33552, ← c33553, k2, ← c33554] at e_527_9
  have hr := e_527_9
  simp only [e_527_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4867 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 527 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2433)) := by
  have c33555 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33556 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33557 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33558 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33559 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33560 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_527_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_10
  simp only [← c33555, ← c33556, ← c33557] at e_527_10
  have e_527_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_11
  simp only [← c33558, ← c33559, k2, ← c33560] at e_527_11
  have hr := e_527_11
  simp only [e_527_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4868 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 527 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2434)) := by
  have c33561 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33562 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33563 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33564 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33565 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33566 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_527_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_12
  simp only [← c33561, ← c33562, ← c33563] at e_527_12
  have e_527_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_13
  simp only [← c33564, ← c33565, k2, ← c33566] at e_527_13
  have hr := e_527_13
  simp only [e_527_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4869 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 527 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2435)) := by
  have c33567 := (publicBatchWrapper86_copies1048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33568 := (publicBatchWrapper86_copies1049 a h).1
  have c33569 := (publicBatchWrapper86_copies1049 a h).2.1
  have c33570 := (publicBatchWrapper86_copies1049 a h).2.2.1
  have c33571 := (publicBatchWrapper86_copies1049 a h).2.2.2.1
  have c33572 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_527_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_14
  simp only [← c33567, ← c33568, ← c33569] at e_527_14
  have e_527_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_15
  simp only [← c33570, ← c33571, k2, ← c33572] at e_527_15
  have hr := e_527_15
  simp only [e_527_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 527 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2436)) := by
  have c33573 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.1
  have c33574 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.1
  have c33575 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.1
  have c33576 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.1
  have c33577 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.1
  have c33578 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_527_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_16
  simp only [← c33573, ← c33574, ← c33575] at e_527_16
  have e_527_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_17
  simp only [← c33576, ← c33577, k2, ← c33578] at e_527_17
  have hr := e_527_17
  simp only [e_527_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 527 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2437)) := by
  have c33579 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33580 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33581 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33582 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33583 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33584 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_527_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_18
  simp only [← c33579, ← c33580, ← c33581] at e_527_18
  have e_527_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row527 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_527_19
  simp only [← c33582, ← c33583, k2, ← c33584] at e_527_19
  have hr := e_527_19
  simp only [e_527_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 528 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2438)) := by
  have c33585 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33586 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33587 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33588 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33589 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33590 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_528_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_0
  simp only [← c33585, ← c33586, ← c33587] at e_528_0
  have e_528_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_1
  simp only [← c33588, ← c33589, k2, ← c33590] at e_528_1
  have hr := e_528_1
  simp only [e_528_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 528 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2439)) := by
  have c33591 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33592 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33593 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33594 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33595 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33596 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_528_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_2
  simp only [← c33591, ← c33592, ← c33593] at e_528_2
  have e_528_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_3
  simp only [← c33594, ← c33595, k2, ← c33596] at e_528_3
  have hr := e_528_3
  simp only [e_528_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 528 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2440)) := by
  have c33597 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33598 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33599 := (publicBatchWrapper86_copies1049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33600 := (publicBatchWrapper86_copies1050 a h).1
  have c33601 := (publicBatchWrapper86_copies1050 a h).2.1
  have c33602 := (publicBatchWrapper86_copies1050 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_528_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_4
  simp only [← c33597, ← c33598, ← c33599] at e_528_4
  have e_528_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_5
  simp only [← c33600, ← c33601, k2, ← c33602] at e_528_5
  have hr := e_528_5
  simp only [e_528_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 528 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2441)) := by
  have c33603 := (publicBatchWrapper86_copies1050 a h).2.2.2.1
  have c33604 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.1
  have c33605 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.1
  have c33606 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.1
  have c33607 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.1
  have c33608 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_528_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_6
  simp only [← c33603, ← c33604, ← c33605] at e_528_6
  have e_528_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_7
  simp only [← c33606, ← c33607, k2, ← c33608] at e_528_7
  have hr := e_528_7
  simp only [e_528_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 528 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2442)) := by
  have c33609 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.1
  have c33610 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33611 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33612 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33613 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33614 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_528_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_8
  simp only [← c33609, ← c33610, ← c33611] at e_528_8
  have e_528_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_9
  simp only [← c33612, ← c33613, k2, ← c33614] at e_528_9
  have hr := e_528_9
  simp only [e_528_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4877 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 528 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2443)) := by
  have c33615 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33616 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33617 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33618 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33619 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33620 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_528_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_10
  simp only [← c33615, ← c33616, ← c33617] at e_528_10
  have e_528_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_11
  simp only [← c33618, ← c33619, k2, ← c33620] at e_528_11
  have hr := e_528_11
  simp only [e_528_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4878 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 528 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2444)) := by
  have c33621 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33622 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33623 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33624 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33625 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33626 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_528_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_12
  simp only [← c33621, ← c33622, ← c33623] at e_528_12
  have e_528_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_13
  simp only [← c33624, ← c33625, k2, ← c33626] at e_528_13
  have hr := e_528_13
  simp only [e_528_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4879 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 528 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2445)) := by
  have c33627 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33628 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33629 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33630 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33631 := (publicBatchWrapper86_copies1050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33632 := (publicBatchWrapper86_copies1051 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_528_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_14
  simp only [← c33627, ← c33628, ← c33629] at e_528_14
  have e_528_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_15
  simp only [← c33630, ← c33631, k2, ← c33632] at e_528_15
  have hr := e_528_15
  simp only [e_528_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4880 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 528 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2446)) := by
  have c33633 := (publicBatchWrapper86_copies1051 a h).2.1
  have c33634 := (publicBatchWrapper86_copies1051 a h).2.2.1
  have c33635 := (publicBatchWrapper86_copies1051 a h).2.2.2.1
  have c33636 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.1
  have c33637 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.1
  have c33638 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_528_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_16
  simp only [← c33633, ← c33634, ← c33635] at e_528_16
  have e_528_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_17
  simp only [← c33636, ← c33637, k2, ← c33638] at e_528_17
  have hr := e_528_17
  simp only [e_528_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4881 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 528 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2447)) := by
  have c33639 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.1
  have c33640 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.1
  have c33641 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.1
  have c33642 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33643 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33644 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_528_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_18
  simp only [← c33639, ← c33640, ← c33641] at e_528_18
  have e_528_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row528 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_528_19
  simp only [← c33642, ← c33643, k2, ← c33644] at e_528_19
  have hr := e_528_19
  simp only [e_528_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4882 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 529 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2448)) := by
  have c33645 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33646 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33647 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33648 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33649 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33650 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_529_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_0
  simp only [← c33645, ← c33646, ← c33647] at e_529_0
  have e_529_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_1
  simp only [← c33648, ← c33649, k2, ← c33650] at e_529_1
  have hr := e_529_1
  simp only [e_529_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4883 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 529 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2449)) := by
  have c33651 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33652 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33653 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33654 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33655 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33656 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_529_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_2
  simp only [← c33651, ← c33652, ← c33653] at e_529_2
  have e_529_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_3
  simp only [← c33654, ← c33655, k2, ← c33656] at e_529_3
  have hr := e_529_3
  simp only [e_529_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4884 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 529 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2450)) := by
  have c33657 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33658 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33659 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33660 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33661 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33662 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_529_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_4
  simp only [← c33657, ← c33658, ← c33659] at e_529_4
  have e_529_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_5
  simp only [← c33660, ← c33661, k2, ← c33662] at e_529_5
  have hr := e_529_5
  simp only [e_529_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4885 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 529 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2451)) := by
  have c33663 := (publicBatchWrapper86_copies1051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33664 := (publicBatchWrapper86_copies1052 a h).1
  have c33665 := (publicBatchWrapper86_copies1052 a h).2.1
  have c33666 := (publicBatchWrapper86_copies1052 a h).2.2.1
  have c33667 := (publicBatchWrapper86_copies1052 a h).2.2.2.1
  have c33668 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_529_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_6
  simp only [← c33663, ← c33664, ← c33665] at e_529_6
  have e_529_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_7
  simp only [← c33666, ← c33667, k2, ← c33668] at e_529_7
  have hr := e_529_7
  simp only [e_529_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4886 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 529 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2452)) := by
  have c33669 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.1
  have c33670 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.1
  have c33671 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.1
  have c33672 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.1
  have c33673 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.1
  have c33674 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_529_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_8
  simp only [← c33669, ← c33670, ← c33671] at e_529_8
  have e_529_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_9
  simp only [← c33672, ← c33673, k2, ← c33674] at e_529_9
  have hr := e_529_9
  simp only [e_529_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4887 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 529 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2453)) := by
  have c33675 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33676 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33677 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33678 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33679 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33680 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_529_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_10
  simp only [← c33675, ← c33676, ← c33677] at e_529_10
  have e_529_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_11
  simp only [← c33678, ← c33679, k2, ← c33680] at e_529_11
  have hr := e_529_11
  simp only [e_529_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4888 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 529 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2454)) := by
  have c33681 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33682 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33683 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33684 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33685 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33686 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_529_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_12
  simp only [← c33681, ← c33682, ← c33683] at e_529_12
  have e_529_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_13
  simp only [← c33684, ← c33685, k2, ← c33686] at e_529_13
  have hr := e_529_13
  simp only [e_529_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4889 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 529 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2455)) := by
  have c33687 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33688 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33689 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33690 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33691 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33692 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_529_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_14
  simp only [← c33687, ← c33688, ← c33689] at e_529_14
  have e_529_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_15
  simp only [← c33690, ← c33691, k2, ← c33692] at e_529_15
  have hr := e_529_15
  simp only [e_529_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4890 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 529 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2456)) := by
  have c33693 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33694 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33695 := (publicBatchWrapper86_copies1052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33696 := (publicBatchWrapper86_copies1053 a h).1
  have c33697 := (publicBatchWrapper86_copies1053 a h).2.1
  have c33698 := (publicBatchWrapper86_copies1053 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_529_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_16
  simp only [← c33693, ← c33694, ← c33695] at e_529_16
  have e_529_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_17
  simp only [← c33696, ← c33697, k2, ← c33698] at e_529_17
  have hr := e_529_17
  simp only [e_529_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4891 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 529 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2457)) := by
  have c33699 := (publicBatchWrapper86_copies1053 a h).2.2.2.1
  have c33700 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.1
  have c33701 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.1
  have c33702 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.1
  have c33703 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.1
  have c33704 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_529_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_18
  simp only [← c33699, ← c33700, ← c33701] at e_529_18
  have e_529_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row529 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_529_19
  simp only [← c33702, ← c33703, k2, ← c33704] at e_529_19
  have hr := e_529_19
  simp only [e_529_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4892 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 530 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2458)) := by
  have c33705 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.1
  have c33706 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33707 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33708 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33709 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33710 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_530_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_0
  simp only [← c33705, ← c33706, ← c33707] at e_530_0
  have e_530_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_1
  simp only [← c33708, ← c33709, k2, ← c33710] at e_530_1
  have hr := e_530_1
  simp only [e_530_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4893 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 530 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2459)) := by
  have c33711 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33712 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33713 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33714 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33715 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33716 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_530_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_2
  simp only [← c33711, ← c33712, ← c33713] at e_530_2
  have e_530_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_3
  simp only [← c33714, ← c33715, k2, ← c33716] at e_530_3
  have hr := e_530_3
  simp only [e_530_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4894 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 530 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2460)) := by
  have c33717 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33718 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33719 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33720 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33721 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33722 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_530_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_4
  simp only [← c33717, ← c33718, ← c33719] at e_530_4
  have e_530_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_5
  simp only [← c33720, ← c33721, k2, ← c33722] at e_530_5
  have hr := e_530_5
  simp only [e_530_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4895 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 530 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2461)) := by
  have c33723 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33724 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33725 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33726 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33727 := (publicBatchWrapper86_copies1053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33728 := (publicBatchWrapper86_copies1054 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_530_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_6
  simp only [← c33723, ← c33724, ← c33725] at e_530_6
  have e_530_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_7
  simp only [← c33726, ← c33727, k2, ← c33728] at e_530_7
  have hr := e_530_7
  simp only [e_530_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4896 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 530 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2462)) := by
  have c33729 := (publicBatchWrapper86_copies1054 a h).2.1
  have c33730 := (publicBatchWrapper86_copies1054 a h).2.2.1
  have c33731 := (publicBatchWrapper86_copies1054 a h).2.2.2.1
  have c33732 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.1
  have c33733 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.1
  have c33734 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_530_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_8
  simp only [← c33729, ← c33730, ← c33731] at e_530_8
  have e_530_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_9
  simp only [← c33732, ← c33733, k2, ← c33734] at e_530_9
  have hr := e_530_9
  simp only [e_530_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4897 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 530 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2463)) := by
  have c33735 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.1
  have c33736 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.1
  have c33737 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.1
  have c33738 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33739 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33740 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_530_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_10
  simp only [← c33735, ← c33736, ← c33737] at e_530_10
  have e_530_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_11
  simp only [← c33738, ← c33739, k2, ← c33740] at e_530_11
  have hr := e_530_11
  simp only [e_530_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4898 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 530 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2464)) := by
  have c33741 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33742 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33743 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33744 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33745 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33746 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_530_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_12
  simp only [← c33741, ← c33742, ← c33743] at e_530_12
  have e_530_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_13
  simp only [← c33744, ← c33745, k2, ← c33746] at e_530_13
  have hr := e_530_13
  simp only [e_530_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4899 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 530 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2465)) := by
  have c33747 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33748 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33749 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33750 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33751 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33752 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_530_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_14
  simp only [← c33747, ← c33748, ← c33749] at e_530_14
  have e_530_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_15
  simp only [← c33750, ← c33751, k2, ← c33752] at e_530_15
  have hr := e_530_15
  simp only [e_530_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4900 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 530 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2466)) := by
  have c33753 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33754 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33755 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33756 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33757 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33758 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_530_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_16
  simp only [← c33753, ← c33754, ← c33755] at e_530_16
  have e_530_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_17
  simp only [← c33756, ← c33757, k2, ← c33758] at e_530_17
  have hr := e_530_17
  simp only [e_530_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4901 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 530 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2467)) := by
  have c33759 := (publicBatchWrapper86_copies1054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33760 := (publicBatchWrapper86_copies1055 a h).1
  have c33761 := (publicBatchWrapper86_copies1055 a h).2.1
  have c33762 := (publicBatchWrapper86_copies1055 a h).2.2.1
  have c33763 := (publicBatchWrapper86_copies1055 a h).2.2.2.1
  have c33764 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_530_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_18
  simp only [← c33759, ← c33760, ← c33761] at e_530_18
  have e_530_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row530 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_530_19
  simp only [← c33762, ← c33763, k2, ← c33764] at e_530_19
  have hr := e_530_19
  simp only [e_530_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4902 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 531 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2468)) := by
  have c33765 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.1
  have c33766 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.1
  have c33767 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.1
  have c33768 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.1
  have c33769 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.1
  have c33770 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_531_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_0
  simp only [← c33765, ← c33766, ← c33767] at e_531_0
  have e_531_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_1
  simp only [← c33768, ← c33769, k2, ← c33770] at e_531_1
  have hr := e_531_1
  simp only [e_531_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4903 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 531 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2469)) := by
  have c33771 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33772 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33773 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33774 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33775 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33776 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_531_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_2
  simp only [← c33771, ← c33772, ← c33773] at e_531_2
  have e_531_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_3
  simp only [← c33774, ← c33775, k2, ← c33776] at e_531_3
  have hr := e_531_3
  simp only [e_531_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4904 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 531 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2470)) := by
  have c33777 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33778 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33779 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33780 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33781 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33782 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_531_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_4
  simp only [← c33777, ← c33778, ← c33779] at e_531_4
  have e_531_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_5
  simp only [← c33780, ← c33781, k2, ← c33782] at e_531_5
  have hr := e_531_5
  simp only [e_531_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4905 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 531 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2471)) := by
  have c33783 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33784 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33785 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33786 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33787 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33788 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_531_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_6
  simp only [← c33783, ← c33784, ← c33785] at e_531_6
  have e_531_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_7
  simp only [← c33786, ← c33787, k2, ← c33788] at e_531_7
  have hr := e_531_7
  simp only [e_531_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4906 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 531 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2472)) := by
  have c33789 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33790 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33791 := (publicBatchWrapper86_copies1055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33792 := (publicBatchWrapper86_copies1056 a h).1
  have c33793 := (publicBatchWrapper86_copies1056 a h).2.1
  have c33794 := (publicBatchWrapper86_copies1056 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_531_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_8
  simp only [← c33789, ← c33790, ← c33791] at e_531_8
  have e_531_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_9
  simp only [← c33792, ← c33793, k2, ← c33794] at e_531_9
  have hr := e_531_9
  simp only [e_531_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4907 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 531 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2473)) := by
  have c33795 := (publicBatchWrapper86_copies1056 a h).2.2.2.1
  have c33796 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.1
  have c33797 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.1
  have c33798 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.1
  have c33799 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.1
  have c33800 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_531_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_10
  simp only [← c33795, ← c33796, ← c33797] at e_531_10
  have e_531_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_11
  simp only [← c33798, ← c33799, k2, ← c33800] at e_531_11
  have hr := e_531_11
  simp only [e_531_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4908 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 531 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2474)) := by
  have c33801 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.1
  have c33802 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33803 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33804 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33805 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33806 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_531_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_12
  simp only [← c33801, ← c33802, ← c33803] at e_531_12
  have e_531_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_13
  simp only [← c33804, ← c33805, k2, ← c33806] at e_531_13
  have hr := e_531_13
  simp only [e_531_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4909 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 531 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2475)) := by
  have c33807 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33808 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33809 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33810 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33811 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33812 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_531_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_14
  simp only [← c33807, ← c33808, ← c33809] at e_531_14
  have e_531_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_15
  simp only [← c33810, ← c33811, k2, ← c33812] at e_531_15
  have hr := e_531_15
  simp only [e_531_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4910 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 531 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2476)) := by
  have c33813 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33814 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33815 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33816 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33817 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33818 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_531_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_16
  simp only [← c33813, ← c33814, ← c33815] at e_531_16
  have e_531_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_17
  simp only [← c33816, ← c33817, k2, ← c33818] at e_531_17
  have hr := e_531_17
  simp only [e_531_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4911 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 531 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2477)) := by
  have c33819 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33820 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33821 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33822 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33823 := (publicBatchWrapper86_copies1056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33824 := (publicBatchWrapper86_copies1057 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_531_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_18
  simp only [← c33819, ← c33820, ← c33821] at e_531_18
  have e_531_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row531 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_531_19
  simp only [← c33822, ← c33823, k2, ← c33824] at e_531_19
  have hr := e_531_19
  simp only [e_531_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4912 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 532 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2478)) := by
  have c33825 := (publicBatchWrapper86_copies1057 a h).2.1
  have c33826 := (publicBatchWrapper86_copies1057 a h).2.2.1
  have c33827 := (publicBatchWrapper86_copies1057 a h).2.2.2.1
  have c33828 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.1
  have c33829 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.1
  have c33830 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_532_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_0
  simp only [← c33825, ← c33826, ← c33827] at e_532_0
  have e_532_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_1
  simp only [← c33828, ← c33829, k2, ← c33830] at e_532_1
  have hr := e_532_1
  simp only [e_532_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4913 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 532 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2479)) := by
  have c33831 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.1
  have c33832 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.1
  have c33833 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.1
  have c33834 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33835 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33836 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_532_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_2
  simp only [← c33831, ← c33832, ← c33833] at e_532_2
  have e_532_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_3
  simp only [← c33834, ← c33835, k2, ← c33836] at e_532_3
  have hr := e_532_3
  simp only [e_532_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4914 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 532 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2480)) := by
  have c33837 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33838 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33839 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33840 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33841 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33842 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_532_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_4
  simp only [← c33837, ← c33838, ← c33839] at e_532_4
  have e_532_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_5
  simp only [← c33840, ← c33841, k2, ← c33842] at e_532_5
  have hr := e_532_5
  simp only [e_532_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4915 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 532 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2481)) := by
  have c33843 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33844 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33845 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33846 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33847 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33848 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_532_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_6
  simp only [← c33843, ← c33844, ← c33845] at e_532_6
  have e_532_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_7
  simp only [← c33846, ← c33847, k2, ← c33848] at e_532_7
  have hr := e_532_7
  simp only [e_532_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4916 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 532 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2482)) := by
  have c33849 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33850 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33851 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33852 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33853 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33854 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_532_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_8
  simp only [← c33849, ← c33850, ← c33851] at e_532_8
  have e_532_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_9
  simp only [← c33852, ← c33853, k2, ← c33854] at e_532_9
  have hr := e_532_9
  simp only [e_532_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4917 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 532 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2483)) := by
  have c33855 := (publicBatchWrapper86_copies1057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33856 := (publicBatchWrapper86_copies1058 a h).1
  have c33857 := (publicBatchWrapper86_copies1058 a h).2.1
  have c33858 := (publicBatchWrapper86_copies1058 a h).2.2.1
  have c33859 := (publicBatchWrapper86_copies1058 a h).2.2.2.1
  have c33860 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_532_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_10
  simp only [← c33855, ← c33856, ← c33857] at e_532_10
  have e_532_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_11
  simp only [← c33858, ← c33859, k2, ← c33860] at e_532_11
  have hr := e_532_11
  simp only [e_532_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4918 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 532 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2484)) := by
  have c33861 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.1
  have c33862 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.1
  have c33863 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.1
  have c33864 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.1
  have c33865 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.1
  have c33866 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_532_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_12
  simp only [← c33861, ← c33862, ← c33863] at e_532_12
  have e_532_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_13
  simp only [← c33864, ← c33865, k2, ← c33866] at e_532_13
  have hr := e_532_13
  simp only [e_532_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4919 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 532 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2485)) := by
  have c33867 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33868 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33869 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33870 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33871 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33872 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_532_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_14
  simp only [← c33867, ← c33868, ← c33869] at e_532_14
  have e_532_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_15
  simp only [← c33870, ← c33871, k2, ← c33872] at e_532_15
  have hr := e_532_15
  simp only [e_532_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4920 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 532 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2486)) := by
  have c33873 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33874 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33875 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33876 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33877 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33878 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_532_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_16
  simp only [← c33873, ← c33874, ← c33875] at e_532_16
  have e_532_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_17
  simp only [← c33876, ← c33877, k2, ← c33878] at e_532_17
  have hr := e_532_17
  simp only [e_532_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4921 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 532 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2487)) := by
  have c33879 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33880 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33881 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33882 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33883 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33884 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_532_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_18
  simp only [← c33879, ← c33880, ← c33881] at e_532_18
  have e_532_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row532 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_532_19
  simp only [← c33882, ← c33883, k2, ← c33884] at e_532_19
  have hr := e_532_19
  simp only [e_532_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4922 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 533 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2488)) := by
  have c33885 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33886 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33887 := (publicBatchWrapper86_copies1058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33888 := (publicBatchWrapper86_copies1059 a h).1
  have c33889 := (publicBatchWrapper86_copies1059 a h).2.1
  have c33890 := (publicBatchWrapper86_copies1059 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_533_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_0
  simp only [← c33885, ← c33886, ← c33887] at e_533_0
  have e_533_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_1
  simp only [← c33888, ← c33889, k2, ← c33890] at e_533_1
  have hr := e_533_1
  simp only [e_533_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4923 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 533 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2489)) := by
  have c33891 := (publicBatchWrapper86_copies1059 a h).2.2.2.1
  have c33892 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.1
  have c33893 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.1
  have c33894 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.1
  have c33895 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.1
  have c33896 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_533_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_2
  simp only [← c33891, ← c33892, ← c33893] at e_533_2
  have e_533_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_3
  simp only [← c33894, ← c33895, k2, ← c33896] at e_533_3
  have hr := e_533_3
  simp only [e_533_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4924 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 533 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2490)) := by
  have c33897 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.1
  have c33898 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33899 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33900 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33901 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33902 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_533_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_4
  simp only [← c33897, ← c33898, ← c33899] at e_533_4
  have e_533_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_5
  simp only [← c33900, ← c33901, k2, ← c33902] at e_533_5
  have hr := e_533_5
  simp only [e_533_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4925 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 533 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2491)) := by
  have c33903 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33904 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33905 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33906 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33907 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33908 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_533_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_6
  simp only [← c33903, ← c33904, ← c33905] at e_533_6
  have e_533_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_7
  simp only [← c33906, ← c33907, k2, ← c33908] at e_533_7
  have hr := e_533_7
  simp only [e_533_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4926 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 533 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2492)) := by
  have c33909 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33910 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33911 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33912 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33913 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33914 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_533_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_8
  simp only [← c33909, ← c33910, ← c33911] at e_533_8
  have e_533_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_9
  simp only [← c33912, ← c33913, k2, ← c33914] at e_533_9
  have hr := e_533_9
  simp only [e_533_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4927 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 533 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2493)) := by
  have c33915 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33916 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33917 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33918 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33919 := (publicBatchWrapper86_copies1059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33920 := (publicBatchWrapper86_copies1060 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_533_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_10
  simp only [← c33915, ← c33916, ← c33917] at e_533_10
  have e_533_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_11
  simp only [← c33918, ← c33919, k2, ← c33920] at e_533_11
  have hr := e_533_11
  simp only [e_533_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4928 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 533 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2494)) := by
  have c33921 := (publicBatchWrapper86_copies1060 a h).2.1
  have c33922 := (publicBatchWrapper86_copies1060 a h).2.2.1
  have c33923 := (publicBatchWrapper86_copies1060 a h).2.2.2.1
  have c33924 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.1
  have c33925 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.1
  have c33926 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_533_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_12
  simp only [← c33921, ← c33922, ← c33923] at e_533_12
  have e_533_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_13
  simp only [← c33924, ← c33925, k2, ← c33926] at e_533_13
  have hr := e_533_13
  simp only [e_533_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4929 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 533 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2495)) := by
  have c33927 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.1
  have c33928 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.1
  have c33929 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.1
  have c33930 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33931 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33932 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_533_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_14
  simp only [← c33927, ← c33928, ← c33929] at e_533_14
  have e_533_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_15
  simp only [← c33930, ← c33931, k2, ← c33932] at e_533_15
  have hr := e_533_15
  simp only [e_533_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4930 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 533 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2496)) := by
  have c33933 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33934 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33935 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33936 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33937 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33938 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_533_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_16
  simp only [← c33933, ← c33934, ← c33935] at e_533_16
  have e_533_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_17
  simp only [← c33936, ← c33937, k2, ← c33938] at e_533_17
  have hr := e_533_17
  simp only [e_533_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4931 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 533 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2497)) := by
  have c33939 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33940 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33941 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33942 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33943 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33944 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_533_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_18
  simp only [← c33939, ← c33940, ← c33941] at e_533_18
  have e_533_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row533 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_533_19
  simp only [← c33942, ← c33943, k2, ← c33944] at e_533_19
  have hr := e_533_19
  simp only [e_533_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4932 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 534 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2498)) := by
  have c33945 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33946 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33947 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33948 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33949 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33950 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_534_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_0
  simp only [← c33945, ← c33946, ← c33947] at e_534_0
  have e_534_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_1
  simp only [← c33948, ← c33949, k2, ← c33950] at e_534_1
  have hr := e_534_1
  simp only [e_534_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4933 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 534 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2499)) := by
  have c33951 := (publicBatchWrapper86_copies1060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33952 := (publicBatchWrapper86_copies1061 a h).1
  have c33953 := (publicBatchWrapper86_copies1061 a h).2.1
  have c33954 := (publicBatchWrapper86_copies1061 a h).2.2.1
  have c33955 := (publicBatchWrapper86_copies1061 a h).2.2.2.1
  have c33956 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_534_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_2
  simp only [← c33951, ← c33952, ← c33953] at e_534_2
  have e_534_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_3
  simp only [← c33954, ← c33955, k2, ← c33956] at e_534_3
  have hr := e_534_3
  simp only [e_534_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4934 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 534 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2500)) := by
  have c33957 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.1
  have c33958 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.1
  have c33959 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.1
  have c33960 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.1
  have c33961 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.1
  have c33962 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_534_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_4
  simp only [← c33957, ← c33958, ← c33959] at e_534_4
  have e_534_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_5
  simp only [← c33960, ← c33961, k2, ← c33962] at e_534_5
  have hr := e_534_5
  simp only [e_534_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4935 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 534 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2501)) := by
  have c33963 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33964 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33965 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33966 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33967 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33968 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_534_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_6
  simp only [← c33963, ← c33964, ← c33965] at e_534_6
  have e_534_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_7
  simp only [← c33966, ← c33967, k2, ← c33968] at e_534_7
  have hr := e_534_7
  simp only [e_534_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4936 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 534 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2502)) := by
  have c33969 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33970 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33971 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33972 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33973 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33974 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_534_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_8
  simp only [← c33969, ← c33970, ← c33971] at e_534_8
  have e_534_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_9
  simp only [← c33972, ← c33973, k2, ← c33974] at e_534_9
  have hr := e_534_9
  simp only [e_534_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4937 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 534 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2503)) := by
  have c33975 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33976 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33977 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33978 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33979 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33980 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_534_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_10
  simp only [← c33975, ← c33976, ← c33977] at e_534_10
  have e_534_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_11
  simp only [← c33978, ← c33979, k2, ← c33980] at e_534_11
  have hr := e_534_11
  simp only [e_534_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4938 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 534 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2504)) := by
  have c33981 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33982 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33983 := (publicBatchWrapper86_copies1061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c33984 := (publicBatchWrapper86_copies1062 a h).1
  have c33985 := (publicBatchWrapper86_copies1062 a h).2.1
  have c33986 := (publicBatchWrapper86_copies1062 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_534_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_12
  simp only [← c33981, ← c33982, ← c33983] at e_534_12
  have e_534_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_13
  simp only [← c33984, ← c33985, k2, ← c33986] at e_534_13
  have hr := e_534_13
  simp only [e_534_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4939 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 534 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2505)) := by
  have c33987 := (publicBatchWrapper86_copies1062 a h).2.2.2.1
  have c33988 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.1
  have c33989 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.1
  have c33990 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.1
  have c33991 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.1
  have c33992 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_534_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_14
  simp only [← c33987, ← c33988, ← c33989] at e_534_14
  have e_534_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_15
  simp only [← c33990, ← c33991, k2, ← c33992] at e_534_15
  have hr := e_534_15
  simp only [e_534_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4940 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 534 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2506)) := by
  have c33993 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.1
  have c33994 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.1
  have c33995 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c33996 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33997 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c33998 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_534_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_16
  simp only [← c33993, ← c33994, ← c33995] at e_534_16
  have e_534_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_17
  simp only [← c33996, ← c33997, k2, ← c33998] at e_534_17
  have hr := e_534_17
  simp only [e_534_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4941 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 534 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2507)) := by
  have c33999 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34000 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34001 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34002 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34003 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34004 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_534_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_18
  simp only [← c33999, ← c34000, ← c34001] at e_534_18
  have e_534_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row534 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_534_19
  simp only [← c34002, ← c34003, k2, ← c34004] at e_534_19
  have hr := e_534_19
  simp only [e_534_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4942 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 535 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2508)) := by
  have c34005 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34006 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34007 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34008 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34009 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34010 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_535_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_0
  simp only [← c34005, ← c34006, ← c34007] at e_535_0
  have e_535_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_1
  simp only [← c34008, ← c34009, k2, ← c34010] at e_535_1
  have hr := e_535_1
  simp only [e_535_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4943 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 535 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2509)) := by
  have c34011 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34012 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34013 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34014 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34015 := (publicBatchWrapper86_copies1062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34016 := (publicBatchWrapper86_copies1063 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_535_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_2
  simp only [← c34011, ← c34012, ← c34013] at e_535_2
  have e_535_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_3
  simp only [← c34014, ← c34015, k2, ← c34016] at e_535_3
  have hr := e_535_3
  simp only [e_535_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4944 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 535 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2510)) := by
  have c34017 := (publicBatchWrapper86_copies1063 a h).2.1
  have c34018 := (publicBatchWrapper86_copies1063 a h).2.2.1
  have c34019 := (publicBatchWrapper86_copies1063 a h).2.2.2.1
  have c34020 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.1
  have c34021 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.1
  have c34022 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_535_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_4
  simp only [← c34017, ← c34018, ← c34019] at e_535_4
  have e_535_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_5
  simp only [← c34020, ← c34021, k2, ← c34022] at e_535_5
  have hr := e_535_5
  simp only [e_535_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4945 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 535 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2511)) := by
  have c34023 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.1
  have c34024 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.1
  have c34025 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.1
  have c34026 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34027 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34028 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_535_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_6
  simp only [← c34023, ← c34024, ← c34025] at e_535_6
  have e_535_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_7
  simp only [← c34026, ← c34027, k2, ← c34028] at e_535_7
  have hr := e_535_7
  simp only [e_535_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4946 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 535 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2512)) := by
  have c34029 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34030 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34031 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34032 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34033 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34034 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_535_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_8
  simp only [← c34029, ← c34030, ← c34031] at e_535_8
  have e_535_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_9
  simp only [← c34032, ← c34033, k2, ← c34034] at e_535_9
  have hr := e_535_9
  simp only [e_535_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4947 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 535 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2513)) := by
  have c34035 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34036 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34037 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34038 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34039 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34040 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_535_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_10
  simp only [← c34035, ← c34036, ← c34037] at e_535_10
  have e_535_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_11
  simp only [← c34038, ← c34039, k2, ← c34040] at e_535_11
  have hr := e_535_11
  simp only [e_535_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4948 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 535 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2514)) := by
  have c34041 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34042 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34043 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34044 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34045 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34046 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_535_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_12
  simp only [← c34041, ← c34042, ← c34043] at e_535_12
  have e_535_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_13
  simp only [← c34044, ← c34045, k2, ← c34046] at e_535_13
  have hr := e_535_13
  simp only [e_535_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4949 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 535 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2515)) := by
  have c34047 := (publicBatchWrapper86_copies1063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34048 := (publicBatchWrapper86_copies1064 a h).1
  have c34049 := (publicBatchWrapper86_copies1064 a h).2.1
  have c34050 := (publicBatchWrapper86_copies1064 a h).2.2.1
  have c34051 := (publicBatchWrapper86_copies1064 a h).2.2.2.1
  have c34052 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_535_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_14
  simp only [← c34047, ← c34048, ← c34049] at e_535_14
  have e_535_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_15
  simp only [← c34050, ← c34051, k2, ← c34052] at e_535_15
  have hr := e_535_15
  simp only [e_535_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4950 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 535 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2516)) := by
  have c34053 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.1
  have c34054 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.1
  have c34055 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.1
  have c34056 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.1
  have c34057 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.1
  have c34058 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_535_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_16
  simp only [← c34053, ← c34054, ← c34055] at e_535_16
  have e_535_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_17
  simp only [← c34056, ← c34057, k2, ← c34058] at e_535_17
  have hr := e_535_17
  simp only [e_535_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4951 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 535 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2517)) := by
  have c34059 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34060 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34061 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34062 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34063 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34064 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_535_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_18
  simp only [← c34059, ← c34060, ← c34061] at e_535_18
  have e_535_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row535 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_535_19
  simp only [← c34062, ← c34063, k2, ← c34064] at e_535_19
  have hr := e_535_19
  simp only [e_535_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4952 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 536 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2518)) := by
  have c34065 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34066 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34067 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34068 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34069 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34070 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_536_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_0
  simp only [← c34065, ← c34066, ← c34067] at e_536_0
  have e_536_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_1
  simp only [← c34068, ← c34069, k2, ← c34070] at e_536_1
  have hr := e_536_1
  simp only [e_536_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 536 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2519)) := by
  have c34071 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34072 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34073 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34074 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34075 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34076 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_536_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_2
  simp only [← c34071, ← c34072, ← c34073] at e_536_2
  have e_536_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_3
  simp only [← c34074, ← c34075, k2, ← c34076] at e_536_3
  have hr := e_536_3
  simp only [e_536_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4954 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 536 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2520)) := by
  have c34077 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34078 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34079 := (publicBatchWrapper86_copies1064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34080 := (publicBatchWrapper86_copies1065 a h).1
  have c34081 := (publicBatchWrapper86_copies1065 a h).2.1
  have c34082 := (publicBatchWrapper86_copies1065 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_536_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_4
  simp only [← c34077, ← c34078, ← c34079] at e_536_4
  have e_536_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_5
  simp only [← c34080, ← c34081, k2, ← c34082] at e_536_5
  have hr := e_536_5
  simp only [e_536_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4955 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 536 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2521)) := by
  have c34083 := (publicBatchWrapper86_copies1065 a h).2.2.2.1
  have c34084 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.1
  have c34085 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.1
  have c34086 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.1
  have c34087 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.1
  have c34088 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_536_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_6
  simp only [← c34083, ← c34084, ← c34085] at e_536_6
  have e_536_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_7
  simp only [← c34086, ← c34087, k2, ← c34088] at e_536_7
  have hr := e_536_7
  simp only [e_536_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4956 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 536 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2522)) := by
  have c34089 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.1
  have c34090 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34091 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34092 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34093 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34094 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_536_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_8
  simp only [← c34089, ← c34090, ← c34091] at e_536_8
  have e_536_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_9
  simp only [← c34092, ← c34093, k2, ← c34094] at e_536_9
  have hr := e_536_9
  simp only [e_536_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4957 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 536 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2523)) := by
  have c34095 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34096 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34097 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34098 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34099 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34100 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_536_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_10
  simp only [← c34095, ← c34096, ← c34097] at e_536_10
  have e_536_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_11
  simp only [← c34098, ← c34099, k2, ← c34100] at e_536_11
  have hr := e_536_11
  simp only [e_536_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4958 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 536 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2524)) := by
  have c34101 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34102 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34103 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34104 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34105 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34106 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_536_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_12
  simp only [← c34101, ← c34102, ← c34103] at e_536_12
  have e_536_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_13
  simp only [← c34104, ← c34105, k2, ← c34106] at e_536_13
  have hr := e_536_13
  simp only [e_536_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4959 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 536 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2525)) := by
  have c34107 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34108 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34109 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34110 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34111 := (publicBatchWrapper86_copies1065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34112 := (publicBatchWrapper86_copies1066 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_536_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_14
  simp only [← c34107, ← c34108, ← c34109] at e_536_14
  have e_536_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_15
  simp only [← c34110, ← c34111, k2, ← c34112] at e_536_15
  have hr := e_536_15
  simp only [e_536_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4960 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 536 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2526)) := by
  have c34113 := (publicBatchWrapper86_copies1066 a h).2.1
  have c34114 := (publicBatchWrapper86_copies1066 a h).2.2.1
  have c34115 := (publicBatchWrapper86_copies1066 a h).2.2.2.1
  have c34116 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.1
  have c34117 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.1
  have c34118 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_536_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_16
  simp only [← c34113, ← c34114, ← c34115] at e_536_16
  have e_536_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_17
  simp only [← c34116, ← c34117, k2, ← c34118] at e_536_17
  have hr := e_536_17
  simp only [e_536_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4961 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 536 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2527)) := by
  have c34119 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.1
  have c34120 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.1
  have c34121 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.1
  have c34122 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34123 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34124 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_536_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_18
  simp only [← c34119, ← c34120, ← c34121] at e_536_18
  have e_536_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row536 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_536_19
  simp only [← c34122, ← c34123, k2, ← c34124] at e_536_19
  have hr := e_536_19
  simp only [e_536_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4962 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 537 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2528)) := by
  have c34125 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34126 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34127 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34128 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34129 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34130 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_537_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_0
  simp only [← c34125, ← c34126, ← c34127] at e_537_0
  have e_537_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_1
  simp only [← c34128, ← c34129, k2, ← c34130] at e_537_1
  have hr := e_537_1
  simp only [e_537_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4963 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 537 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2529)) := by
  have c34131 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34132 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34133 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34134 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34135 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34136 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_537_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_2
  simp only [← c34131, ← c34132, ← c34133] at e_537_2
  have e_537_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_3
  simp only [← c34134, ← c34135, k2, ← c34136] at e_537_3
  have hr := e_537_3
  simp only [e_537_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4964 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 537 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2530)) := by
  have c34137 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34138 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34139 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34140 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34141 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34142 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_537_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_4
  simp only [← c34137, ← c34138, ← c34139] at e_537_4
  have e_537_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_5
  simp only [← c34140, ← c34141, k2, ← c34142] at e_537_5
  have hr := e_537_5
  simp only [e_537_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4965 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 537 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2531)) := by
  have c34143 := (publicBatchWrapper86_copies1066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34144 := (publicBatchWrapper86_copies1067 a h).1
  have c34145 := (publicBatchWrapper86_copies1067 a h).2.1
  have c34146 := (publicBatchWrapper86_copies1067 a h).2.2.1
  have c34147 := (publicBatchWrapper86_copies1067 a h).2.2.2.1
  have c34148 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_537_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_6
  simp only [← c34143, ← c34144, ← c34145] at e_537_6
  have e_537_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_7
  simp only [← c34146, ← c34147, k2, ← c34148] at e_537_7
  have hr := e_537_7
  simp only [e_537_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4966 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 537 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2532)) := by
  have c34149 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.1
  have c34150 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.1
  have c34151 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.1
  have c34152 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.1
  have c34153 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.1
  have c34154 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_537_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_8
  simp only [← c34149, ← c34150, ← c34151] at e_537_8
  have e_537_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_9
  simp only [← c34152, ← c34153, k2, ← c34154] at e_537_9
  have hr := e_537_9
  simp only [e_537_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4967 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 537 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2533)) := by
  have c34155 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34156 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34157 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34158 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34159 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34160 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_537_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_10
  simp only [← c34155, ← c34156, ← c34157] at e_537_10
  have e_537_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_11
  simp only [← c34158, ← c34159, k2, ← c34160] at e_537_11
  have hr := e_537_11
  simp only [e_537_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4968 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 537 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2534)) := by
  have c34161 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34162 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34163 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34164 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34165 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34166 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_537_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_12
  simp only [← c34161, ← c34162, ← c34163] at e_537_12
  have e_537_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_13
  simp only [← c34164, ← c34165, k2, ← c34166] at e_537_13
  have hr := e_537_13
  simp only [e_537_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4969 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 537 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2535)) := by
  have c34167 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34168 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34169 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34170 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34171 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34172 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_537_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_14
  simp only [← c34167, ← c34168, ← c34169] at e_537_14
  have e_537_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_15
  simp only [← c34170, ← c34171, k2, ← c34172] at e_537_15
  have hr := e_537_15
  simp only [e_537_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4970 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 537 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2536)) := by
  have c34173 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34174 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34175 := (publicBatchWrapper86_copies1067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34176 := (publicBatchWrapper86_copies1068 a h).1
  have c34177 := (publicBatchWrapper86_copies1068 a h).2.1
  have c34178 := (publicBatchWrapper86_copies1068 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_537_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_16
  simp only [← c34173, ← c34174, ← c34175] at e_537_16
  have e_537_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_17
  simp only [← c34176, ← c34177, k2, ← c34178] at e_537_17
  have hr := e_537_17
  simp only [e_537_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4971 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 537 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2537)) := by
  have c34179 := (publicBatchWrapper86_copies1068 a h).2.2.2.1
  have c34180 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.1
  have c34181 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.1
  have c34182 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.1
  have c34183 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.1
  have c34184 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_537_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_18
  simp only [← c34179, ← c34180, ← c34181] at e_537_18
  have e_537_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row537 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_537_19
  simp only [← c34182, ← c34183, k2, ← c34184] at e_537_19
  have hr := e_537_19
  simp only [e_537_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4972 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 538 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2538)) := by
  have c34185 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.1
  have c34186 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34187 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34188 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34189 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34190 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_538_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_0
  simp only [← c34185, ← c34186, ← c34187] at e_538_0
  have e_538_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_1
  simp only [← c34188, ← c34189, k2, ← c34190] at e_538_1
  have hr := e_538_1
  simp only [e_538_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4973 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 538 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2539)) := by
  have c34191 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34192 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34193 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34194 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34195 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34196 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_538_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_2
  simp only [← c34191, ← c34192, ← c34193] at e_538_2
  have e_538_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_3
  simp only [← c34194, ← c34195, k2, ← c34196] at e_538_3
  have hr := e_538_3
  simp only [e_538_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 538 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2540)) := by
  have c34197 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34198 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34199 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34200 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34201 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34202 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_538_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_4
  simp only [← c34197, ← c34198, ← c34199] at e_538_4
  have e_538_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_5
  simp only [← c34200, ← c34201, k2, ← c34202] at e_538_5
  have hr := e_538_5
  simp only [e_538_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 538 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2541)) := by
  have c34203 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34204 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34205 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34206 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34207 := (publicBatchWrapper86_copies1068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34208 := (publicBatchWrapper86_copies1069 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_538_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_6
  simp only [← c34203, ← c34204, ← c34205] at e_538_6
  have e_538_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_7
  simp only [← c34206, ← c34207, k2, ← c34208] at e_538_7
  have hr := e_538_7
  simp only [e_538_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 538 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2542)) := by
  have c34209 := (publicBatchWrapper86_copies1069 a h).2.1
  have c34210 := (publicBatchWrapper86_copies1069 a h).2.2.1
  have c34211 := (publicBatchWrapper86_copies1069 a h).2.2.2.1
  have c34212 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.1
  have c34213 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.1
  have c34214 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_538_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_8
  simp only [← c34209, ← c34210, ← c34211] at e_538_8
  have e_538_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_9
  simp only [← c34212, ← c34213, k2, ← c34214] at e_538_9
  have hr := e_538_9
  simp only [e_538_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 538 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2543)) := by
  have c34215 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.1
  have c34216 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.1
  have c34217 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.1
  have c34218 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34219 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34220 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_538_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_10
  simp only [← c34215, ← c34216, ← c34217] at e_538_10
  have e_538_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_11
  simp only [← c34218, ← c34219, k2, ← c34220] at e_538_11
  have hr := e_538_11
  simp only [e_538_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 538 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2544)) := by
  have c34221 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34222 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34223 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34224 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34225 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34226 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_538_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_12
  simp only [← c34221, ← c34222, ← c34223] at e_538_12
  have e_538_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_13
  simp only [← c34224, ← c34225, k2, ← c34226] at e_538_13
  have hr := e_538_13
  simp only [e_538_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 538 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2545)) := by
  have c34227 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34228 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34229 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34230 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34231 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34232 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_538_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_14
  simp only [← c34227, ← c34228, ← c34229] at e_538_14
  have e_538_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_15
  simp only [← c34230, ← c34231, k2, ← c34232] at e_538_15
  have hr := e_538_15
  simp only [e_538_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 538 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2546)) := by
  have c34233 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34234 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34235 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34236 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34237 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34238 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_538_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_16
  simp only [← c34233, ← c34234, ← c34235] at e_538_16
  have e_538_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_17
  simp only [← c34236, ← c34237, k2, ← c34238] at e_538_17
  have hr := e_538_17
  simp only [e_538_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 538 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2547)) := by
  have c34239 := (publicBatchWrapper86_copies1069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34240 := (publicBatchWrapper86_copies1070 a h).1
  have c34241 := (publicBatchWrapper86_copies1070 a h).2.1
  have c34242 := (publicBatchWrapper86_copies1070 a h).2.2.1
  have c34243 := (publicBatchWrapper86_copies1070 a h).2.2.2.1
  have c34244 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_538_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_18
  simp only [← c34239, ← c34240, ← c34241] at e_538_18
  have e_538_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row538 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_538_19
  simp only [← c34242, ← c34243, k2, ← c34244] at e_538_19
  have hr := e_538_19
  simp only [e_538_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 539 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2548)) := by
  have c34245 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.1
  have c34246 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.1
  have c34247 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.1
  have c34248 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.1
  have c34249 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.1
  have c34250 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_539_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_0
  simp only [← c34245, ← c34246, ← c34247] at e_539_0
  have e_539_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_1
  simp only [← c34248, ← c34249, k2, ← c34250] at e_539_1
  have hr := e_539_1
  simp only [e_539_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 539 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2549)) := by
  have c34251 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34252 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34253 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34254 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34255 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34256 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_539_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_2
  simp only [← c34251, ← c34252, ← c34253] at e_539_2
  have e_539_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_3
  simp only [← c34254, ← c34255, k2, ← c34256] at e_539_3
  have hr := e_539_3
  simp only [e_539_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 539 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2550)) := by
  have c34257 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34258 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34259 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34260 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34261 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34262 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_539_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_4
  simp only [← c34257, ← c34258, ← c34259] at e_539_4
  have e_539_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_5
  simp only [← c34260, ← c34261, k2, ← c34262] at e_539_5
  have hr := e_539_5
  simp only [e_539_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 539 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2551)) := by
  have c34263 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34264 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34265 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34266 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34267 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34268 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_539_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_6
  simp only [← c34263, ← c34264, ← c34265] at e_539_6
  have e_539_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_7
  simp only [← c34266, ← c34267, k2, ← c34268] at e_539_7
  have hr := e_539_7
  simp only [e_539_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 539 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2552)) := by
  have c34269 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34270 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34271 := (publicBatchWrapper86_copies1070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34272 := (publicBatchWrapper86_copies1071 a h).1
  have c34273 := (publicBatchWrapper86_copies1071 a h).2.1
  have c34274 := (publicBatchWrapper86_copies1071 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_539_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_8
  simp only [← c34269, ← c34270, ← c34271] at e_539_8
  have e_539_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_9
  simp only [← c34272, ← c34273, k2, ← c34274] at e_539_9
  have hr := e_539_9
  simp only [e_539_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 539 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2553)) := by
  have c34275 := (publicBatchWrapper86_copies1071 a h).2.2.2.1
  have c34276 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.1
  have c34277 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.1
  have c34278 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.1
  have c34279 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.1
  have c34280 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_539_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_10
  simp only [← c34275, ← c34276, ← c34277] at e_539_10
  have e_539_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_11
  simp only [← c34278, ← c34279, k2, ← c34280] at e_539_11
  have hr := e_539_11
  simp only [e_539_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 539 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2554)) := by
  have c34281 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.1
  have c34282 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34283 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34284 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34285 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34286 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_539_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_12
  simp only [← c34281, ← c34282, ← c34283] at e_539_12
  have e_539_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_13
  simp only [← c34284, ← c34285, k2, ← c34286] at e_539_13
  have hr := e_539_13
  simp only [e_539_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 539 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2555)) := by
  have c34287 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34288 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34289 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34290 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34291 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34292 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_539_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_14
  simp only [← c34287, ← c34288, ← c34289] at e_539_14
  have e_539_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_15
  simp only [← c34290, ← c34291, k2, ← c34292] at e_539_15
  have hr := e_539_15
  simp only [e_539_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 539 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2556)) := by
  have c34293 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34294 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34295 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34296 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34297 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34298 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_539_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_16
  simp only [← c34293, ← c34294, ← c34295] at e_539_16
  have e_539_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_17
  simp only [← c34296, ← c34297, k2, ← c34298] at e_539_17
  have hr := e_539_17
  simp only [e_539_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 539 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2557)) := by
  have c34299 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34300 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34301 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34302 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34303 := (publicBatchWrapper86_copies1071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34304 := (publicBatchWrapper86_copies1072 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_539_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_18
  simp only [← c34299, ← c34300, ← c34301] at e_539_18
  have e_539_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row539 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_539_19
  simp only [← c34302, ← c34303, k2, ← c34304] at e_539_19
  have hr := e_539_19
  simp only [e_539_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 540 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2558)) := by
  have c34305 := (publicBatchWrapper86_copies1072 a h).2.1
  have c34306 := (publicBatchWrapper86_copies1072 a h).2.2.1
  have c34307 := (publicBatchWrapper86_copies1072 a h).2.2.2.1
  have c34308 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.1
  have c34309 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.1
  have c34310 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_540_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_0
  simp only [← c34305, ← c34306, ← c34307] at e_540_0
  have e_540_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_1
  simp only [← c34308, ← c34309, k2, ← c34310] at e_540_1
  have hr := e_540_1
  simp only [e_540_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 540 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2559)) := by
  have c34311 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.1
  have c34312 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.1
  have c34313 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.1
  have c34314 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34315 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34316 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_540_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_2
  simp only [← c34311, ← c34312, ← c34313] at e_540_2
  have e_540_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_3
  simp only [← c34314, ← c34315, k2, ← c34316] at e_540_3
  have hr := e_540_3
  simp only [e_540_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 540 23) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2560)) := by
  have c34317 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34318 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34319 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34320 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34321 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34322 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_540_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_4
  simp only [← c34317, ← c34318, ← c34319] at e_540_4
  have e_540_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_5
  simp only [← c34320, ← c34321, k2, ← c34322] at e_540_5
  have hr := e_540_5
  simp only [e_540_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 540 31) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2561)) := by
  have c34323 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34324 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34325 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34326 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34327 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34328 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_540_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_6
  simp only [← c34323, ← c34324, ← c34325] at e_540_6
  have e_540_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_7
  simp only [← c34326, ← c34327, k2, ← c34328] at e_540_7
  have hr := e_540_7
  simp only [e_540_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 540 39) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2562)) := by
  have c34329 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34330 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34331 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34332 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34333 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34334 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_540_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_8
  simp only [← c34329, ← c34330, ← c34331] at e_540_8
  have e_540_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_9
  simp only [← c34332, ← c34333, k2, ← c34334] at e_540_9
  have hr := e_540_9
  simp only [e_540_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 540 47) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2563)) := by
  have c34335 := (publicBatchWrapper86_copies1072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34336 := (publicBatchWrapper86_copies1073 a h).1
  have c34337 := (publicBatchWrapper86_copies1073 a h).2.1
  have c34338 := (publicBatchWrapper86_copies1073 a h).2.2.1
  have c34339 := (publicBatchWrapper86_copies1073 a h).2.2.2.1
  have c34340 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_540_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_10
  simp only [← c34335, ← c34336, ← c34337] at e_540_10
  have e_540_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_11
  simp only [← c34338, ← c34339, k2, ← c34340] at e_540_11
  have hr := e_540_11
  simp only [e_540_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 540 55) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2564)) := by
  have c34341 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.1
  have c34342 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.1
  have c34343 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.1
  have c34344 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.1
  have c34345 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.1
  have c34346 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_540_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_12
  simp only [← c34341, ← c34342, ← c34343] at e_540_12
  have e_540_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_13
  simp only [← c34344, ← c34345, k2, ← c34346] at e_540_13
  have hr := e_540_13
  simp only [e_540_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f4999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 540 63) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2565)) := by
  have c34347 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34348 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34349 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34350 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34351 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34352 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_540_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_14
  simp only [← c34347, ← c34348, ← c34349] at e_540_14
  have e_540_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_15
  simp only [← c34350, ← c34351, k2, ← c34352] at e_540_15
  have hr := e_540_15
  simp only [e_540_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 540 71) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2566)) := by
  have c34353 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34354 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34355 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34356 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34357 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34358 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_540_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_16
  simp only [← c34353, ← c34354, ← c34355] at e_540_16
  have e_540_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_17
  simp only [← c34356, ← c34357, k2, ← c34358] at e_540_17
  have hr := e_540_17
  simp only [e_540_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 540 79) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2567)) := by
  have c34359 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34360 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34361 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34362 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34363 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34364 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_540_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_18
  simp only [← c34359, ← c34360, ← c34361] at e_540_18
  have e_540_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row540 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_540_19
  simp only [← c34362, ← c34363, k2, ← c34364] at e_540_19
  have hr := e_540_19
  simp only [e_540_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 541 7) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2568)) := by
  have c34365 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34366 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34367 := (publicBatchWrapper86_copies1073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34368 := (publicBatchWrapper86_copies1074 a h).1
  have c34369 := (publicBatchWrapper86_copies1074 a h).2.1
  have c34370 := (publicBatchWrapper86_copies1074 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_541_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_0
  simp only [← c34365, ← c34366, ← c34367] at e_541_0
  have e_541_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_1
  simp only [← c34368, ← c34369, k2, ← c34370] at e_541_1
  have hr := e_541_1
  simp only [e_541_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 541 15) = bselect (a (.wire 13 75)) (a (.virt 200)) (a (.virt 2569)) := by
  have c34371 := (publicBatchWrapper86_copies1074 a h).2.2.2.1
  have c34372 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.1
  have c34373 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.1
  have c34374 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.1
  have c34375 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.1
  have c34376 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_541_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_2
  simp only [← c34371, ← c34372, ← c34373] at e_541_2
  have e_541_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_3
  simp only [← c34374, ← c34375, k2, ← c34376] at e_541_3
  have hr := e_541_3
  simp only [e_541_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 541 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2610)) := by
  have c34377 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.1
  have c34378 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34379 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34380 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34381 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34382 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_541_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_4
  simp only [← c34377, ← c34378, ← c34379] at e_541_4
  have e_541_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_5
  simp only [← c34380, ← c34381, k2, ← c34382] at e_541_5
  have hr := e_541_5
  simp only [e_541_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 541 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2611)) := by
  have c34383 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34384 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34385 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34386 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34387 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34388 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_541_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_6
  simp only [← c34383, ← c34384, ← c34385] at e_541_6
  have e_541_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_7
  simp only [← c34386, ← c34387, k2, ← c34388] at e_541_7
  have hr := e_541_7
  simp only [e_541_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 541 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2612)) := by
  have c34389 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34390 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34391 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34392 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34393 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34394 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_541_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_8
  simp only [← c34389, ← c34390, ← c34391] at e_541_8
  have e_541_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_9
  simp only [← c34392, ← c34393, k2, ← c34394] at e_541_9
  have hr := e_541_9
  simp only [e_541_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 541 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2613)) := by
  have c34395 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34396 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34397 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34398 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34399 := (publicBatchWrapper86_copies1074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34400 := (publicBatchWrapper86_copies1075 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_541_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_10
  simp only [← c34395, ← c34396, ← c34397] at e_541_10
  have e_541_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_11
  simp only [← c34398, ← c34399, k2, ← c34400] at e_541_11
  have hr := e_541_11
  simp only [e_541_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 541 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2614)) := by
  have c34401 := (publicBatchWrapper86_copies1075 a h).2.1
  have c34402 := (publicBatchWrapper86_copies1075 a h).2.2.1
  have c34403 := (publicBatchWrapper86_copies1075 a h).2.2.2.1
  have c34404 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.1
  have c34405 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.1
  have c34406 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_541_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_12
  simp only [← c34401, ← c34402, ← c34403] at e_541_12
  have e_541_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_13
  simp only [← c34404, ← c34405, k2, ← c34406] at e_541_13
  have hr := e_541_13
  simp only [e_541_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 541 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2615)) := by
  have c34407 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.1
  have c34408 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.1
  have c34409 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.1
  have c34410 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34411 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34412 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_541_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_14
  simp only [← c34407, ← c34408, ← c34409] at e_541_14
  have e_541_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_15
  simp only [← c34410, ← c34411, k2, ← c34412] at e_541_15
  have hr := e_541_15
  simp only [e_541_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 541 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2616)) := by
  have c34413 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34414 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34415 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34416 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34417 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34418 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_541_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_16
  simp only [← c34413, ← c34414, ← c34415] at e_541_16
  have e_541_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_17
  simp only [← c34416, ← c34417, k2, ← c34418] at e_541_17
  have hr := e_541_17
  simp only [e_541_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 541 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2617)) := by
  have c34419 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34420 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34421 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34422 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34423 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34424 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_541_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_18
  simp only [← c34419, ← c34420, ← c34421] at e_541_18
  have e_541_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row541 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_541_19
  simp only [← c34422, ← c34423, k2, ← c34424] at e_541_19
  have hr := e_541_19
  simp only [e_541_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 542 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2618)) := by
  have c34425 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34426 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34427 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34428 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34429 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34430 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_542_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_0
  simp only [← c34425, ← c34426, ← c34427] at e_542_0
  have e_542_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_1
  simp only [← c34428, ← c34429, k2, ← c34430] at e_542_1
  have hr := e_542_1
  simp only [e_542_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 542 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2619)) := by
  have c34431 := (publicBatchWrapper86_copies1075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34432 := (publicBatchWrapper86_copies1076 a h).1
  have c34433 := (publicBatchWrapper86_copies1076 a h).2.1
  have c34434 := (publicBatchWrapper86_copies1076 a h).2.2.1
  have c34435 := (publicBatchWrapper86_copies1076 a h).2.2.2.1
  have c34436 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_542_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_2
  simp only [← c34431, ← c34432, ← c34433] at e_542_2
  have e_542_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_3
  simp only [← c34434, ← c34435, k2, ← c34436] at e_542_3
  have hr := e_542_3
  simp only [e_542_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 542 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2620)) := by
  have c34437 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.1
  have c34438 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.1
  have c34439 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.1
  have c34440 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.1
  have c34441 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.1
  have c34442 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_542_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_4
  simp only [← c34437, ← c34438, ← c34439] at e_542_4
  have e_542_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_5
  simp only [← c34440, ← c34441, k2, ← c34442] at e_542_5
  have hr := e_542_5
  simp only [e_542_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 542 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2621)) := by
  have c34443 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34444 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34445 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34446 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34447 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34448 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_542_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_6
  simp only [← c34443, ← c34444, ← c34445] at e_542_6
  have e_542_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_7
  simp only [← c34446, ← c34447, k2, ← c34448] at e_542_7
  have hr := e_542_7
  simp only [e_542_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 542 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2622)) := by
  have c34449 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34450 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34451 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34452 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34453 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34454 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_542_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_8
  simp only [← c34449, ← c34450, ← c34451] at e_542_8
  have e_542_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_9
  simp only [← c34452, ← c34453, k2, ← c34454] at e_542_9
  have hr := e_542_9
  simp only [e_542_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 542 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2623)) := by
  have c34455 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34456 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34457 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34458 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34459 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34460 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_542_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_10
  simp only [← c34455, ← c34456, ← c34457] at e_542_10
  have e_542_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_11
  simp only [← c34458, ← c34459, k2, ← c34460] at e_542_11
  have hr := e_542_11
  simp only [e_542_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 542 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2624)) := by
  have c34461 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34462 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34463 := (publicBatchWrapper86_copies1076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34464 := (publicBatchWrapper86_copies1077 a h).1
  have c34465 := (publicBatchWrapper86_copies1077 a h).2.1
  have c34466 := (publicBatchWrapper86_copies1077 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_542_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_12
  simp only [← c34461, ← c34462, ← c34463] at e_542_12
  have e_542_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_13
  simp only [← c34464, ← c34465, k2, ← c34466] at e_542_13
  have hr := e_542_13
  simp only [e_542_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 542 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2625)) := by
  have c34467 := (publicBatchWrapper86_copies1077 a h).2.2.2.1
  have c34468 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.1
  have c34469 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.1
  have c34470 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.1
  have c34471 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.1
  have c34472 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_542_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_14
  simp only [← c34467, ← c34468, ← c34469] at e_542_14
  have e_542_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_15
  simp only [← c34470, ← c34471, k2, ← c34472] at e_542_15
  have hr := e_542_15
  simp only [e_542_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 542 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2626)) := by
  have c34473 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.1
  have c34474 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34475 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34476 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34477 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34478 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_542_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_16
  simp only [← c34473, ← c34474, ← c34475] at e_542_16
  have e_542_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_17
  simp only [← c34476, ← c34477, k2, ← c34478] at e_542_17
  have hr := e_542_17
  simp only [e_542_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 542 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2627)) := by
  have c34479 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34480 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34481 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34482 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34483 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34484 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_542_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_18
  simp only [← c34479, ← c34480, ← c34481] at e_542_18
  have e_542_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row542 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_542_19
  simp only [← c34482, ← c34483, k2, ← c34484] at e_542_19
  have hr := e_542_19
  simp only [e_542_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 543 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2628)) := by
  have c34485 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34486 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34487 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34488 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34489 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34490 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_543_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_0
  simp only [← c34485, ← c34486, ← c34487] at e_543_0
  have e_543_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_1
  simp only [← c34488, ← c34489, k2, ← c34490] at e_543_1
  have hr := e_543_1
  simp only [e_543_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 543 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2629)) := by
  have c34491 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34492 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34493 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34494 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34495 := (publicBatchWrapper86_copies1077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34496 := (publicBatchWrapper86_copies1078 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_543_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_2
  simp only [← c34491, ← c34492, ← c34493] at e_543_2
  have e_543_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_3
  simp only [← c34494, ← c34495, k2, ← c34496] at e_543_3
  have hr := e_543_3
  simp only [e_543_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 543 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2630)) := by
  have c34497 := (publicBatchWrapper86_copies1078 a h).2.1
  have c34498 := (publicBatchWrapper86_copies1078 a h).2.2.1
  have c34499 := (publicBatchWrapper86_copies1078 a h).2.2.2.1
  have c34500 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.1
  have c34501 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.1
  have c34502 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_543_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_4
  simp only [← c34497, ← c34498, ← c34499] at e_543_4
  have e_543_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_5
  simp only [← c34500, ← c34501, k2, ← c34502] at e_543_5
  have hr := e_543_5
  simp only [e_543_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 543 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2631)) := by
  have c34503 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.1
  have c34504 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.1
  have c34505 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.1
  have c34506 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34507 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34508 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_543_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_6
  simp only [← c34503, ← c34504, ← c34505] at e_543_6
  have e_543_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_7
  simp only [← c34506, ← c34507, k2, ← c34508] at e_543_7
  have hr := e_543_7
  simp only [e_543_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 543 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2632)) := by
  have c34509 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34510 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34511 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34512 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34513 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34514 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_543_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_8
  simp only [← c34509, ← c34510, ← c34511] at e_543_8
  have e_543_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_9
  simp only [← c34512, ← c34513, k2, ← c34514] at e_543_9
  have hr := e_543_9
  simp only [e_543_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 543 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2633)) := by
  have c34515 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34516 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34517 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34518 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34519 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34520 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_543_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_10
  simp only [← c34515, ← c34516, ← c34517] at e_543_10
  have e_543_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_11
  simp only [← c34518, ← c34519, k2, ← c34520] at e_543_11
  have hr := e_543_11
  simp only [e_543_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 543 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2634)) := by
  have c34521 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34522 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34523 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34524 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34525 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34526 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_543_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_12
  simp only [← c34521, ← c34522, ← c34523] at e_543_12
  have e_543_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_13
  simp only [← c34524, ← c34525, k2, ← c34526] at e_543_13
  have hr := e_543_13
  simp only [e_543_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 543 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2635)) := by
  have c34527 := (publicBatchWrapper86_copies1078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34528 := (publicBatchWrapper86_copies1079 a h).1
  have c34529 := (publicBatchWrapper86_copies1079 a h).2.1
  have c34530 := (publicBatchWrapper86_copies1079 a h).2.2.1
  have c34531 := (publicBatchWrapper86_copies1079 a h).2.2.2.1
  have c34532 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_543_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_14
  simp only [← c34527, ← c34528, ← c34529] at e_543_14
  have e_543_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_15
  simp only [← c34530, ← c34531, k2, ← c34532] at e_543_15
  have hr := e_543_15
  simp only [e_543_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 543 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2636)) := by
  have c34533 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.1
  have c34534 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.1
  have c34535 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.1
  have c34536 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.1
  have c34537 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.1
  have c34538 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_543_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_16
  simp only [← c34533, ← c34534, ← c34535] at e_543_16
  have e_543_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_17
  simp only [← c34536, ← c34537, k2, ← c34538] at e_543_17
  have hr := e_543_17
  simp only [e_543_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 543 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2637)) := by
  have c34539 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34540 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34541 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34542 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34543 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34544 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_543_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_18
  simp only [← c34539, ← c34540, ← c34541] at e_543_18
  have e_543_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row543 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_543_19
  simp only [← c34542, ← c34543, k2, ← c34544] at e_543_19
  have hr := e_543_19
  simp only [e_543_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 544 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2638)) := by
  have c34545 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34546 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34547 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34548 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34549 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34550 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_544_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_0
  simp only [← c34545, ← c34546, ← c34547] at e_544_0
  have e_544_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_1
  simp only [← c34548, ← c34549, k2, ← c34550] at e_544_1
  have hr := e_544_1
  simp only [e_544_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 544 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2639)) := by
  have c34551 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34552 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34553 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34554 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34555 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34556 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_544_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_2
  simp only [← c34551, ← c34552, ← c34553] at e_544_2
  have e_544_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_3
  simp only [← c34554, ← c34555, k2, ← c34556] at e_544_3
  have hr := e_544_3
  simp only [e_544_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 544 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2640)) := by
  have c34557 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34558 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34559 := (publicBatchWrapper86_copies1079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34560 := (publicBatchWrapper86_copies1080 a h).1
  have c34561 := (publicBatchWrapper86_copies1080 a h).2.1
  have c34562 := (publicBatchWrapper86_copies1080 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_544_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_4
  simp only [← c34557, ← c34558, ← c34559] at e_544_4
  have e_544_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_5
  simp only [← c34560, ← c34561, k2, ← c34562] at e_544_5
  have hr := e_544_5
  simp only [e_544_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 544 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2641)) := by
  have c34563 := (publicBatchWrapper86_copies1080 a h).2.2.2.1
  have c34564 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.1
  have c34565 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.1
  have c34566 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.1
  have c34567 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.1
  have c34568 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_544_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_6
  simp only [← c34563, ← c34564, ← c34565] at e_544_6
  have e_544_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_7
  simp only [← c34566, ← c34567, k2, ← c34568] at e_544_7
  have hr := e_544_7
  simp only [e_544_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 544 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2642)) := by
  have c34569 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.1
  have c34570 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34571 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34572 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34573 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34574 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_544_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_8
  simp only [← c34569, ← c34570, ← c34571] at e_544_8
  have e_544_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_9
  simp only [← c34572, ← c34573, k2, ← c34574] at e_544_9
  have hr := e_544_9
  simp only [e_544_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 544 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2643)) := by
  have c34575 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34576 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34577 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34578 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34579 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34580 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_544_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_10
  simp only [← c34575, ← c34576, ← c34577] at e_544_10
  have e_544_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_11
  simp only [← c34578, ← c34579, k2, ← c34580] at e_544_11
  have hr := e_544_11
  simp only [e_544_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 544 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2644)) := by
  have c34581 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34582 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34583 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34584 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34585 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34586 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_544_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_12
  simp only [← c34581, ← c34582, ← c34583] at e_544_12
  have e_544_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_13
  simp only [← c34584, ← c34585, k2, ← c34586] at e_544_13
  have hr := e_544_13
  simp only [e_544_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 544 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2645)) := by
  have c34587 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34588 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34589 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34590 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34591 := (publicBatchWrapper86_copies1080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34592 := (publicBatchWrapper86_copies1081 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_544_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_14
  simp only [← c34587, ← c34588, ← c34589] at e_544_14
  have e_544_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_15
  simp only [← c34590, ← c34591, k2, ← c34592] at e_544_15
  have hr := e_544_15
  simp only [e_544_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 544 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2646)) := by
  have c34593 := (publicBatchWrapper86_copies1081 a h).2.1
  have c34594 := (publicBatchWrapper86_copies1081 a h).2.2.1
  have c34595 := (publicBatchWrapper86_copies1081 a h).2.2.2.1
  have c34596 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.1
  have c34597 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.1
  have c34598 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_544_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_16
  simp only [← c34593, ← c34594, ← c34595] at e_544_16
  have e_544_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_17
  simp only [← c34596, ← c34597, k2, ← c34598] at e_544_17
  have hr := e_544_17
  simp only [e_544_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 544 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2647)) := by
  have c34599 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.1
  have c34600 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.1
  have c34601 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.1
  have c34602 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34603 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34604 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_544_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_18
  simp only [← c34599, ← c34600, ← c34601] at e_544_18
  have e_544_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row544 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_544_19
  simp only [← c34602, ← c34603, k2, ← c34604] at e_544_19
  have hr := e_544_19
  simp only [e_544_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 545 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2648)) := by
  have c34605 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34606 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34607 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34608 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34609 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34610 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_545_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_0
  simp only [← c34605, ← c34606, ← c34607] at e_545_0
  have e_545_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_1
  simp only [← c34608, ← c34609, k2, ← c34610] at e_545_1
  have hr := e_545_1
  simp only [e_545_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 545 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2649)) := by
  have c34611 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34612 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34613 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34614 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34615 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34616 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_545_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_2
  simp only [← c34611, ← c34612, ← c34613] at e_545_2
  have e_545_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_3
  simp only [← c34614, ← c34615, k2, ← c34616] at e_545_3
  have hr := e_545_3
  simp only [e_545_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 545 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2650)) := by
  have c34617 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34618 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34619 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34620 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34621 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34622 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_545_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_4
  simp only [← c34617, ← c34618, ← c34619] at e_545_4
  have e_545_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_5
  simp only [← c34620, ← c34621, k2, ← c34622] at e_545_5
  have hr := e_545_5
  simp only [e_545_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 545 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2651)) := by
  have c34623 := (publicBatchWrapper86_copies1081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34624 := (publicBatchWrapper86_copies1082 a h).1
  have c34625 := (publicBatchWrapper86_copies1082 a h).2.1
  have c34626 := (publicBatchWrapper86_copies1082 a h).2.2.1
  have c34627 := (publicBatchWrapper86_copies1082 a h).2.2.2.1
  have c34628 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_545_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_6
  simp only [← c34623, ← c34624, ← c34625] at e_545_6
  have e_545_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_7
  simp only [← c34626, ← c34627, k2, ← c34628] at e_545_7
  have hr := e_545_7
  simp only [e_545_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 545 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2652)) := by
  have c34629 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.1
  have c34630 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.1
  have c34631 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.1
  have c34632 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.1
  have c34633 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.1
  have c34634 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_545_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_8
  simp only [← c34629, ← c34630, ← c34631] at e_545_8
  have e_545_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_9
  simp only [← c34632, ← c34633, k2, ← c34634] at e_545_9
  have hr := e_545_9
  simp only [e_545_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 545 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2653)) := by
  have c34635 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34636 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34637 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34638 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34639 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34640 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_545_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_10
  simp only [← c34635, ← c34636, ← c34637] at e_545_10
  have e_545_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_11
  simp only [← c34638, ← c34639, k2, ← c34640] at e_545_11
  have hr := e_545_11
  simp only [e_545_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 545 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2654)) := by
  have c34641 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34642 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34643 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34644 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34645 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34646 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_545_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_12
  simp only [← c34641, ← c34642, ← c34643] at e_545_12
  have e_545_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_13
  simp only [← c34644, ← c34645, k2, ← c34646] at e_545_13
  have hr := e_545_13
  simp only [e_545_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 545 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2655)) := by
  have c34647 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34648 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34649 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34650 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34651 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34652 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_545_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_14
  simp only [← c34647, ← c34648, ← c34649] at e_545_14
  have e_545_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_15
  simp only [← c34650, ← c34651, k2, ← c34652] at e_545_15
  have hr := e_545_15
  simp only [e_545_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 545 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2656)) := by
  have c34653 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34654 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34655 := (publicBatchWrapper86_copies1082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34656 := (publicBatchWrapper86_copies1083 a h).1
  have c34657 := (publicBatchWrapper86_copies1083 a h).2.1
  have c34658 := (publicBatchWrapper86_copies1083 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_545_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_16
  simp only [← c34653, ← c34654, ← c34655] at e_545_16
  have e_545_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_17
  simp only [← c34656, ← c34657, k2, ← c34658] at e_545_17
  have hr := e_545_17
  simp only [e_545_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 545 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2657)) := by
  have c34659 := (publicBatchWrapper86_copies1083 a h).2.2.2.1
  have c34660 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.1
  have c34661 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.1
  have c34662 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.1
  have c34663 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.1
  have c34664 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_545_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_18
  simp only [← c34659, ← c34660, ← c34661] at e_545_18
  have e_545_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row545 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_545_19
  simp only [← c34662, ← c34663, k2, ← c34664] at e_545_19
  have hr := e_545_19
  simp only [e_545_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 546 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2658)) := by
  have c34665 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.1
  have c34666 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34667 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34668 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34669 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34670 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_546_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_0
  simp only [← c34665, ← c34666, ← c34667] at e_546_0
  have e_546_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_1
  simp only [← c34668, ← c34669, k2, ← c34670] at e_546_1
  have hr := e_546_1
  simp only [e_546_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 546 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2659)) := by
  have c34671 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34672 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34673 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34674 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34675 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34676 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_546_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_2
  simp only [← c34671, ← c34672, ← c34673] at e_546_2
  have e_546_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_3
  simp only [← c34674, ← c34675, k2, ← c34676] at e_546_3
  have hr := e_546_3
  simp only [e_546_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 546 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2660)) := by
  have c34677 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34678 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34679 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34680 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34681 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34682 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_546_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_4
  simp only [← c34677, ← c34678, ← c34679] at e_546_4
  have e_546_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_5
  simp only [← c34680, ← c34681, k2, ← c34682] at e_546_5
  have hr := e_546_5
  simp only [e_546_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 546 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2661)) := by
  have c34683 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34684 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34685 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34686 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34687 := (publicBatchWrapper86_copies1083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34688 := (publicBatchWrapper86_copies1084 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_546_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_6
  simp only [← c34683, ← c34684, ← c34685] at e_546_6
  have e_546_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_7
  simp only [← c34686, ← c34687, k2, ← c34688] at e_546_7
  have hr := e_546_7
  simp only [e_546_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 546 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2662)) := by
  have c34689 := (publicBatchWrapper86_copies1084 a h).2.1
  have c34690 := (publicBatchWrapper86_copies1084 a h).2.2.1
  have c34691 := (publicBatchWrapper86_copies1084 a h).2.2.2.1
  have c34692 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.1
  have c34693 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.1
  have c34694 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_546_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_8
  simp only [← c34689, ← c34690, ← c34691] at e_546_8
  have e_546_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_9
  simp only [← c34692, ← c34693, k2, ← c34694] at e_546_9
  have hr := e_546_9
  simp only [e_546_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 546 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2663)) := by
  have c34695 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.1
  have c34696 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.1
  have c34697 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.1
  have c34698 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34699 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34700 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_546_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_10
  simp only [← c34695, ← c34696, ← c34697] at e_546_10
  have e_546_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_11
  simp only [← c34698, ← c34699, k2, ← c34700] at e_546_11
  have hr := e_546_11
  simp only [e_546_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 546 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2664)) := by
  have c34701 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34702 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34703 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34704 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34705 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34706 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_546_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_12
  simp only [← c34701, ← c34702, ← c34703] at e_546_12
  have e_546_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_13
  simp only [← c34704, ← c34705, k2, ← c34706] at e_546_13
  have hr := e_546_13
  simp only [e_546_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 546 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2665)) := by
  have c34707 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34708 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34709 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34710 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34711 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34712 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_546_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_14
  simp only [← c34707, ← c34708, ← c34709] at e_546_14
  have e_546_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_15
  simp only [← c34710, ← c34711, k2, ← c34712] at e_546_15
  have hr := e_546_15
  simp only [e_546_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 546 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2666)) := by
  have c34713 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34714 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34715 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34716 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34717 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34718 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_546_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_16
  simp only [← c34713, ← c34714, ← c34715] at e_546_16
  have e_546_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_17
  simp only [← c34716, ← c34717, k2, ← c34718] at e_546_17
  have hr := e_546_17
  simp only [e_546_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 546 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2667)) := by
  have c34719 := (publicBatchWrapper86_copies1084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34720 := (publicBatchWrapper86_copies1085 a h).1
  have c34721 := (publicBatchWrapper86_copies1085 a h).2.1
  have c34722 := (publicBatchWrapper86_copies1085 a h).2.2.1
  have c34723 := (publicBatchWrapper86_copies1085 a h).2.2.2.1
  have c34724 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_546_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_18
  simp only [← c34719, ← c34720, ← c34721] at e_546_18
  have e_546_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row546 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_546_19
  simp only [← c34722, ← c34723, k2, ← c34724] at e_546_19
  have hr := e_546_19
  simp only [e_546_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 547 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2668)) := by
  have c34725 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.1
  have c34726 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.1
  have c34727 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.1
  have c34728 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.1
  have c34729 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.1
  have c34730 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_547_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_0
  simp only [← c34725, ← c34726, ← c34727] at e_547_0
  have e_547_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_1
  simp only [← c34728, ← c34729, k2, ← c34730] at e_547_1
  have hr := e_547_1
  simp only [e_547_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 547 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2669)) := by
  have c34731 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34732 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34733 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34734 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34735 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34736 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_547_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_2
  simp only [← c34731, ← c34732, ← c34733] at e_547_2
  have e_547_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_3
  simp only [← c34734, ← c34735, k2, ← c34736] at e_547_3
  have hr := e_547_3
  simp only [e_547_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 547 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2670)) := by
  have c34737 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34738 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34739 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34740 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34741 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34742 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_547_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_4
  simp only [← c34737, ← c34738, ← c34739] at e_547_4
  have e_547_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_5
  simp only [← c34740, ← c34741, k2, ← c34742] at e_547_5
  have hr := e_547_5
  simp only [e_547_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 547 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2671)) := by
  have c34743 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34744 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34745 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34746 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34747 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34748 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_547_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_6
  simp only [← c34743, ← c34744, ← c34745] at e_547_6
  have e_547_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_7
  simp only [← c34746, ← c34747, k2, ← c34748] at e_547_7
  have hr := e_547_7
  simp only [e_547_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 547 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2672)) := by
  have c34749 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34750 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34751 := (publicBatchWrapper86_copies1085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34752 := (publicBatchWrapper86_copies1086 a h).1
  have c34753 := (publicBatchWrapper86_copies1086 a h).2.1
  have c34754 := (publicBatchWrapper86_copies1086 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_547_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_8
  simp only [← c34749, ← c34750, ← c34751] at e_547_8
  have e_547_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_9
  simp only [← c34752, ← c34753, k2, ← c34754] at e_547_9
  have hr := e_547_9
  simp only [e_547_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 547 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2673)) := by
  have c34755 := (publicBatchWrapper86_copies1086 a h).2.2.2.1
  have c34756 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.1
  have c34757 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.1
  have c34758 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.1
  have c34759 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.1
  have c34760 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_547_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_10
  simp only [← c34755, ← c34756, ← c34757] at e_547_10
  have e_547_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_11
  simp only [← c34758, ← c34759, k2, ← c34760] at e_547_11
  have hr := e_547_11
  simp only [e_547_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 547 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2674)) := by
  have c34761 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.1
  have c34762 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34763 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34764 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34765 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34766 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_547_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_12
  simp only [← c34761, ← c34762, ← c34763] at e_547_12
  have e_547_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_13
  simp only [← c34764, ← c34765, k2, ← c34766] at e_547_13
  have hr := e_547_13
  simp only [e_547_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 547 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2675)) := by
  have c34767 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34768 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34769 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34770 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34771 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34772 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_547_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_14
  simp only [← c34767, ← c34768, ← c34769] at e_547_14
  have e_547_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_15
  simp only [← c34770, ← c34771, k2, ← c34772] at e_547_15
  have hr := e_547_15
  simp only [e_547_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 547 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2676)) := by
  have c34773 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34774 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34775 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34776 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34777 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34778 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_547_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_16
  simp only [← c34773, ← c34774, ← c34775] at e_547_16
  have e_547_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_17
  simp only [← c34776, ← c34777, k2, ← c34778] at e_547_17
  have hr := e_547_17
  simp only [e_547_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 547 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2677)) := by
  have c34779 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34780 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34781 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34782 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34783 := (publicBatchWrapper86_copies1086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34784 := (publicBatchWrapper86_copies1087 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_547_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_18
  simp only [← c34779, ← c34780, ← c34781] at e_547_18
  have e_547_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row547 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_547_19
  simp only [← c34782, ← c34783, k2, ← c34784] at e_547_19
  have hr := e_547_19
  simp only [e_547_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 548 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2678)) := by
  have c34785 := (publicBatchWrapper86_copies1087 a h).2.1
  have c34786 := (publicBatchWrapper86_copies1087 a h).2.2.1
  have c34787 := (publicBatchWrapper86_copies1087 a h).2.2.2.1
  have c34788 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.1
  have c34789 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.1
  have c34790 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_548_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_0
  simp only [← c34785, ← c34786, ← c34787] at e_548_0
  have e_548_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_1
  simp only [← c34788, ← c34789, k2, ← c34790] at e_548_1
  have hr := e_548_1
  simp only [e_548_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 548 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2679)) := by
  have c34791 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.1
  have c34792 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.1
  have c34793 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.1
  have c34794 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34795 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34796 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_548_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_2
  simp only [← c34791, ← c34792, ← c34793] at e_548_2
  have e_548_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_3
  simp only [← c34794, ← c34795, k2, ← c34796] at e_548_3
  have hr := e_548_3
  simp only [e_548_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 548 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2680)) := by
  have c34797 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34798 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34799 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34800 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34801 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34802 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_548_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_4
  simp only [← c34797, ← c34798, ← c34799] at e_548_4
  have e_548_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_5
  simp only [← c34800, ← c34801, k2, ← c34802] at e_548_5
  have hr := e_548_5
  simp only [e_548_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 548 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2681)) := by
  have c34803 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34804 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34805 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34806 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34807 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34808 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_548_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_6
  simp only [← c34803, ← c34804, ← c34805] at e_548_6
  have e_548_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_7
  simp only [← c34806, ← c34807, k2, ← c34808] at e_548_7
  have hr := e_548_7
  simp only [e_548_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 548 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2682)) := by
  have c34809 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34810 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34811 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34812 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34813 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34814 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_548_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_8
  simp only [← c34809, ← c34810, ← c34811] at e_548_8
  have e_548_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_9
  simp only [← c34812, ← c34813, k2, ← c34814] at e_548_9
  have hr := e_548_9
  simp only [e_548_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 548 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2683)) := by
  have c34815 := (publicBatchWrapper86_copies1087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34816 := (publicBatchWrapper86_copies1088 a h).1
  have c34817 := (publicBatchWrapper86_copies1088 a h).2.1
  have c34818 := (publicBatchWrapper86_copies1088 a h).2.2.1
  have c34819 := (publicBatchWrapper86_copies1088 a h).2.2.2.1
  have c34820 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_548_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_10
  simp only [← c34815, ← c34816, ← c34817] at e_548_10
  have e_548_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_11
  simp only [← c34818, ← c34819, k2, ← c34820] at e_548_11
  have hr := e_548_11
  simp only [e_548_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 548 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2684)) := by
  have c34821 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.1
  have c34822 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.1
  have c34823 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.1
  have c34824 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.1
  have c34825 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.1
  have c34826 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_548_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_12
  simp only [← c34821, ← c34822, ← c34823] at e_548_12
  have e_548_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_13
  simp only [← c34824, ← c34825, k2, ← c34826] at e_548_13
  have hr := e_548_13
  simp only [e_548_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 548 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2685)) := by
  have c34827 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34828 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34829 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34830 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34831 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34832 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_548_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_14
  simp only [← c34827, ← c34828, ← c34829] at e_548_14
  have e_548_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_15
  simp only [← c34830, ← c34831, k2, ← c34832] at e_548_15
  have hr := e_548_15
  simp only [e_548_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 548 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2686)) := by
  have c34833 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34834 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34835 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34836 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34837 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34838 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_548_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_16
  simp only [← c34833, ← c34834, ← c34835] at e_548_16
  have e_548_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_17
  simp only [← c34836, ← c34837, k2, ← c34838] at e_548_17
  have hr := e_548_17
  simp only [e_548_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 548 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2687)) := by
  have c34839 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34840 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34841 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34842 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34843 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34844 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_548_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_18
  simp only [← c34839, ← c34840, ← c34841] at e_548_18
  have e_548_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row548 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_548_19
  simp only [← c34842, ← c34843, k2, ← c34844] at e_548_19
  have hr := e_548_19
  simp only [e_548_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 549 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2688)) := by
  have c34845 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34846 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34847 := (publicBatchWrapper86_copies1088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34848 := (publicBatchWrapper86_copies1089 a h).1
  have c34849 := (publicBatchWrapper86_copies1089 a h).2.1
  have c34850 := (publicBatchWrapper86_copies1089 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_549_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_0
  simp only [← c34845, ← c34846, ← c34847] at e_549_0
  have e_549_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_1
  simp only [← c34848, ← c34849, k2, ← c34850] at e_549_1
  have hr := e_549_1
  simp only [e_549_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 549 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2689)) := by
  have c34851 := (publicBatchWrapper86_copies1089 a h).2.2.2.1
  have c34852 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.1
  have c34853 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.1
  have c34854 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.1
  have c34855 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.1
  have c34856 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_549_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_2
  simp only [← c34851, ← c34852, ← c34853] at e_549_2
  have e_549_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_3
  simp only [← c34854, ← c34855, k2, ← c34856] at e_549_3
  have hr := e_549_3
  simp only [e_549_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 549 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2690)) := by
  have c34857 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.1
  have c34858 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34859 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34860 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34861 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34862 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_549_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_4
  simp only [← c34857, ← c34858, ← c34859] at e_549_4
  have e_549_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_5
  simp only [← c34860, ← c34861, k2, ← c34862] at e_549_5
  have hr := e_549_5
  simp only [e_549_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 549 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2691)) := by
  have c34863 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34864 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34865 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34866 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34867 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34868 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_549_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_6
  simp only [← c34863, ← c34864, ← c34865] at e_549_6
  have e_549_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_7
  simp only [← c34866, ← c34867, k2, ← c34868] at e_549_7
  have hr := e_549_7
  simp only [e_549_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 549 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2692)) := by
  have c34869 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34870 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34871 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34872 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34873 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34874 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_549_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_8
  simp only [← c34869, ← c34870, ← c34871] at e_549_8
  have e_549_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_9
  simp only [← c34872, ← c34873, k2, ← c34874] at e_549_9
  have hr := e_549_9
  simp only [e_549_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 549 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2693)) := by
  have c34875 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34876 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34877 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34878 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34879 := (publicBatchWrapper86_copies1089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34880 := (publicBatchWrapper86_copies1090 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_549_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_10
  simp only [← c34875, ← c34876, ← c34877] at e_549_10
  have e_549_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_11
  simp only [← c34878, ← c34879, k2, ← c34880] at e_549_11
  have hr := e_549_11
  simp only [e_549_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 549 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2694)) := by
  have c34881 := (publicBatchWrapper86_copies1090 a h).2.1
  have c34882 := (publicBatchWrapper86_copies1090 a h).2.2.1
  have c34883 := (publicBatchWrapper86_copies1090 a h).2.2.2.1
  have c34884 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.1
  have c34885 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.1
  have c34886 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_549_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_12
  simp only [← c34881, ← c34882, ← c34883] at e_549_12
  have e_549_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_13
  simp only [← c34884, ← c34885, k2, ← c34886] at e_549_13
  have hr := e_549_13
  simp only [e_549_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 549 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2695)) := by
  have c34887 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.1
  have c34888 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.1
  have c34889 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.1
  have c34890 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34891 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34892 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_549_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_14
  simp only [← c34887, ← c34888, ← c34889] at e_549_14
  have e_549_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_15
  simp only [← c34890, ← c34891, k2, ← c34892] at e_549_15
  have hr := e_549_15
  simp only [e_549_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 549 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2696)) := by
  have c34893 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34894 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34895 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34896 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34897 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34898 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_549_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_16
  simp only [← c34893, ← c34894, ← c34895] at e_549_16
  have e_549_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_17
  simp only [← c34896, ← c34897, k2, ← c34898] at e_549_17
  have hr := e_549_17
  simp only [e_549_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 549 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2697)) := by
  have c34899 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34900 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34901 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34902 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34903 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34904 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_549_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_18
  simp only [← c34899, ← c34900, ← c34901] at e_549_18
  have e_549_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row549 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_549_19
  simp only [← c34902, ← c34903, k2, ← c34904] at e_549_19
  have hr := e_549_19
  simp only [e_549_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 550 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2698)) := by
  have c34905 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34906 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34907 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34908 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34909 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34910 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_550_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_0
  simp only [← c34905, ← c34906, ← c34907] at e_550_0
  have e_550_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_1
  simp only [← c34908, ← c34909, k2, ← c34910] at e_550_1
  have hr := e_550_1
  simp only [e_550_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 550 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2699)) := by
  have c34911 := (publicBatchWrapper86_copies1090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34912 := (publicBatchWrapper86_copies1091 a h).1
  have c34913 := (publicBatchWrapper86_copies1091 a h).2.1
  have c34914 := (publicBatchWrapper86_copies1091 a h).2.2.1
  have c34915 := (publicBatchWrapper86_copies1091 a h).2.2.2.1
  have c34916 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_550_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_2
  simp only [← c34911, ← c34912, ← c34913] at e_550_2
  have e_550_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_3
  simp only [← c34914, ← c34915, k2, ← c34916] at e_550_3
  have hr := e_550_3
  simp only [e_550_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 550 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2700)) := by
  have c34917 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.1
  have c34918 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.1
  have c34919 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.1
  have c34920 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.1
  have c34921 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.1
  have c34922 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_550_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_4
  simp only [← c34917, ← c34918, ← c34919] at e_550_4
  have e_550_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_5
  simp only [← c34920, ← c34921, k2, ← c34922] at e_550_5
  have hr := e_550_5
  simp only [e_550_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 550 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2701)) := by
  have c34923 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34924 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34925 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34926 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34927 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34928 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_550_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_6
  simp only [← c34923, ← c34924, ← c34925] at e_550_6
  have e_550_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_7
  simp only [← c34926, ← c34927, k2, ← c34928] at e_550_7
  have hr := e_550_7
  simp only [e_550_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 550 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2702)) := by
  have c34929 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34930 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34931 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34932 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34933 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34934 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_550_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_8
  simp only [← c34929, ← c34930, ← c34931] at e_550_8
  have e_550_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_9
  simp only [← c34932, ← c34933, k2, ← c34934] at e_550_9
  have hr := e_550_9
  simp only [e_550_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 550 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2703)) := by
  have c34935 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34936 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34937 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34938 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34939 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34940 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_550_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_10
  simp only [← c34935, ← c34936, ← c34937] at e_550_10
  have e_550_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_11
  simp only [← c34938, ← c34939, k2, ← c34940] at e_550_11
  have hr := e_550_11
  simp only [e_550_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 550 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2704)) := by
  have c34941 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34942 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34943 := (publicBatchWrapper86_copies1091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34944 := (publicBatchWrapper86_copies1092 a h).1
  have c34945 := (publicBatchWrapper86_copies1092 a h).2.1
  have c34946 := (publicBatchWrapper86_copies1092 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_550_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_12
  simp only [← c34941, ← c34942, ← c34943] at e_550_12
  have e_550_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_13
  simp only [← c34944, ← c34945, k2, ← c34946] at e_550_13
  have hr := e_550_13
  simp only [e_550_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 550 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2705)) := by
  have c34947 := (publicBatchWrapper86_copies1092 a h).2.2.2.1
  have c34948 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.1
  have c34949 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.1
  have c34950 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.1
  have c34951 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.1
  have c34952 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_550_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_14
  simp only [← c34947, ← c34948, ← c34949] at e_550_14
  have e_550_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_15
  simp only [← c34950, ← c34951, k2, ← c34952] at e_550_15
  have hr := e_550_15
  simp only [e_550_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 550 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2706)) := by
  have c34953 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.1
  have c34954 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34955 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34956 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34957 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34958 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_550_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_16
  simp only [← c34953, ← c34954, ← c34955] at e_550_16
  have e_550_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_17
  simp only [← c34956, ← c34957, k2, ← c34958] at e_550_17
  have hr := e_550_17
  simp only [e_550_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 550 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2707)) := by
  have c34959 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34960 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34961 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34962 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34963 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34964 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_550_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_18
  simp only [← c34959, ← c34960, ← c34961] at e_550_18
  have e_550_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row550 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_550_19
  simp only [← c34962, ← c34963, k2, ← c34964] at e_550_19
  have hr := e_550_19
  simp only [e_550_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 551 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2708)) := by
  have c34965 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34966 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34967 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34968 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34969 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34970 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_551_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_0
  simp only [← c34965, ← c34966, ← c34967] at e_551_0
  have e_551_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_1
  simp only [← c34968, ← c34969, k2, ← c34970] at e_551_1
  have hr := e_551_1
  simp only [e_551_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 551 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2709)) := by
  have c34971 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34972 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34973 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34974 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34975 := (publicBatchWrapper86_copies1092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c34976 := (publicBatchWrapper86_copies1093 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_551_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_2
  simp only [← c34971, ← c34972, ← c34973] at e_551_2
  have e_551_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_3
  simp only [← c34974, ← c34975, k2, ← c34976] at e_551_3
  have hr := e_551_3
  simp only [e_551_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 551 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2710)) := by
  have c34977 := (publicBatchWrapper86_copies1093 a h).2.1
  have c34978 := (publicBatchWrapper86_copies1093 a h).2.2.1
  have c34979 := (publicBatchWrapper86_copies1093 a h).2.2.2.1
  have c34980 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.1
  have c34981 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.1
  have c34982 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_551_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_4
  simp only [← c34977, ← c34978, ← c34979] at e_551_4
  have e_551_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_5
  simp only [← c34980, ← c34981, k2, ← c34982] at e_551_5
  have hr := e_551_5
  simp only [e_551_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 551 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2711)) := by
  have c34983 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.1
  have c34984 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.1
  have c34985 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.1
  have c34986 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.1
  have c34987 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c34988 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_551_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_6
  simp only [← c34983, ← c34984, ← c34985] at e_551_6
  have e_551_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_7
  simp only [← c34986, ← c34987, k2, ← c34988] at e_551_7
  have hr := e_551_7
  simp only [e_551_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 551 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2712)) := by
  have c34989 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34990 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34991 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34992 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34993 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34994 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_551_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_8
  simp only [← c34989, ← c34990, ← c34991] at e_551_8
  have e_551_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_9
  simp only [← c34992, ← c34993, k2, ← c34994] at e_551_9
  have hr := e_551_9
  simp only [e_551_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 551 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2713)) := by
  have c34995 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34996 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34997 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34998 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c34999 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35000 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_551_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_10
  simp only [← c34995, ← c34996, ← c34997] at e_551_10
  have e_551_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_11
  simp only [← c34998, ← c34999, k2, ← c35000] at e_551_11
  have hr := e_551_11
  simp only [e_551_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 551 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2714)) := by
  have c35001 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35002 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35003 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35004 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35005 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35006 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_551_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_12
  simp only [← c35001, ← c35002, ← c35003] at e_551_12
  have e_551_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_13
  simp only [← c35004, ← c35005, k2, ← c35006] at e_551_13
  have hr := e_551_13
  simp only [e_551_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 551 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2715)) := by
  have c35007 := (publicBatchWrapper86_copies1093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35008 := (publicBatchWrapper86_copies1094 a h).1
  have c35009 := (publicBatchWrapper86_copies1094 a h).2.1
  have c35010 := (publicBatchWrapper86_copies1094 a h).2.2.1
  have c35011 := (publicBatchWrapper86_copies1094 a h).2.2.2.1
  have c35012 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_551_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_14
  simp only [← c35007, ← c35008, ← c35009] at e_551_14
  have e_551_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_15
  simp only [← c35010, ← c35011, k2, ← c35012] at e_551_15
  have hr := e_551_15
  simp only [e_551_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 551 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2716)) := by
  have c35013 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.1
  have c35014 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.1
  have c35015 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.1
  have c35016 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.1
  have c35017 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.1
  have c35018 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_551_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_16
  simp only [← c35013, ← c35014, ← c35015] at e_551_16
  have e_551_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_17
  simp only [← c35016, ← c35017, k2, ← c35018] at e_551_17
  have hr := e_551_17
  simp only [e_551_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 551 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2717)) := by
  have c35019 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35020 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35021 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35022 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35023 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35024 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_551_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_18
  simp only [← c35019, ← c35020, ← c35021] at e_551_18
  have e_551_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row551 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_551_19
  simp only [← c35022, ← c35023, k2, ← c35024] at e_551_19
  have hr := e_551_19
  simp only [e_551_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 552 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2718)) := by
  have c35025 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35026 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35027 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35028 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35029 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35030 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_552_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_0
  simp only [← c35025, ← c35026, ← c35027] at e_552_0
  have e_552_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_1
  simp only [← c35028, ← c35029, k2, ← c35030] at e_552_1
  have hr := e_552_1
  simp only [e_552_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 552 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2719)) := by
  have c35031 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35032 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35033 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35034 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35035 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35036 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_552_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_2
  simp only [← c35031, ← c35032, ← c35033] at e_552_2
  have e_552_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_3
  simp only [← c35034, ← c35035, k2, ← c35036] at e_552_3
  have hr := e_552_3
  simp only [e_552_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 552 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2720)) := by
  have c35037 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35038 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35039 := (publicBatchWrapper86_copies1094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35040 := (publicBatchWrapper86_copies1095 a h).1
  have c35041 := (publicBatchWrapper86_copies1095 a h).2.1
  have c35042 := (publicBatchWrapper86_copies1095 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_552_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_4
  simp only [← c35037, ← c35038, ← c35039] at e_552_4
  have e_552_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_5
  simp only [← c35040, ← c35041, k2, ← c35042] at e_552_5
  have hr := e_552_5
  simp only [e_552_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 552 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2721)) := by
  have c35043 := (publicBatchWrapper86_copies1095 a h).2.2.2.1
  have c35044 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.1
  have c35045 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.1
  have c35046 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.1
  have c35047 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.1
  have c35048 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_552_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_6
  simp only [← c35043, ← c35044, ← c35045] at e_552_6
  have e_552_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_7
  simp only [← c35046, ← c35047, k2, ← c35048] at e_552_7
  have hr := e_552_7
  simp only [e_552_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 552 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2722)) := by
  have c35049 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.1
  have c35050 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35051 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35052 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35053 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35054 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_552_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_8
  simp only [← c35049, ← c35050, ← c35051] at e_552_8
  have e_552_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_9
  simp only [← c35052, ← c35053, k2, ← c35054] at e_552_9
  have hr := e_552_9
  simp only [e_552_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 552 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2723)) := by
  have c35055 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35056 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35057 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35058 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35059 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35060 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_552_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_10
  simp only [← c35055, ← c35056, ← c35057] at e_552_10
  have e_552_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_11
  simp only [← c35058, ← c35059, k2, ← c35060] at e_552_11
  have hr := e_552_11
  simp only [e_552_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 552 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2724)) := by
  have c35061 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35062 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35063 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35064 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35065 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35066 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_552_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_12
  simp only [← c35061, ← c35062, ← c35063] at e_552_12
  have e_552_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_13
  simp only [← c35064, ← c35065, k2, ← c35066] at e_552_13
  have hr := e_552_13
  simp only [e_552_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 552 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2725)) := by
  have c35067 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35068 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35069 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35070 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35071 := (publicBatchWrapper86_copies1095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35072 := (publicBatchWrapper86_copies1096 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_552_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_14
  simp only [← c35067, ← c35068, ← c35069] at e_552_14
  have e_552_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_15
  simp only [← c35070, ← c35071, k2, ← c35072] at e_552_15
  have hr := e_552_15
  simp only [e_552_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 552 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2726)) := by
  have c35073 := (publicBatchWrapper86_copies1096 a h).2.1
  have c35074 := (publicBatchWrapper86_copies1096 a h).2.2.1
  have c35075 := (publicBatchWrapper86_copies1096 a h).2.2.2.1
  have c35076 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.1
  have c35077 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.1
  have c35078 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_552_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_16
  simp only [← c35073, ← c35074, ← c35075] at e_552_16
  have e_552_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_17
  simp only [← c35076, ← c35077, k2, ← c35078] at e_552_17
  have hr := e_552_17
  simp only [e_552_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 552 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2727)) := by
  have c35079 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.1
  have c35080 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.1
  have c35081 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.1
  have c35082 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35083 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35084 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_552_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_18
  simp only [← c35079, ← c35080, ← c35081] at e_552_18
  have e_552_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row552 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_552_19
  simp only [← c35082, ← c35083, k2, ← c35084] at e_552_19
  have hr := e_552_19
  simp only [e_552_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 553 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2728)) := by
  have c35085 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35086 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35087 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35088 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35089 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35090 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_553_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_0
  simp only [← c35085, ← c35086, ← c35087] at e_553_0
  have e_553_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_1
  simp only [← c35088, ← c35089, k2, ← c35090] at e_553_1
  have hr := e_553_1
  simp only [e_553_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 553 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2729)) := by
  have c35091 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35092 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35093 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35094 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35095 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35096 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_553_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_2
  simp only [← c35091, ← c35092, ← c35093] at e_553_2
  have e_553_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_3
  simp only [← c35094, ← c35095, k2, ← c35096] at e_553_3
  have hr := e_553_3
  simp only [e_553_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 553 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2730)) := by
  have c35097 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35098 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35099 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35100 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35101 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35102 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_553_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_4
  simp only [← c35097, ← c35098, ← c35099] at e_553_4
  have e_553_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_5
  simp only [← c35100, ← c35101, k2, ← c35102] at e_553_5
  have hr := e_553_5
  simp only [e_553_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 553 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2731)) := by
  have c35103 := (publicBatchWrapper86_copies1096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35104 := (publicBatchWrapper86_copies1097 a h).1
  have c35105 := (publicBatchWrapper86_copies1097 a h).2.1
  have c35106 := (publicBatchWrapper86_copies1097 a h).2.2.1
  have c35107 := (publicBatchWrapper86_copies1097 a h).2.2.2.1
  have c35108 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_553_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_6
  simp only [← c35103, ← c35104, ← c35105] at e_553_6
  have e_553_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_7
  simp only [← c35106, ← c35107, k2, ← c35108] at e_553_7
  have hr := e_553_7
  simp only [e_553_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 553 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2732)) := by
  have c35109 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.1
  have c35110 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.1
  have c35111 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.1
  have c35112 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.1
  have c35113 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.1
  have c35114 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_553_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_8
  simp only [← c35109, ← c35110, ← c35111] at e_553_8
  have e_553_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_9
  simp only [← c35112, ← c35113, k2, ← c35114] at e_553_9
  have hr := e_553_9
  simp only [e_553_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 553 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2733)) := by
  have c35115 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35116 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35117 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35118 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35119 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35120 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_553_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_10
  simp only [← c35115, ← c35116, ← c35117] at e_553_10
  have e_553_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_11
  simp only [← c35118, ← c35119, k2, ← c35120] at e_553_11
  have hr := e_553_11
  simp only [e_553_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 553 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2734)) := by
  have c35121 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35122 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35123 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35124 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35125 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35126 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_553_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_12
  simp only [← c35121, ← c35122, ← c35123] at e_553_12
  have e_553_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_13
  simp only [← c35124, ← c35125, k2, ← c35126] at e_553_13
  have hr := e_553_13
  simp only [e_553_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 553 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2735)) := by
  have c35127 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35128 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35129 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35130 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35131 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35132 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_553_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_14
  simp only [← c35127, ← c35128, ← c35129] at e_553_14
  have e_553_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_15
  simp only [← c35130, ← c35131, k2, ← c35132] at e_553_15
  have hr := e_553_15
  simp only [e_553_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 553 71) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2736)) := by
  have c35133 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35134 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35135 := (publicBatchWrapper86_copies1097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35136 := (publicBatchWrapper86_copies1098 a h).1
  have c35137 := (publicBatchWrapper86_copies1098 a h).2.1
  have c35138 := (publicBatchWrapper86_copies1098 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_553_16 := arithEq_of_rows h (i := 16) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_16
  simp only [← c35133, ← c35134, ← c35135] at e_553_16
  have e_553_17 := arithEq_of_rows h (i := 17) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_17
  simp only [← c35136, ← c35137, k2, ← c35138] at e_553_17
  have hr := e_553_17
  simp only [e_553_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 553 79) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2737)) := by
  have c35139 := (publicBatchWrapper86_copies1098 a h).2.2.2.1
  have c35140 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.1
  have c35141 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.1
  have c35142 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.1
  have c35143 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.1
  have c35144 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_553_18 := arithEq_of_rows h (i := 18) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_18
  simp only [← c35139, ← c35140, ← c35141] at e_553_18
  have e_553_19 := arithEq_of_rows h (i := 19) publicBatchWrapper86_row553 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_553_19
  simp only [← c35142, ← c35143, k2, ← c35144] at e_553_19
  have hr := e_553_19
  simp only [e_553_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 554 7) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2738)) := by
  have c35145 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.1
  have c35146 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35147 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35148 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35149 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35150 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_554_0 := arithEq_of_rows h (i := 0) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_0
  simp only [← c35145, ← c35146, ← c35147] at e_554_0
  have e_554_1 := arithEq_of_rows h (i := 1) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_1
  simp only [← c35148, ← c35149, k2, ← c35150] at e_554_1
  have hr := e_554_1
  simp only [e_554_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 554 15) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2739)) := by
  have c35151 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35152 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35153 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35154 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35155 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35156 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_554_2 := arithEq_of_rows h (i := 2) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_2
  simp only [← c35151, ← c35152, ← c35153] at e_554_2
  have e_554_3 := arithEq_of_rows h (i := 3) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_3
  simp only [← c35154, ← c35155, k2, ← c35156] at e_554_3
  have hr := e_554_3
  simp only [e_554_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 554 23) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2740)) := by
  have c35157 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35158 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35159 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35160 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35161 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35162 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_554_4 := arithEq_of_rows h (i := 4) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_4
  simp only [← c35157, ← c35158, ← c35159] at e_554_4
  have e_554_5 := arithEq_of_rows h (i := 5) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_5
  simp only [← c35160, ← c35161, k2, ← c35162] at e_554_5
  have hr := e_554_5
  simp only [e_554_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 554 31) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2741)) := by
  have c35163 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35164 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35165 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35166 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35167 := (publicBatchWrapper86_copies1098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c35168 := (publicBatchWrapper86_copies1099 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_554_6 := arithEq_of_rows h (i := 6) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_6
  simp only [← c35163, ← c35164, ← c35165] at e_554_6
  have e_554_7 := arithEq_of_rows h (i := 7) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_7
  simp only [← c35166, ← c35167, k2, ← c35168] at e_554_7
  have hr := e_554_7
  simp only [e_554_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 554 39) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2742)) := by
  have c35169 := (publicBatchWrapper86_copies1099 a h).2.1
  have c35170 := (publicBatchWrapper86_copies1099 a h).2.2.1
  have c35171 := (publicBatchWrapper86_copies1099 a h).2.2.2.1
  have c35172 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.1
  have c35173 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.1
  have c35174 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_554_8 := arithEq_of_rows h (i := 8) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_8
  simp only [← c35169, ← c35170, ← c35171] at e_554_8
  have e_554_9 := arithEq_of_rows h (i := 9) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_9
  simp only [← c35172, ← c35173, k2, ← c35174] at e_554_9
  have hr := e_554_9
  simp only [e_554_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 554 47) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2743)) := by
  have c35175 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.1
  have c35176 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.1
  have c35177 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.1
  have c35178 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.1
  have c35179 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c35180 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_554_10 := arithEq_of_rows h (i := 10) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_10
  simp only [← c35175, ← c35176, ← c35177] at e_554_10
  have e_554_11 := arithEq_of_rows h (i := 11) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_11
  simp only [← c35178, ← c35179, k2, ← c35180] at e_554_11
  have hr := e_554_11
  simp only [e_554_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 554 55) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2744)) := by
  have c35181 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35182 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35183 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35184 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35185 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35186 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_554_12 := arithEq_of_rows h (i := 12) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_12
  simp only [← c35181, ← c35182, ← c35183] at e_554_12
  have e_554_13 := arithEq_of_rows h (i := 13) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_13
  simp only [← c35184, ← c35185, k2, ← c35186] at e_554_13
  have hr := e_554_13
  simp only [e_554_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f5139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 554 63) = bselect (a (.wire 15 39)) (a (.virt 200)) (a (.virt 2745)) := by
  have c35187 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35188 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35189 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35190 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35191 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c35192 := (publicBatchWrapper86_copies1099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_554_14 := arithEq_of_rows h (i := 14) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_14
  simp only [← c35187, ← c35188, ← c35189] at e_554_14
  have e_554_15 := arithEq_of_rows h (i := 15) publicBatchWrapper86_row554 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_554_15
  simp only [← c35190, ← c35191, k2, ← c35192] at e_554_15
  have hr := e_554_15
  simp only [e_554_14] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
