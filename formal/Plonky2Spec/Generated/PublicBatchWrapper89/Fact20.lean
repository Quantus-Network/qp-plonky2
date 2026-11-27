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

theorem publicBatchWrapper89_f15173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15157)) := by
  have c95544 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95545 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95546 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95547 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95548 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95549 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_4
  simp only [← c95544, ← c95545, ← c95546] at e_1559_4
  have e_1559_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_5
  simp only [← c95547, ← c95548, k2, ← c95549] at e_1559_5
  have hr := e_1559_5
  simp only [e_1559_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15158)) := by
  have c95550 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95551 := (publicBatchWrapper89_copies2985 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95552 := (publicBatchWrapper89_copies2986 a h).1
  have c95553 := (publicBatchWrapper89_copies2986 a h).2.1
  have c95554 := (publicBatchWrapper89_copies2986 a h).2.2.1
  have c95555 := (publicBatchWrapper89_copies2986 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_6
  simp only [← c95550, ← c95551, ← c95552] at e_1559_6
  have e_1559_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_7
  simp only [← c95553, ← c95554, k2, ← c95555] at e_1559_7
  have hr := e_1559_7
  simp only [e_1559_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15159)) := by
  have c95556 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.1
  have c95557 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.1
  have c95558 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.1
  have c95559 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.1
  have c95560 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.1
  have c95561 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_8
  simp only [← c95556, ← c95557, ← c95558] at e_1559_8
  have e_1559_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_9
  simp only [← c95559, ← c95560, k2, ← c95561] at e_1559_9
  have hr := e_1559_9
  simp only [e_1559_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 47) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15160)) := by
  have c95562 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95563 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95564 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95565 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95566 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95567 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_10
  simp only [← c95562, ← c95563, ← c95564] at e_1559_10
  have e_1559_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_11
  simp only [← c95565, ← c95566, k2, ← c95567] at e_1559_11
  have hr := e_1559_11
  simp only [e_1559_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 55) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15161)) := by
  have c95568 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95569 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95570 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95571 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95572 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95573 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_12
  simp only [← c95568, ← c95569, ← c95570] at e_1559_12
  have e_1559_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_13
  simp only [← c95571, ← c95572, k2, ← c95573] at e_1559_13
  have hr := e_1559_13
  simp only [e_1559_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 63) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15162)) := by
  have c95574 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95575 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95576 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95577 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95578 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95579 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_14
  simp only [← c95574, ← c95575, ← c95576] at e_1559_14
  have e_1559_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_15
  simp only [← c95577, ← c95578, k2, ← c95579] at e_1559_15
  have hr := e_1559_15
  simp only [e_1559_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 71) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15163)) := by
  have c95580 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95581 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95582 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95583 := (publicBatchWrapper89_copies2986 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95584 := (publicBatchWrapper89_copies2987 a h).1
  have c95585 := (publicBatchWrapper89_copies2987 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_16
  simp only [← c95580, ← c95581, ← c95582] at e_1559_16
  have e_1559_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_17
  simp only [← c95583, ← c95584, k2, ← c95585] at e_1559_17
  have hr := e_1559_17
  simp only [e_1559_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1559 79) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15164)) := by
  have c95586 := (publicBatchWrapper89_copies2987 a h).2.2.1
  have c95587 := (publicBatchWrapper89_copies2987 a h).2.2.2.1
  have c95588 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.1
  have c95589 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.1
  have c95590 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.1
  have c95591 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1559_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_18
  simp only [← c95586, ← c95587, ← c95588] at e_1559_18
  have e_1559_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1559 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1559_19
  simp only [← c95589, ← c95590, k2, ← c95591] at e_1559_19
  have hr := e_1559_19
  simp only [e_1559_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 7) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15165)) := by
  have c95592 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.1
  have c95593 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.1
  have c95594 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95595 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95596 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95597 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_0
  simp only [← c95592, ← c95593, ← c95594] at e_1560_0
  have e_1560_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_1
  simp only [← c95595, ← c95596, k2, ← c95597] at e_1560_1
  have hr := e_1560_1
  simp only [e_1560_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 15) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15166)) := by
  have c95598 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95599 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95600 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95601 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95602 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95603 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_2
  simp only [← c95598, ← c95599, ← c95600] at e_1560_2
  have e_1560_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_3
  simp only [← c95601, ← c95602, k2, ← c95603] at e_1560_3
  have hr := e_1560_3
  simp only [e_1560_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 23) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15167)) := by
  have c95604 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95605 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95606 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95607 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95608 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95609 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_4
  simp only [← c95604, ← c95605, ← c95606] at e_1560_4
  have e_1560_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_5
  simp only [← c95607, ← c95608, k2, ← c95609] at e_1560_5
  have hr := e_1560_5
  simp only [e_1560_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 31) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15168)) := by
  have c95610 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95611 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95612 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95613 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95614 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95615 := (publicBatchWrapper89_copies2987 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_6
  simp only [← c95610, ← c95611, ← c95612] at e_1560_6
  have e_1560_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_7
  simp only [← c95613, ← c95614, k2, ← c95615] at e_1560_7
  have hr := e_1560_7
  simp only [e_1560_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 39) = bselect (a (.wire 72 79)) (a (.virt 200)) (a (.virt 15169)) := by
  have c95616 := (publicBatchWrapper89_copies2988 a h).1
  have c95617 := (publicBatchWrapper89_copies2988 a h).2.1
  have c95618 := (publicBatchWrapper89_copies2988 a h).2.2.1
  have c95619 := (publicBatchWrapper89_copies2988 a h).2.2.2.1
  have c95620 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.1
  have c95621 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_8
  simp only [← c95616, ← c95617, ← c95618] at e_1560_8
  have e_1560_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_9
  simp only [← c95619, ← c95620, k2, ← c95621] at e_1560_9
  have hr := e_1560_9
  simp only [e_1560_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15210)) := by
  have c95622 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.1
  have c95623 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.1
  have c95624 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.1
  have c95625 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.1
  have c95626 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95627 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_10
  simp only [← c95622, ← c95623, ← c95624] at e_1560_10
  have e_1560_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_11
  simp only [← c95625, ← c95626, k2, ← c95627] at e_1560_11
  have hr := e_1560_11
  simp only [e_1560_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15211)) := by
  have c95628 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95629 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95630 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95631 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95632 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95633 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_12
  simp only [← c95628, ← c95629, ← c95630] at e_1560_12
  have e_1560_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_13
  simp only [← c95631, ← c95632, k2, ← c95633] at e_1560_13
  have hr := e_1560_13
  simp only [e_1560_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15212)) := by
  have c95634 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95635 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95636 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95637 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95638 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95639 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_14
  simp only [← c95634, ← c95635, ← c95636] at e_1560_14
  have e_1560_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_15
  simp only [← c95637, ← c95638, k2, ← c95639] at e_1560_15
  have hr := e_1560_15
  simp only [e_1560_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15213)) := by
  have c95640 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95641 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95642 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95643 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95644 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95645 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_16
  simp only [← c95640, ← c95641, ← c95642] at e_1560_16
  have e_1560_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_17
  simp only [← c95643, ← c95644, k2, ← c95645] at e_1560_17
  have hr := e_1560_17
  simp only [e_1560_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1560 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15214)) := by
  have c95646 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95647 := (publicBatchWrapper89_copies2988 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95648 := (publicBatchWrapper89_copies2989 a h).1
  have c95649 := (publicBatchWrapper89_copies2989 a h).2.1
  have c95650 := (publicBatchWrapper89_copies2989 a h).2.2.1
  have c95651 := (publicBatchWrapper89_copies2989 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1560_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_18
  simp only [← c95646, ← c95647, ← c95648] at e_1560_18
  have e_1560_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1560 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1560_19
  simp only [← c95649, ← c95650, k2, ← c95651] at e_1560_19
  have hr := e_1560_19
  simp only [e_1560_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15215)) := by
  have c95652 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.1
  have c95653 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.1
  have c95654 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.1
  have c95655 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.1
  have c95656 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.1
  have c95657 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_0
  simp only [← c95652, ← c95653, ← c95654] at e_1561_0
  have e_1561_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_1
  simp only [← c95655, ← c95656, k2, ← c95657] at e_1561_1
  have hr := e_1561_1
  simp only [e_1561_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15216)) := by
  have c95658 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95659 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95660 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95661 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95662 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95663 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_2
  simp only [← c95658, ← c95659, ← c95660] at e_1561_2
  have e_1561_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_3
  simp only [← c95661, ← c95662, k2, ← c95663] at e_1561_3
  have hr := e_1561_3
  simp only [e_1561_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15217)) := by
  have c95664 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95665 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95666 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95667 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95668 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95669 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_4
  simp only [← c95664, ← c95665, ← c95666] at e_1561_4
  have e_1561_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_5
  simp only [← c95667, ← c95668, k2, ← c95669] at e_1561_5
  have hr := e_1561_5
  simp only [e_1561_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15218)) := by
  have c95670 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95671 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95672 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95673 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95674 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95675 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_6
  simp only [← c95670, ← c95671, ← c95672] at e_1561_6
  have e_1561_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_7
  simp only [← c95673, ← c95674, k2, ← c95675] at e_1561_7
  have hr := e_1561_7
  simp only [e_1561_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15219)) := by
  have c95676 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95677 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95678 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95679 := (publicBatchWrapper89_copies2989 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95680 := (publicBatchWrapper89_copies2990 a h).1
  have c95681 := (publicBatchWrapper89_copies2990 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_8
  simp only [← c95676, ← c95677, ← c95678] at e_1561_8
  have e_1561_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_9
  simp only [← c95679, ← c95680, k2, ← c95681] at e_1561_9
  have hr := e_1561_9
  simp only [e_1561_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15220)) := by
  have c95682 := (publicBatchWrapper89_copies2990 a h).2.2.1
  have c95683 := (publicBatchWrapper89_copies2990 a h).2.2.2.1
  have c95684 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.1
  have c95685 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.1
  have c95686 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.1
  have c95687 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_10
  simp only [← c95682, ← c95683, ← c95684] at e_1561_10
  have e_1561_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_11
  simp only [← c95685, ← c95686, k2, ← c95687] at e_1561_11
  have hr := e_1561_11
  simp only [e_1561_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15221)) := by
  have c95688 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.1
  have c95689 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.1
  have c95690 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95691 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95692 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95693 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_12
  simp only [← c95688, ← c95689, ← c95690] at e_1561_12
  have e_1561_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_13
  simp only [← c95691, ← c95692, k2, ← c95693] at e_1561_13
  have hr := e_1561_13
  simp only [e_1561_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15222)) := by
  have c95694 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95695 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95696 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95697 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95698 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95699 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_14
  simp only [← c95694, ← c95695, ← c95696] at e_1561_14
  have e_1561_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_15
  simp only [← c95697, ← c95698, k2, ← c95699] at e_1561_15
  have hr := e_1561_15
  simp only [e_1561_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15223)) := by
  have c95700 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95701 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95702 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95703 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95704 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95705 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_16
  simp only [← c95700, ← c95701, ← c95702] at e_1561_16
  have e_1561_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_17
  simp only [← c95703, ← c95704, k2, ← c95705] at e_1561_17
  have hr := e_1561_17
  simp only [e_1561_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1561 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15224)) := by
  have c95706 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95707 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95708 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95709 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95710 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95711 := (publicBatchWrapper89_copies2990 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1561_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_18
  simp only [← c95706, ← c95707, ← c95708] at e_1561_18
  have e_1561_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1561 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1561_19
  simp only [← c95709, ← c95710, k2, ← c95711] at e_1561_19
  have hr := e_1561_19
  simp only [e_1561_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15225)) := by
  have c95712 := (publicBatchWrapper89_copies2991 a h).1
  have c95713 := (publicBatchWrapper89_copies2991 a h).2.1
  have c95714 := (publicBatchWrapper89_copies2991 a h).2.2.1
  have c95715 := (publicBatchWrapper89_copies2991 a h).2.2.2.1
  have c95716 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.1
  have c95717 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_0
  simp only [← c95712, ← c95713, ← c95714] at e_1562_0
  have e_1562_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_1
  simp only [← c95715, ← c95716, k2, ← c95717] at e_1562_1
  have hr := e_1562_1
  simp only [e_1562_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15226)) := by
  have c95718 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.1
  have c95719 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.1
  have c95720 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.1
  have c95721 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.1
  have c95722 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95723 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_2
  simp only [← c95718, ← c95719, ← c95720] at e_1562_2
  have e_1562_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_3
  simp only [← c95721, ← c95722, k2, ← c95723] at e_1562_3
  have hr := e_1562_3
  simp only [e_1562_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15227)) := by
  have c95724 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95725 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95726 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95727 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95728 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95729 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_4
  simp only [← c95724, ← c95725, ← c95726] at e_1562_4
  have e_1562_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_5
  simp only [← c95727, ← c95728, k2, ← c95729] at e_1562_5
  have hr := e_1562_5
  simp only [e_1562_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15228)) := by
  have c95730 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95731 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95732 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95733 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95734 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95735 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_6
  simp only [← c95730, ← c95731, ← c95732] at e_1562_6
  have e_1562_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_7
  simp only [← c95733, ← c95734, k2, ← c95735] at e_1562_7
  have hr := e_1562_7
  simp only [e_1562_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15229)) := by
  have c95736 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95737 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95738 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95739 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95740 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95741 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_8
  simp only [← c95736, ← c95737, ← c95738] at e_1562_8
  have e_1562_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_9
  simp only [← c95739, ← c95740, k2, ← c95741] at e_1562_9
  have hr := e_1562_9
  simp only [e_1562_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15230)) := by
  have c95742 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95743 := (publicBatchWrapper89_copies2991 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95744 := (publicBatchWrapper89_copies2992 a h).1
  have c95745 := (publicBatchWrapper89_copies2992 a h).2.1
  have c95746 := (publicBatchWrapper89_copies2992 a h).2.2.1
  have c95747 := (publicBatchWrapper89_copies2992 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_10
  simp only [← c95742, ← c95743, ← c95744] at e_1562_10
  have e_1562_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_11
  simp only [← c95745, ← c95746, k2, ← c95747] at e_1562_11
  have hr := e_1562_11
  simp only [e_1562_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15231)) := by
  have c95748 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.1
  have c95749 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.1
  have c95750 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.1
  have c95751 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.1
  have c95752 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.1
  have c95753 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_12
  simp only [← c95748, ← c95749, ← c95750] at e_1562_12
  have e_1562_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_13
  simp only [← c95751, ← c95752, k2, ← c95753] at e_1562_13
  have hr := e_1562_13
  simp only [e_1562_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15232)) := by
  have c95754 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95755 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95756 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95757 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95758 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95759 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_14
  simp only [← c95754, ← c95755, ← c95756] at e_1562_14
  have e_1562_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_15
  simp only [← c95757, ← c95758, k2, ← c95759] at e_1562_15
  have hr := e_1562_15
  simp only [e_1562_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15233)) := by
  have c95760 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95761 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95762 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95763 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95764 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95765 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_16
  simp only [← c95760, ← c95761, ← c95762] at e_1562_16
  have e_1562_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_17
  simp only [← c95763, ← c95764, k2, ← c95765] at e_1562_17
  have hr := e_1562_17
  simp only [e_1562_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1562 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15234)) := by
  have c95766 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95767 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95768 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95769 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95770 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95771 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1562_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_18
  simp only [← c95766, ← c95767, ← c95768] at e_1562_18
  have e_1562_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1562 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1562_19
  simp only [← c95769, ← c95770, k2, ← c95771] at e_1562_19
  have hr := e_1562_19
  simp only [e_1562_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15235)) := by
  have c95772 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95773 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95774 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95775 := (publicBatchWrapper89_copies2992 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95776 := (publicBatchWrapper89_copies2993 a h).1
  have c95777 := (publicBatchWrapper89_copies2993 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_0
  simp only [← c95772, ← c95773, ← c95774] at e_1563_0
  have e_1563_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_1
  simp only [← c95775, ← c95776, k2, ← c95777] at e_1563_1
  have hr := e_1563_1
  simp only [e_1563_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15236)) := by
  have c95778 := (publicBatchWrapper89_copies2993 a h).2.2.1
  have c95779 := (publicBatchWrapper89_copies2993 a h).2.2.2.1
  have c95780 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.1
  have c95781 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.1
  have c95782 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.1
  have c95783 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_2
  simp only [← c95778, ← c95779, ← c95780] at e_1563_2
  have e_1563_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_3
  simp only [← c95781, ← c95782, k2, ← c95783] at e_1563_3
  have hr := e_1563_3
  simp only [e_1563_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15237)) := by
  have c95784 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.1
  have c95785 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.1
  have c95786 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95787 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95788 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95789 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_4
  simp only [← c95784, ← c95785, ← c95786] at e_1563_4
  have e_1563_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_5
  simp only [← c95787, ← c95788, k2, ← c95789] at e_1563_5
  have hr := e_1563_5
  simp only [e_1563_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15238)) := by
  have c95790 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95791 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95792 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95793 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95794 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95795 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_6
  simp only [← c95790, ← c95791, ← c95792] at e_1563_6
  have e_1563_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_7
  simp only [← c95793, ← c95794, k2, ← c95795] at e_1563_7
  have hr := e_1563_7
  simp only [e_1563_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15239)) := by
  have c95796 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95797 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95798 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95799 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95800 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95801 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_8
  simp only [← c95796, ← c95797, ← c95798] at e_1563_8
  have e_1563_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_9
  simp only [← c95799, ← c95800, k2, ← c95801] at e_1563_9
  have hr := e_1563_9
  simp only [e_1563_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15240)) := by
  have c95802 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95803 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95804 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95805 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95806 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95807 := (publicBatchWrapper89_copies2993 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_10
  simp only [← c95802, ← c95803, ← c95804] at e_1563_10
  have e_1563_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_11
  simp only [← c95805, ← c95806, k2, ← c95807] at e_1563_11
  have hr := e_1563_11
  simp only [e_1563_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15241)) := by
  have c95808 := (publicBatchWrapper89_copies2994 a h).1
  have c95809 := (publicBatchWrapper89_copies2994 a h).2.1
  have c95810 := (publicBatchWrapper89_copies2994 a h).2.2.1
  have c95811 := (publicBatchWrapper89_copies2994 a h).2.2.2.1
  have c95812 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.1
  have c95813 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_12
  simp only [← c95808, ← c95809, ← c95810] at e_1563_12
  have e_1563_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_13
  simp only [← c95811, ← c95812, k2, ← c95813] at e_1563_13
  have hr := e_1563_13
  simp only [e_1563_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15242)) := by
  have c95814 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.1
  have c95815 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.1
  have c95816 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.1
  have c95817 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.1
  have c95818 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95819 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_14
  simp only [← c95814, ← c95815, ← c95816] at e_1563_14
  have e_1563_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_15
  simp only [← c95817, ← c95818, k2, ← c95819] at e_1563_15
  have hr := e_1563_15
  simp only [e_1563_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15243)) := by
  have c95820 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95821 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95822 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95823 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95824 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95825 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_16
  simp only [← c95820, ← c95821, ← c95822] at e_1563_16
  have e_1563_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_17
  simp only [← c95823, ← c95824, k2, ← c95825] at e_1563_17
  have hr := e_1563_17
  simp only [e_1563_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1563 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15244)) := by
  have c95826 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95827 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95828 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95829 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95830 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95831 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1563_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_18
  simp only [← c95826, ← c95827, ← c95828] at e_1563_18
  have e_1563_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1563 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1563_19
  simp only [← c95829, ← c95830, k2, ← c95831] at e_1563_19
  have hr := e_1563_19
  simp only [e_1563_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15245)) := by
  have c95832 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95833 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95834 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95835 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95836 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95837 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_0
  simp only [← c95832, ← c95833, ← c95834] at e_1564_0
  have e_1564_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_1
  simp only [← c95835, ← c95836, k2, ← c95837] at e_1564_1
  have hr := e_1564_1
  simp only [e_1564_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15246)) := by
  have c95838 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95839 := (publicBatchWrapper89_copies2994 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95840 := (publicBatchWrapper89_copies2995 a h).1
  have c95841 := (publicBatchWrapper89_copies2995 a h).2.1
  have c95842 := (publicBatchWrapper89_copies2995 a h).2.2.1
  have c95843 := (publicBatchWrapper89_copies2995 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_2
  simp only [← c95838, ← c95839, ← c95840] at e_1564_2
  have e_1564_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_3
  simp only [← c95841, ← c95842, k2, ← c95843] at e_1564_3
  have hr := e_1564_3
  simp only [e_1564_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15247)) := by
  have c95844 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.1
  have c95845 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.1
  have c95846 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.1
  have c95847 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.1
  have c95848 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.1
  have c95849 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_4
  simp only [← c95844, ← c95845, ← c95846] at e_1564_4
  have e_1564_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_5
  simp only [← c95847, ← c95848, k2, ← c95849] at e_1564_5
  have hr := e_1564_5
  simp only [e_1564_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15248)) := by
  have c95850 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95851 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95852 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95853 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95854 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95855 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_6
  simp only [← c95850, ← c95851, ← c95852] at e_1564_6
  have e_1564_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_7
  simp only [← c95853, ← c95854, k2, ← c95855] at e_1564_7
  have hr := e_1564_7
  simp only [e_1564_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15249)) := by
  have c95856 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95857 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95858 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95859 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95860 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95861 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_8
  simp only [← c95856, ← c95857, ← c95858] at e_1564_8
  have e_1564_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_9
  simp only [← c95859, ← c95860, k2, ← c95861] at e_1564_9
  have hr := e_1564_9
  simp only [e_1564_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15250)) := by
  have c95862 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95863 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95864 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95865 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95866 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95867 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_10
  simp only [← c95862, ← c95863, ← c95864] at e_1564_10
  have e_1564_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_11
  simp only [← c95865, ← c95866, k2, ← c95867] at e_1564_11
  have hr := e_1564_11
  simp only [e_1564_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15251)) := by
  have c95868 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95869 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95870 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95871 := (publicBatchWrapper89_copies2995 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95872 := (publicBatchWrapper89_copies2996 a h).1
  have c95873 := (publicBatchWrapper89_copies2996 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_12
  simp only [← c95868, ← c95869, ← c95870] at e_1564_12
  have e_1564_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_13
  simp only [← c95871, ← c95872, k2, ← c95873] at e_1564_13
  have hr := e_1564_13
  simp only [e_1564_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15252)) := by
  have c95874 := (publicBatchWrapper89_copies2996 a h).2.2.1
  have c95875 := (publicBatchWrapper89_copies2996 a h).2.2.2.1
  have c95876 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.1
  have c95877 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.1
  have c95878 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.1
  have c95879 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_14
  simp only [← c95874, ← c95875, ← c95876] at e_1564_14
  have e_1564_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_15
  simp only [← c95877, ← c95878, k2, ← c95879] at e_1564_15
  have hr := e_1564_15
  simp only [e_1564_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15253)) := by
  have c95880 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.1
  have c95881 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.1
  have c95882 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95883 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95884 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95885 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_16
  simp only [← c95880, ← c95881, ← c95882] at e_1564_16
  have e_1564_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_17
  simp only [← c95883, ← c95884, k2, ← c95885] at e_1564_17
  have hr := e_1564_17
  simp only [e_1564_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1564 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15254)) := by
  have c95886 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95887 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95888 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95889 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95890 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95891 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1564_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_18
  simp only [← c95886, ← c95887, ← c95888] at e_1564_18
  have e_1564_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1564 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1564_19
  simp only [← c95889, ← c95890, k2, ← c95891] at e_1564_19
  have hr := e_1564_19
  simp only [e_1564_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15255)) := by
  have c95892 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95893 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95894 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95895 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95896 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95897 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_0
  simp only [← c95892, ← c95893, ← c95894] at e_1565_0
  have e_1565_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_1
  simp only [← c95895, ← c95896, k2, ← c95897] at e_1565_1
  have hr := e_1565_1
  simp only [e_1565_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15256)) := by
  have c95898 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95899 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95900 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95901 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95902 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95903 := (publicBatchWrapper89_copies2996 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_2
  simp only [← c95898, ← c95899, ← c95900] at e_1565_2
  have e_1565_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_3
  simp only [← c95901, ← c95902, k2, ← c95903] at e_1565_3
  have hr := e_1565_3
  simp only [e_1565_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15257)) := by
  have c95904 := (publicBatchWrapper89_copies2997 a h).1
  have c95905 := (publicBatchWrapper89_copies2997 a h).2.1
  have c95906 := (publicBatchWrapper89_copies2997 a h).2.2.1
  have c95907 := (publicBatchWrapper89_copies2997 a h).2.2.2.1
  have c95908 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.1
  have c95909 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_4
  simp only [← c95904, ← c95905, ← c95906] at e_1565_4
  have e_1565_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_5
  simp only [← c95907, ← c95908, k2, ← c95909] at e_1565_5
  have hr := e_1565_5
  simp only [e_1565_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15258)) := by
  have c95910 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.1
  have c95911 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.1
  have c95912 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.1
  have c95913 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.1
  have c95914 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95915 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_6
  simp only [← c95910, ← c95911, ← c95912] at e_1565_6
  have e_1565_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_7
  simp only [← c95913, ← c95914, k2, ← c95915] at e_1565_7
  have hr := e_1565_7
  simp only [e_1565_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15259)) := by
  have c95916 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95917 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95918 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95919 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95920 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95921 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_8
  simp only [← c95916, ← c95917, ← c95918] at e_1565_8
  have e_1565_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_9
  simp only [← c95919, ← c95920, k2, ← c95921] at e_1565_9
  have hr := e_1565_9
  simp only [e_1565_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15260)) := by
  have c95922 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95923 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95924 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95925 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95926 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95927 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_10
  simp only [← c95922, ← c95923, ← c95924] at e_1565_10
  have e_1565_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_11
  simp only [← c95925, ← c95926, k2, ← c95927] at e_1565_11
  have hr := e_1565_11
  simp only [e_1565_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15261)) := by
  have c95928 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95929 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95930 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95931 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95932 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95933 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_12
  simp only [← c95928, ← c95929, ← c95930] at e_1565_12
  have e_1565_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_13
  simp only [← c95931, ← c95932, k2, ← c95933] at e_1565_13
  have hr := e_1565_13
  simp only [e_1565_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15262)) := by
  have c95934 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95935 := (publicBatchWrapper89_copies2997 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95936 := (publicBatchWrapper89_copies2998 a h).1
  have c95937 := (publicBatchWrapper89_copies2998 a h).2.1
  have c95938 := (publicBatchWrapper89_copies2998 a h).2.2.1
  have c95939 := (publicBatchWrapper89_copies2998 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_14
  simp only [← c95934, ← c95935, ← c95936] at e_1565_14
  have e_1565_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_15
  simp only [← c95937, ← c95938, k2, ← c95939] at e_1565_15
  have hr := e_1565_15
  simp only [e_1565_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15263)) := by
  have c95940 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.1
  have c95941 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.1
  have c95942 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.1
  have c95943 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.1
  have c95944 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.1
  have c95945 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_16
  simp only [← c95940, ← c95941, ← c95942] at e_1565_16
  have e_1565_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_17
  simp only [← c95943, ← c95944, k2, ← c95945] at e_1565_17
  have hr := e_1565_17
  simp only [e_1565_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1565 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15264)) := by
  have c95946 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95947 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95948 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95949 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95950 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95951 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1565_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_18
  simp only [← c95946, ← c95947, ← c95948] at e_1565_18
  have e_1565_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1565 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1565_19
  simp only [← c95949, ← c95950, k2, ← c95951] at e_1565_19
  have hr := e_1565_19
  simp only [e_1565_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15265)) := by
  have c95952 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95953 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95954 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95955 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95956 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95957 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_0
  simp only [← c95952, ← c95953, ← c95954] at e_1566_0
  have e_1566_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_1
  simp only [← c95955, ← c95956, k2, ← c95957] at e_1566_1
  have hr := e_1566_1
  simp only [e_1566_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15266)) := by
  have c95958 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95959 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95960 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95961 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95962 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95963 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_2
  simp only [← c95958, ← c95959, ← c95960] at e_1566_2
  have e_1566_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_3
  simp only [← c95961, ← c95962, k2, ← c95963] at e_1566_3
  have hr := e_1566_3
  simp only [e_1566_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15267)) := by
  have c95964 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95965 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95966 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95967 := (publicBatchWrapper89_copies2998 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c95968 := (publicBatchWrapper89_copies2999 a h).1
  have c95969 := (publicBatchWrapper89_copies2999 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_4
  simp only [← c95964, ← c95965, ← c95966] at e_1566_4
  have e_1566_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_5
  simp only [← c95967, ← c95968, k2, ← c95969] at e_1566_5
  have hr := e_1566_5
  simp only [e_1566_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15268)) := by
  have c95970 := (publicBatchWrapper89_copies2999 a h).2.2.1
  have c95971 := (publicBatchWrapper89_copies2999 a h).2.2.2.1
  have c95972 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.1
  have c95973 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.1
  have c95974 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.1
  have c95975 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_6
  simp only [← c95970, ← c95971, ← c95972] at e_1566_6
  have e_1566_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_7
  simp only [← c95973, ← c95974, k2, ← c95975] at e_1566_7
  have hr := e_1566_7
  simp only [e_1566_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15269)) := by
  have c95976 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.1
  have c95977 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.1
  have c95978 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.1
  have c95979 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c95980 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95981 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_8
  simp only [← c95976, ← c95977, ← c95978] at e_1566_8
  have e_1566_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_9
  simp only [← c95979, ← c95980, k2, ← c95981] at e_1566_9
  have hr := e_1566_9
  simp only [e_1566_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15270)) := by
  have c95982 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95983 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95984 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95985 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95986 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95987 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_10
  simp only [← c95982, ← c95983, ← c95984] at e_1566_10
  have e_1566_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_11
  simp only [← c95985, ← c95986, k2, ← c95987] at e_1566_11
  have hr := e_1566_11
  simp only [e_1566_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15271)) := by
  have c95988 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95989 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95990 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95991 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95992 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95993 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_12
  simp only [← c95988, ← c95989, ← c95990] at e_1566_12
  have e_1566_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_13
  simp only [← c95991, ← c95992, k2, ← c95993] at e_1566_13
  have hr := e_1566_13
  simp only [e_1566_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15272)) := by
  have c95994 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95995 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95996 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95997 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95998 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c95999 := (publicBatchWrapper89_copies2999 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_14
  simp only [← c95994, ← c95995, ← c95996] at e_1566_14
  have e_1566_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_15
  simp only [← c95997, ← c95998, k2, ← c95999] at e_1566_15
  have hr := e_1566_15
  simp only [e_1566_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15273)) := by
  have c96000 := (publicBatchWrapper89_copies3000 a h).1
  have c96001 := (publicBatchWrapper89_copies3000 a h).2.1
  have c96002 := (publicBatchWrapper89_copies3000 a h).2.2.1
  have c96003 := (publicBatchWrapper89_copies3000 a h).2.2.2.1
  have c96004 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.1
  have c96005 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_16
  simp only [← c96000, ← c96001, ← c96002] at e_1566_16
  have e_1566_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_17
  simp only [← c96003, ← c96004, k2, ← c96005] at e_1566_17
  have hr := e_1566_17
  simp only [e_1566_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1566 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15274)) := by
  have c96006 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.1
  have c96007 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.1
  have c96008 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.1
  have c96009 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.1
  have c96010 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96011 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1566_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_18
  simp only [← c96006, ← c96007, ← c96008] at e_1566_18
  have e_1566_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1566 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1566_19
  simp only [← c96009, ← c96010, k2, ← c96011] at e_1566_19
  have hr := e_1566_19
  simp only [e_1566_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15275)) := by
  have c96012 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96013 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96014 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96015 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96016 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96017 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_0
  simp only [← c96012, ← c96013, ← c96014] at e_1567_0
  have e_1567_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_1
  simp only [← c96015, ← c96016, k2, ← c96017] at e_1567_1
  have hr := e_1567_1
  simp only [e_1567_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15276)) := by
  have c96018 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96019 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96020 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96021 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96022 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96023 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_2
  simp only [← c96018, ← c96019, ← c96020] at e_1567_2
  have e_1567_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_3
  simp only [← c96021, ← c96022, k2, ← c96023] at e_1567_3
  have hr := e_1567_3
  simp only [e_1567_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15277)) := by
  have c96024 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96025 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96026 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96027 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96028 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96029 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_4
  simp only [← c96024, ← c96025, ← c96026] at e_1567_4
  have e_1567_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_5
  simp only [← c96027, ← c96028, k2, ← c96029] at e_1567_5
  have hr := e_1567_5
  simp only [e_1567_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15278)) := by
  have c96030 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96031 := (publicBatchWrapper89_copies3000 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96032 := (publicBatchWrapper89_copies3001 a h).1
  have c96033 := (publicBatchWrapper89_copies3001 a h).2.1
  have c96034 := (publicBatchWrapper89_copies3001 a h).2.2.1
  have c96035 := (publicBatchWrapper89_copies3001 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_6
  simp only [← c96030, ← c96031, ← c96032] at e_1567_6
  have e_1567_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_7
  simp only [← c96033, ← c96034, k2, ← c96035] at e_1567_7
  have hr := e_1567_7
  simp only [e_1567_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15279)) := by
  have c96036 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.1
  have c96037 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.1
  have c96038 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.1
  have c96039 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.1
  have c96040 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.1
  have c96041 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_8
  simp only [← c96036, ← c96037, ← c96038] at e_1567_8
  have e_1567_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_9
  simp only [← c96039, ← c96040, k2, ← c96041] at e_1567_9
  have hr := e_1567_9
  simp only [e_1567_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15280)) := by
  have c96042 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96043 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96044 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96045 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96046 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96047 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_10
  simp only [← c96042, ← c96043, ← c96044] at e_1567_10
  have e_1567_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_11
  simp only [← c96045, ← c96046, k2, ← c96047] at e_1567_11
  have hr := e_1567_11
  simp only [e_1567_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15281)) := by
  have c96048 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96049 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96050 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96051 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96052 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96053 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_12
  simp only [← c96048, ← c96049, ← c96050] at e_1567_12
  have e_1567_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_13
  simp only [← c96051, ← c96052, k2, ← c96053] at e_1567_13
  have hr := e_1567_13
  simp only [e_1567_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15282)) := by
  have c96054 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96055 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96056 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96057 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96058 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96059 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_14
  simp only [← c96054, ← c96055, ← c96056] at e_1567_14
  have e_1567_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_15
  simp only [← c96057, ← c96058, k2, ← c96059] at e_1567_15
  have hr := e_1567_15
  simp only [e_1567_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15283)) := by
  have c96060 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96061 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96062 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96063 := (publicBatchWrapper89_copies3001 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96064 := (publicBatchWrapper89_copies3002 a h).1
  have c96065 := (publicBatchWrapper89_copies3002 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_16
  simp only [← c96060, ← c96061, ← c96062] at e_1567_16
  have e_1567_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_17
  simp only [← c96063, ← c96064, k2, ← c96065] at e_1567_17
  have hr := e_1567_17
  simp only [e_1567_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1567 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15284)) := by
  have c96066 := (publicBatchWrapper89_copies3002 a h).2.2.1
  have c96067 := (publicBatchWrapper89_copies3002 a h).2.2.2.1
  have c96068 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.1
  have c96069 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.1
  have c96070 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.1
  have c96071 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1567_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_18
  simp only [← c96066, ← c96067, ← c96068] at e_1567_18
  have e_1567_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1567 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1567_19
  simp only [← c96069, ← c96070, k2, ← c96071] at e_1567_19
  have hr := e_1567_19
  simp only [e_1567_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15285)) := by
  have c96072 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.1
  have c96073 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.1
  have c96074 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96075 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96076 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96077 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_0
  simp only [← c96072, ← c96073, ← c96074] at e_1568_0
  have e_1568_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_1
  simp only [← c96075, ← c96076, k2, ← c96077] at e_1568_1
  have hr := e_1568_1
  simp only [e_1568_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15286)) := by
  have c96078 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96079 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96080 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96081 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96082 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96083 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_2
  simp only [← c96078, ← c96079, ← c96080] at e_1568_2
  have e_1568_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_3
  simp only [← c96081, ← c96082, k2, ← c96083] at e_1568_3
  have hr := e_1568_3
  simp only [e_1568_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15287)) := by
  have c96084 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96085 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96086 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96087 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96088 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96089 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_4
  simp only [← c96084, ← c96085, ← c96086] at e_1568_4
  have e_1568_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_5
  simp only [← c96087, ← c96088, k2, ← c96089] at e_1568_5
  have hr := e_1568_5
  simp only [e_1568_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15288)) := by
  have c96090 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96091 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96092 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96093 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96094 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96095 := (publicBatchWrapper89_copies3002 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_6
  simp only [← c96090, ← c96091, ← c96092] at e_1568_6
  have e_1568_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_7
  simp only [← c96093, ← c96094, k2, ← c96095] at e_1568_7
  have hr := e_1568_7
  simp only [e_1568_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15289)) := by
  have c96096 := (publicBatchWrapper89_copies3003 a h).1
  have c96097 := (publicBatchWrapper89_copies3003 a h).2.1
  have c96098 := (publicBatchWrapper89_copies3003 a h).2.2.1
  have c96099 := (publicBatchWrapper89_copies3003 a h).2.2.2.1
  have c96100 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.1
  have c96101 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_8
  simp only [← c96096, ← c96097, ← c96098] at e_1568_8
  have e_1568_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_9
  simp only [← c96099, ← c96100, k2, ← c96101] at e_1568_9
  have hr := e_1568_9
  simp only [e_1568_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15290)) := by
  have c96102 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.1
  have c96103 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.1
  have c96104 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.1
  have c96105 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.1
  have c96106 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96107 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_10
  simp only [← c96102, ← c96103, ← c96104] at e_1568_10
  have e_1568_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_11
  simp only [← c96105, ← c96106, k2, ← c96107] at e_1568_11
  have hr := e_1568_11
  simp only [e_1568_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15291)) := by
  have c96108 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96109 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96110 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96111 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96112 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96113 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_12
  simp only [← c96108, ← c96109, ← c96110] at e_1568_12
  have e_1568_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_13
  simp only [← c96111, ← c96112, k2, ← c96113] at e_1568_13
  have hr := e_1568_13
  simp only [e_1568_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15292)) := by
  have c96114 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96115 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96116 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96117 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96118 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96119 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_14
  simp only [← c96114, ← c96115, ← c96116] at e_1568_14
  have e_1568_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_15
  simp only [← c96117, ← c96118, k2, ← c96119] at e_1568_15
  have hr := e_1568_15
  simp only [e_1568_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15293)) := by
  have c96120 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96121 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96122 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96123 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96124 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96125 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_16
  simp only [← c96120, ← c96121, ← c96122] at e_1568_16
  have e_1568_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_17
  simp only [← c96123, ← c96124, k2, ← c96125] at e_1568_17
  have hr := e_1568_17
  simp only [e_1568_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1568 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15294)) := by
  have c96126 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96127 := (publicBatchWrapper89_copies3003 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96128 := (publicBatchWrapper89_copies3004 a h).1
  have c96129 := (publicBatchWrapper89_copies3004 a h).2.1
  have c96130 := (publicBatchWrapper89_copies3004 a h).2.2.1
  have c96131 := (publicBatchWrapper89_copies3004 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1568_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_18
  simp only [← c96126, ← c96127, ← c96128] at e_1568_18
  have e_1568_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1568 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1568_19
  simp only [← c96129, ← c96130, k2, ← c96131] at e_1568_19
  have hr := e_1568_19
  simp only [e_1568_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15295)) := by
  have c96132 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.1
  have c96133 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.1
  have c96134 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.1
  have c96135 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.1
  have c96136 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.1
  have c96137 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_0
  simp only [← c96132, ← c96133, ← c96134] at e_1569_0
  have e_1569_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_1
  simp only [← c96135, ← c96136, k2, ← c96137] at e_1569_1
  have hr := e_1569_1
  simp only [e_1569_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15296)) := by
  have c96138 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96139 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96140 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96141 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96142 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96143 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_2
  simp only [← c96138, ← c96139, ← c96140] at e_1569_2
  have e_1569_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_3
  simp only [← c96141, ← c96142, k2, ← c96143] at e_1569_3
  have hr := e_1569_3
  simp only [e_1569_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15297)) := by
  have c96144 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96145 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96146 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96147 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96148 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96149 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_4
  simp only [← c96144, ← c96145, ← c96146] at e_1569_4
  have e_1569_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_5
  simp only [← c96147, ← c96148, k2, ← c96149] at e_1569_5
  have hr := e_1569_5
  simp only [e_1569_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15298)) := by
  have c96150 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96151 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96152 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96153 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96154 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96155 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_6
  simp only [← c96150, ← c96151, ← c96152] at e_1569_6
  have e_1569_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_7
  simp only [← c96153, ← c96154, k2, ← c96155] at e_1569_7
  have hr := e_1569_7
  simp only [e_1569_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15299)) := by
  have c96156 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96157 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96158 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96159 := (publicBatchWrapper89_copies3004 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96160 := (publicBatchWrapper89_copies3005 a h).1
  have c96161 := (publicBatchWrapper89_copies3005 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_8
  simp only [← c96156, ← c96157, ← c96158] at e_1569_8
  have e_1569_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_9
  simp only [← c96159, ← c96160, k2, ← c96161] at e_1569_9
  have hr := e_1569_9
  simp only [e_1569_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15300)) := by
  have c96162 := (publicBatchWrapper89_copies3005 a h).2.2.1
  have c96163 := (publicBatchWrapper89_copies3005 a h).2.2.2.1
  have c96164 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.1
  have c96165 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.1
  have c96166 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.1
  have c96167 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_10
  simp only [← c96162, ← c96163, ← c96164] at e_1569_10
  have e_1569_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_11
  simp only [← c96165, ← c96166, k2, ← c96167] at e_1569_11
  have hr := e_1569_11
  simp only [e_1569_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15301)) := by
  have c96168 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.1
  have c96169 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.1
  have c96170 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96171 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96172 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96173 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_12
  simp only [← c96168, ← c96169, ← c96170] at e_1569_12
  have e_1569_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_13
  simp only [← c96171, ← c96172, k2, ← c96173] at e_1569_13
  have hr := e_1569_13
  simp only [e_1569_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15302)) := by
  have c96174 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96175 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96176 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96177 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96178 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96179 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_14
  simp only [← c96174, ← c96175, ← c96176] at e_1569_14
  have e_1569_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_15
  simp only [← c96177, ← c96178, k2, ← c96179] at e_1569_15
  have hr := e_1569_15
  simp only [e_1569_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15303)) := by
  have c96180 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96181 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96182 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96183 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96184 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96185 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_16
  simp only [← c96180, ← c96181, ← c96182] at e_1569_16
  have e_1569_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_17
  simp only [← c96183, ← c96184, k2, ← c96185] at e_1569_17
  have hr := e_1569_17
  simp only [e_1569_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1569 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15304)) := by
  have c96186 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96187 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96188 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96189 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96190 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96191 := (publicBatchWrapper89_copies3005 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1569_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_18
  simp only [← c96186, ← c96187, ← c96188] at e_1569_18
  have e_1569_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1569 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1569_19
  simp only [← c96189, ← c96190, k2, ← c96191] at e_1569_19
  have hr := e_1569_19
  simp only [e_1569_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15305)) := by
  have c96192 := (publicBatchWrapper89_copies3006 a h).1
  have c96193 := (publicBatchWrapper89_copies3006 a h).2.1
  have c96194 := (publicBatchWrapper89_copies3006 a h).2.2.1
  have c96195 := (publicBatchWrapper89_copies3006 a h).2.2.2.1
  have c96196 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.1
  have c96197 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_0
  simp only [← c96192, ← c96193, ← c96194] at e_1570_0
  have e_1570_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_1
  simp only [← c96195, ← c96196, k2, ← c96197] at e_1570_1
  have hr := e_1570_1
  simp only [e_1570_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15306)) := by
  have c96198 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.1
  have c96199 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.1
  have c96200 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.1
  have c96201 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.1
  have c96202 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96203 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_2
  simp only [← c96198, ← c96199, ← c96200] at e_1570_2
  have e_1570_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_3
  simp only [← c96201, ← c96202, k2, ← c96203] at e_1570_3
  have hr := e_1570_3
  simp only [e_1570_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15307)) := by
  have c96204 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96205 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96206 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96207 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96208 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96209 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_4
  simp only [← c96204, ← c96205, ← c96206] at e_1570_4
  have e_1570_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_5
  simp only [← c96207, ← c96208, k2, ← c96209] at e_1570_5
  have hr := e_1570_5
  simp only [e_1570_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15308)) := by
  have c96210 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96211 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96212 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96213 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96214 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96215 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_6
  simp only [← c96210, ← c96211, ← c96212] at e_1570_6
  have e_1570_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_7
  simp only [← c96213, ← c96214, k2, ← c96215] at e_1570_7
  have hr := e_1570_7
  simp only [e_1570_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15309)) := by
  have c96216 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96217 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96218 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96219 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96220 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96221 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_8
  simp only [← c96216, ← c96217, ← c96218] at e_1570_8
  have e_1570_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_9
  simp only [← c96219, ← c96220, k2, ← c96221] at e_1570_9
  have hr := e_1570_9
  simp only [e_1570_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15310)) := by
  have c96222 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96223 := (publicBatchWrapper89_copies3006 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96224 := (publicBatchWrapper89_copies3007 a h).1
  have c96225 := (publicBatchWrapper89_copies3007 a h).2.1
  have c96226 := (publicBatchWrapper89_copies3007 a h).2.2.1
  have c96227 := (publicBatchWrapper89_copies3007 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_10
  simp only [← c96222, ← c96223, ← c96224] at e_1570_10
  have e_1570_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_11
  simp only [← c96225, ← c96226, k2, ← c96227] at e_1570_11
  have hr := e_1570_11
  simp only [e_1570_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15311)) := by
  have c96228 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.1
  have c96229 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.1
  have c96230 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.1
  have c96231 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.1
  have c96232 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.1
  have c96233 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_12
  simp only [← c96228, ← c96229, ← c96230] at e_1570_12
  have e_1570_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_13
  simp only [← c96231, ← c96232, k2, ← c96233] at e_1570_13
  have hr := e_1570_13
  simp only [e_1570_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15312)) := by
  have c96234 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96235 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96236 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96237 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96238 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96239 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_14
  simp only [← c96234, ← c96235, ← c96236] at e_1570_14
  have e_1570_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_15
  simp only [← c96237, ← c96238, k2, ← c96239] at e_1570_15
  have hr := e_1570_15
  simp only [e_1570_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15313)) := by
  have c96240 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96241 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96242 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96243 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96244 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96245 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_16
  simp only [← c96240, ← c96241, ← c96242] at e_1570_16
  have e_1570_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_17
  simp only [← c96243, ← c96244, k2, ← c96245] at e_1570_17
  have hr := e_1570_17
  simp only [e_1570_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1570 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15314)) := by
  have c96246 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96247 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96248 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96249 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96250 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96251 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1570_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_18
  simp only [← c96246, ← c96247, ← c96248] at e_1570_18
  have e_1570_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1570 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1570_19
  simp only [← c96249, ← c96250, k2, ← c96251] at e_1570_19
  have hr := e_1570_19
  simp only [e_1570_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15315)) := by
  have c96252 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96253 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96254 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96255 := (publicBatchWrapper89_copies3007 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96256 := (publicBatchWrapper89_copies3008 a h).1
  have c96257 := (publicBatchWrapper89_copies3008 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_0
  simp only [← c96252, ← c96253, ← c96254] at e_1571_0
  have e_1571_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_1
  simp only [← c96255, ← c96256, k2, ← c96257] at e_1571_1
  have hr := e_1571_1
  simp only [e_1571_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15316)) := by
  have c96258 := (publicBatchWrapper89_copies3008 a h).2.2.1
  have c96259 := (publicBatchWrapper89_copies3008 a h).2.2.2.1
  have c96260 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.1
  have c96261 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.1
  have c96262 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.1
  have c96263 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_2
  simp only [← c96258, ← c96259, ← c96260] at e_1571_2
  have e_1571_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_3
  simp only [← c96261, ← c96262, k2, ← c96263] at e_1571_3
  have hr := e_1571_3
  simp only [e_1571_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15317)) := by
  have c96264 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.1
  have c96265 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.1
  have c96266 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96267 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96268 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96269 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_4
  simp only [← c96264, ← c96265, ← c96266] at e_1571_4
  have e_1571_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_5
  simp only [← c96267, ← c96268, k2, ← c96269] at e_1571_5
  have hr := e_1571_5
  simp only [e_1571_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15318)) := by
  have c96270 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96271 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96272 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96273 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96274 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96275 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_6
  simp only [← c96270, ← c96271, ← c96272] at e_1571_6
  have e_1571_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_7
  simp only [← c96273, ← c96274, k2, ← c96275] at e_1571_7
  have hr := e_1571_7
  simp only [e_1571_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15319)) := by
  have c96276 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96277 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96278 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96279 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96280 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96281 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_8
  simp only [← c96276, ← c96277, ← c96278] at e_1571_8
  have e_1571_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_9
  simp only [← c96279, ← c96280, k2, ← c96281] at e_1571_9
  have hr := e_1571_9
  simp only [e_1571_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15320)) := by
  have c96282 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96283 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96284 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96285 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96286 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96287 := (publicBatchWrapper89_copies3008 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_10
  simp only [← c96282, ← c96283, ← c96284] at e_1571_10
  have e_1571_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_11
  simp only [← c96285, ← c96286, k2, ← c96287] at e_1571_11
  have hr := e_1571_11
  simp only [e_1571_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15321)) := by
  have c96288 := (publicBatchWrapper89_copies3009 a h).1
  have c96289 := (publicBatchWrapper89_copies3009 a h).2.1
  have c96290 := (publicBatchWrapper89_copies3009 a h).2.2.1
  have c96291 := (publicBatchWrapper89_copies3009 a h).2.2.2.1
  have c96292 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.1
  have c96293 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_12
  simp only [← c96288, ← c96289, ← c96290] at e_1571_12
  have e_1571_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_13
  simp only [← c96291, ← c96292, k2, ← c96293] at e_1571_13
  have hr := e_1571_13
  simp only [e_1571_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15322)) := by
  have c96294 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.1
  have c96295 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.1
  have c96296 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.1
  have c96297 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.1
  have c96298 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96299 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_14
  simp only [← c96294, ← c96295, ← c96296] at e_1571_14
  have e_1571_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_15
  simp only [← c96297, ← c96298, k2, ← c96299] at e_1571_15
  have hr := e_1571_15
  simp only [e_1571_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15323)) := by
  have c96300 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96301 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96302 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96303 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96304 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96305 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_16
  simp only [← c96300, ← c96301, ← c96302] at e_1571_16
  have e_1571_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_17
  simp only [← c96303, ← c96304, k2, ← c96305] at e_1571_17
  have hr := e_1571_17
  simp only [e_1571_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1571 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15324)) := by
  have c96306 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96307 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96308 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96309 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96310 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96311 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1571_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_18
  simp only [← c96306, ← c96307, ← c96308] at e_1571_18
  have e_1571_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1571 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1571_19
  simp only [← c96309, ← c96310, k2, ← c96311] at e_1571_19
  have hr := e_1571_19
  simp only [e_1571_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15325)) := by
  have c96312 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96313 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96314 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96315 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96316 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96317 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_0
  simp only [← c96312, ← c96313, ← c96314] at e_1572_0
  have e_1572_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_1
  simp only [← c96315, ← c96316, k2, ← c96317] at e_1572_1
  have hr := e_1572_1
  simp only [e_1572_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15326)) := by
  have c96318 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96319 := (publicBatchWrapper89_copies3009 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96320 := (publicBatchWrapper89_copies3010 a h).1
  have c96321 := (publicBatchWrapper89_copies3010 a h).2.1
  have c96322 := (publicBatchWrapper89_copies3010 a h).2.2.1
  have c96323 := (publicBatchWrapper89_copies3010 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_2
  simp only [← c96318, ← c96319, ← c96320] at e_1572_2
  have e_1572_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_3
  simp only [← c96321, ← c96322, k2, ← c96323] at e_1572_3
  have hr := e_1572_3
  simp only [e_1572_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15327)) := by
  have c96324 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.1
  have c96325 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.1
  have c96326 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.1
  have c96327 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.1
  have c96328 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.1
  have c96329 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_4
  simp only [← c96324, ← c96325, ← c96326] at e_1572_4
  have e_1572_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_5
  simp only [← c96327, ← c96328, k2, ← c96329] at e_1572_5
  have hr := e_1572_5
  simp only [e_1572_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15328)) := by
  have c96330 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96331 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96332 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96333 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96334 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96335 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_6
  simp only [← c96330, ← c96331, ← c96332] at e_1572_6
  have e_1572_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_7
  simp only [← c96333, ← c96334, k2, ← c96335] at e_1572_7
  have hr := e_1572_7
  simp only [e_1572_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15329)) := by
  have c96336 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96337 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96338 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96339 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96340 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96341 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_8
  simp only [← c96336, ← c96337, ← c96338] at e_1572_8
  have e_1572_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_9
  simp only [← c96339, ← c96340, k2, ← c96341] at e_1572_9
  have hr := e_1572_9
  simp only [e_1572_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15330)) := by
  have c96342 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96343 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96344 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96345 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96346 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96347 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_10
  simp only [← c96342, ← c96343, ← c96344] at e_1572_10
  have e_1572_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_11
  simp only [← c96345, ← c96346, k2, ← c96347] at e_1572_11
  have hr := e_1572_11
  simp only [e_1572_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15331)) := by
  have c96348 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96349 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96350 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96351 := (publicBatchWrapper89_copies3010 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96352 := (publicBatchWrapper89_copies3011 a h).1
  have c96353 := (publicBatchWrapper89_copies3011 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_12
  simp only [← c96348, ← c96349, ← c96350] at e_1572_12
  have e_1572_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_13
  simp only [← c96351, ← c96352, k2, ← c96353] at e_1572_13
  have hr := e_1572_13
  simp only [e_1572_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15332)) := by
  have c96354 := (publicBatchWrapper89_copies3011 a h).2.2.1
  have c96355 := (publicBatchWrapper89_copies3011 a h).2.2.2.1
  have c96356 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.1
  have c96357 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.1
  have c96358 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.1
  have c96359 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_14
  simp only [← c96354, ← c96355, ← c96356] at e_1572_14
  have e_1572_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_15
  simp only [← c96357, ← c96358, k2, ← c96359] at e_1572_15
  have hr := e_1572_15
  simp only [e_1572_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15333)) := by
  have c96360 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.1
  have c96361 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.1
  have c96362 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96363 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96364 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96365 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_16
  simp only [← c96360, ← c96361, ← c96362] at e_1572_16
  have e_1572_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_17
  simp only [← c96363, ← c96364, k2, ← c96365] at e_1572_17
  have hr := e_1572_17
  simp only [e_1572_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1572 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15334)) := by
  have c96366 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96367 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96368 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96369 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96370 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96371 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1572_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_18
  simp only [← c96366, ← c96367, ← c96368] at e_1572_18
  have e_1572_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1572 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1572_19
  simp only [← c96369, ← c96370, k2, ← c96371] at e_1572_19
  have hr := e_1572_19
  simp only [e_1572_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15335)) := by
  have c96372 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96373 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96374 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96375 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96376 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96377 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_0
  simp only [← c96372, ← c96373, ← c96374] at e_1573_0
  have e_1573_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_1
  simp only [← c96375, ← c96376, k2, ← c96377] at e_1573_1
  have hr := e_1573_1
  simp only [e_1573_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15336)) := by
  have c96378 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96379 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96380 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96381 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96382 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96383 := (publicBatchWrapper89_copies3011 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_2
  simp only [← c96378, ← c96379, ← c96380] at e_1573_2
  have e_1573_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_3
  simp only [← c96381, ← c96382, k2, ← c96383] at e_1573_3
  have hr := e_1573_3
  simp only [e_1573_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15337)) := by
  have c96384 := (publicBatchWrapper89_copies3012 a h).1
  have c96385 := (publicBatchWrapper89_copies3012 a h).2.1
  have c96386 := (publicBatchWrapper89_copies3012 a h).2.2.1
  have c96387 := (publicBatchWrapper89_copies3012 a h).2.2.2.1
  have c96388 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.1
  have c96389 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_4
  simp only [← c96384, ← c96385, ← c96386] at e_1573_4
  have e_1573_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_5
  simp only [← c96387, ← c96388, k2, ← c96389] at e_1573_5
  have hr := e_1573_5
  simp only [e_1573_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15338)) := by
  have c96390 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.1
  have c96391 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.1
  have c96392 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.1
  have c96393 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.1
  have c96394 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96395 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_6
  simp only [← c96390, ← c96391, ← c96392] at e_1573_6
  have e_1573_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_7
  simp only [← c96393, ← c96394, k2, ← c96395] at e_1573_7
  have hr := e_1573_7
  simp only [e_1573_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15339)) := by
  have c96396 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96397 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96398 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96399 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96400 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96401 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_8
  simp only [← c96396, ← c96397, ← c96398] at e_1573_8
  have e_1573_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_9
  simp only [← c96399, ← c96400, k2, ← c96401] at e_1573_9
  have hr := e_1573_9
  simp only [e_1573_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15340)) := by
  have c96402 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96403 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96404 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96405 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96406 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96407 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_10
  simp only [← c96402, ← c96403, ← c96404] at e_1573_10
  have e_1573_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_11
  simp only [← c96405, ← c96406, k2, ← c96407] at e_1573_11
  have hr := e_1573_11
  simp only [e_1573_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15341)) := by
  have c96408 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96409 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96410 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96411 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96412 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96413 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_12
  simp only [← c96408, ← c96409, ← c96410] at e_1573_12
  have e_1573_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_13
  simp only [← c96411, ← c96412, k2, ← c96413] at e_1573_13
  have hr := e_1573_13
  simp only [e_1573_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15318 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15342)) := by
  have c96414 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96415 := (publicBatchWrapper89_copies3012 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96416 := (publicBatchWrapper89_copies3013 a h).1
  have c96417 := (publicBatchWrapper89_copies3013 a h).2.1
  have c96418 := (publicBatchWrapper89_copies3013 a h).2.2.1
  have c96419 := (publicBatchWrapper89_copies3013 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_14
  simp only [← c96414, ← c96415, ← c96416] at e_1573_14
  have e_1573_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_15
  simp only [← c96417, ← c96418, k2, ← c96419] at e_1573_15
  have hr := e_1573_15
  simp only [e_1573_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15319 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15343)) := by
  have c96420 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.1
  have c96421 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.1
  have c96422 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.1
  have c96423 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.1
  have c96424 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.1
  have c96425 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_16
  simp only [← c96420, ← c96421, ← c96422] at e_1573_16
  have e_1573_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_17
  simp only [← c96423, ← c96424, k2, ← c96425] at e_1573_17
  have hr := e_1573_17
  simp only [e_1573_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15320 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1573 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15344)) := by
  have c96426 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96427 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96428 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96429 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96430 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96431 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1573_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_18
  simp only [← c96426, ← c96427, ← c96428] at e_1573_18
  have e_1573_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1573 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1573_19
  simp only [← c96429, ← c96430, k2, ← c96431] at e_1573_19
  have hr := e_1573_19
  simp only [e_1573_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15321 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15345)) := by
  have c96432 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96433 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96434 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96435 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96436 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96437 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_0
  simp only [← c96432, ← c96433, ← c96434] at e_1574_0
  have e_1574_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_1
  simp only [← c96435, ← c96436, k2, ← c96437] at e_1574_1
  have hr := e_1574_1
  simp only [e_1574_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15322 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15346)) := by
  have c96438 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96439 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96440 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96441 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96442 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96443 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_2
  simp only [← c96438, ← c96439, ← c96440] at e_1574_2
  have e_1574_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_3
  simp only [← c96441, ← c96442, k2, ← c96443] at e_1574_3
  have hr := e_1574_3
  simp only [e_1574_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15323 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15347)) := by
  have c96444 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96445 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96446 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96447 := (publicBatchWrapper89_copies3013 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96448 := (publicBatchWrapper89_copies3014 a h).1
  have c96449 := (publicBatchWrapper89_copies3014 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_4
  simp only [← c96444, ← c96445, ← c96446] at e_1574_4
  have e_1574_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_5
  simp only [← c96447, ← c96448, k2, ← c96449] at e_1574_5
  have hr := e_1574_5
  simp only [e_1574_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15324 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15348)) := by
  have c96450 := (publicBatchWrapper89_copies3014 a h).2.2.1
  have c96451 := (publicBatchWrapper89_copies3014 a h).2.2.2.1
  have c96452 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.1
  have c96453 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.1
  have c96454 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.1
  have c96455 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_6
  simp only [← c96450, ← c96451, ← c96452] at e_1574_6
  have e_1574_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_7
  simp only [← c96453, ← c96454, k2, ← c96455] at e_1574_7
  have hr := e_1574_7
  simp only [e_1574_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15325 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15349)) := by
  have c96456 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.1
  have c96457 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.1
  have c96458 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96459 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96460 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96461 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_8
  simp only [← c96456, ← c96457, ← c96458] at e_1574_8
  have e_1574_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_9
  simp only [← c96459, ← c96460, k2, ← c96461] at e_1574_9
  have hr := e_1574_9
  simp only [e_1574_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15326 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15350)) := by
  have c96462 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96463 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96464 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96465 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96466 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96467 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_10
  simp only [← c96462, ← c96463, ← c96464] at e_1574_10
  have e_1574_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_11
  simp only [← c96465, ← c96466, k2, ← c96467] at e_1574_11
  have hr := e_1574_11
  simp only [e_1574_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15327 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15351)) := by
  have c96468 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96469 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96470 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96471 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96472 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96473 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_12
  simp only [← c96468, ← c96469, ← c96470] at e_1574_12
  have e_1574_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_13
  simp only [← c96471, ← c96472, k2, ← c96473] at e_1574_13
  have hr := e_1574_13
  simp only [e_1574_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15328 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15352)) := by
  have c96474 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96475 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96476 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96477 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96478 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96479 := (publicBatchWrapper89_copies3014 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_14
  simp only [← c96474, ← c96475, ← c96476] at e_1574_14
  have e_1574_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_15
  simp only [← c96477, ← c96478, k2, ← c96479] at e_1574_15
  have hr := e_1574_15
  simp only [e_1574_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15329 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15353)) := by
  have c96480 := (publicBatchWrapper89_copies3015 a h).1
  have c96481 := (publicBatchWrapper89_copies3015 a h).2.1
  have c96482 := (publicBatchWrapper89_copies3015 a h).2.2.1
  have c96483 := (publicBatchWrapper89_copies3015 a h).2.2.2.1
  have c96484 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.1
  have c96485 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_16
  simp only [← c96480, ← c96481, ← c96482] at e_1574_16
  have e_1574_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_17
  simp only [← c96483, ← c96484, k2, ← c96485] at e_1574_17
  have hr := e_1574_17
  simp only [e_1574_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15330 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1574 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15354)) := by
  have c96486 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.1
  have c96487 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.1
  have c96488 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.1
  have c96489 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.1
  have c96490 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96491 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1574_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_18
  simp only [← c96486, ← c96487, ← c96488] at e_1574_18
  have e_1574_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1574 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1574_19
  simp only [← c96489, ← c96490, k2, ← c96491] at e_1574_19
  have hr := e_1574_19
  simp only [e_1574_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15331 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15355)) := by
  have c96492 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96493 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96494 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96495 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96496 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96497 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_0
  simp only [← c96492, ← c96493, ← c96494] at e_1575_0
  have e_1575_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_1
  simp only [← c96495, ← c96496, k2, ← c96497] at e_1575_1
  have hr := e_1575_1
  simp only [e_1575_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15332 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15356)) := by
  have c96498 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96499 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96500 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96501 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96502 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96503 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_2
  simp only [← c96498, ← c96499, ← c96500] at e_1575_2
  have e_1575_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_3
  simp only [← c96501, ← c96502, k2, ← c96503] at e_1575_3
  have hr := e_1575_3
  simp only [e_1575_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15333 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15357)) := by
  have c96504 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96505 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96506 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96507 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96508 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96509 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_4
  simp only [← c96504, ← c96505, ← c96506] at e_1575_4
  have e_1575_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_5
  simp only [← c96507, ← c96508, k2, ← c96509] at e_1575_5
  have hr := e_1575_5
  simp only [e_1575_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15334 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15358)) := by
  have c96510 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96511 := (publicBatchWrapper89_copies3015 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96512 := (publicBatchWrapper89_copies3016 a h).1
  have c96513 := (publicBatchWrapper89_copies3016 a h).2.1
  have c96514 := (publicBatchWrapper89_copies3016 a h).2.2.1
  have c96515 := (publicBatchWrapper89_copies3016 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_6
  simp only [← c96510, ← c96511, ← c96512] at e_1575_6
  have e_1575_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_7
  simp only [← c96513, ← c96514, k2, ← c96515] at e_1575_7
  have hr := e_1575_7
  simp only [e_1575_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15335 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15359)) := by
  have c96516 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.1
  have c96517 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.1
  have c96518 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.1
  have c96519 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.1
  have c96520 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.1
  have c96521 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_8
  simp only [← c96516, ← c96517, ← c96518] at e_1575_8
  have e_1575_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_9
  simp only [← c96519, ← c96520, k2, ← c96521] at e_1575_9
  have hr := e_1575_9
  simp only [e_1575_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15336 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 47) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15360)) := by
  have c96522 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96523 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96524 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96525 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96526 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96527 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_10
  simp only [← c96522, ← c96523, ← c96524] at e_1575_10
  have e_1575_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_11
  simp only [← c96525, ← c96526, k2, ← c96527] at e_1575_11
  have hr := e_1575_11
  simp only [e_1575_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15337 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 55) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15361)) := by
  have c96528 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96529 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96530 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96531 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96532 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96533 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_12
  simp only [← c96528, ← c96529, ← c96530] at e_1575_12
  have e_1575_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_13
  simp only [← c96531, ← c96532, k2, ← c96533] at e_1575_13
  have hr := e_1575_13
  simp only [e_1575_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15338 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 63) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15362)) := by
  have c96534 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96535 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96536 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96537 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96538 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96539 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_14
  simp only [← c96534, ← c96535, ← c96536] at e_1575_14
  have e_1575_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_15
  simp only [← c96537, ← c96538, k2, ← c96539] at e_1575_15
  have hr := e_1575_15
  simp only [e_1575_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15339 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 71) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15363)) := by
  have c96540 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96541 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96542 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96543 := (publicBatchWrapper89_copies3016 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96544 := (publicBatchWrapper89_copies3017 a h).1
  have c96545 := (publicBatchWrapper89_copies3017 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_16
  simp only [← c96540, ← c96541, ← c96542] at e_1575_16
  have e_1575_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_17
  simp only [← c96543, ← c96544, k2, ← c96545] at e_1575_17
  have hr := e_1575_17
  simp only [e_1575_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15340 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1575 79) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15364)) := by
  have c96546 := (publicBatchWrapper89_copies3017 a h).2.2.1
  have c96547 := (publicBatchWrapper89_copies3017 a h).2.2.2.1
  have c96548 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.1
  have c96549 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.1
  have c96550 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.1
  have c96551 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1575_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_18
  simp only [← c96546, ← c96547, ← c96548] at e_1575_18
  have e_1575_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1575 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1575_19
  simp only [← c96549, ← c96550, k2, ← c96551] at e_1575_19
  have hr := e_1575_19
  simp only [e_1575_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15341 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 7) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15365)) := by
  have c96552 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.1
  have c96553 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.1
  have c96554 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96555 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96556 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96557 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_0
  simp only [← c96552, ← c96553, ← c96554] at e_1576_0
  have e_1576_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_1
  simp only [← c96555, ← c96556, k2, ← c96557] at e_1576_1
  have hr := e_1576_1
  simp only [e_1576_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15342 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 15) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15366)) := by
  have c96558 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96559 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96560 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96561 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96562 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96563 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_2
  simp only [← c96558, ← c96559, ← c96560] at e_1576_2
  have e_1576_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_3
  simp only [← c96561, ← c96562, k2, ← c96563] at e_1576_3
  have hr := e_1576_3
  simp only [e_1576_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15343 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 23) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15367)) := by
  have c96564 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96565 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96566 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96567 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96568 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96569 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_4
  simp only [← c96564, ← c96565, ← c96566] at e_1576_4
  have e_1576_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_5
  simp only [← c96567, ← c96568, k2, ← c96569] at e_1576_5
  have hr := e_1576_5
  simp only [e_1576_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15344 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 31) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15368)) := by
  have c96570 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96571 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96572 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96573 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96574 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96575 := (publicBatchWrapper89_copies3017 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_6
  simp only [← c96570, ← c96571, ← c96572] at e_1576_6
  have e_1576_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_7
  simp only [← c96573, ← c96574, k2, ← c96575] at e_1576_7
  have hr := e_1576_7
  simp only [e_1576_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15345 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 39) = bselect (a (.wire 74 43)) (a (.virt 200)) (a (.virt 15369)) := by
  have c96576 := (publicBatchWrapper89_copies3018 a h).1
  have c96577 := (publicBatchWrapper89_copies3018 a h).2.1
  have c96578 := (publicBatchWrapper89_copies3018 a h).2.2.1
  have c96579 := (publicBatchWrapper89_copies3018 a h).2.2.2.1
  have c96580 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.1
  have c96581 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_8
  simp only [← c96576, ← c96577, ← c96578] at e_1576_8
  have e_1576_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_9
  simp only [← c96579, ← c96580, k2, ← c96581] at e_1576_9
  have hr := e_1576_9
  simp only [e_1576_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15346 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15410)) := by
  have c96582 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.1
  have c96583 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.1
  have c96584 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.1
  have c96585 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.1
  have c96586 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96587 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_10
  simp only [← c96582, ← c96583, ← c96584] at e_1576_10
  have e_1576_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_11
  simp only [← c96585, ← c96586, k2, ← c96587] at e_1576_11
  have hr := e_1576_11
  simp only [e_1576_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15347 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15411)) := by
  have c96588 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96589 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96590 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96591 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96592 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96593 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_12
  simp only [← c96588, ← c96589, ← c96590] at e_1576_12
  have e_1576_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_13
  simp only [← c96591, ← c96592, k2, ← c96593] at e_1576_13
  have hr := e_1576_13
  simp only [e_1576_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15348 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15412)) := by
  have c96594 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96595 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96596 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96597 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96598 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96599 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_14
  simp only [← c96594, ← c96595, ← c96596] at e_1576_14
  have e_1576_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_15
  simp only [← c96597, ← c96598, k2, ← c96599] at e_1576_15
  have hr := e_1576_15
  simp only [e_1576_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15349 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15413)) := by
  have c96600 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96601 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96602 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96603 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96604 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96605 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_16
  simp only [← c96600, ← c96601, ← c96602] at e_1576_16
  have e_1576_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_17
  simp only [← c96603, ← c96604, k2, ← c96605] at e_1576_17
  have hr := e_1576_17
  simp only [e_1576_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15350 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1576 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15414)) := by
  have c96606 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96607 := (publicBatchWrapper89_copies3018 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96608 := (publicBatchWrapper89_copies3019 a h).1
  have c96609 := (publicBatchWrapper89_copies3019 a h).2.1
  have c96610 := (publicBatchWrapper89_copies3019 a h).2.2.1
  have c96611 := (publicBatchWrapper89_copies3019 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1576_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_18
  simp only [← c96606, ← c96607, ← c96608] at e_1576_18
  have e_1576_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1576 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1576_19
  simp only [← c96609, ← c96610, k2, ← c96611] at e_1576_19
  have hr := e_1576_19
  simp only [e_1576_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15351 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15415)) := by
  have c96612 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.1
  have c96613 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.1
  have c96614 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.1
  have c96615 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.1
  have c96616 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.1
  have c96617 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_0
  simp only [← c96612, ← c96613, ← c96614] at e_1577_0
  have e_1577_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_1
  simp only [← c96615, ← c96616, k2, ← c96617] at e_1577_1
  have hr := e_1577_1
  simp only [e_1577_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15352 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15416)) := by
  have c96618 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96619 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96620 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96621 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96622 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96623 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_2
  simp only [← c96618, ← c96619, ← c96620] at e_1577_2
  have e_1577_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_3
  simp only [← c96621, ← c96622, k2, ← c96623] at e_1577_3
  have hr := e_1577_3
  simp only [e_1577_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15353 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15417)) := by
  have c96624 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96625 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96626 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96627 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96628 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96629 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_4
  simp only [← c96624, ← c96625, ← c96626] at e_1577_4
  have e_1577_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_5
  simp only [← c96627, ← c96628, k2, ← c96629] at e_1577_5
  have hr := e_1577_5
  simp only [e_1577_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15354 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15418)) := by
  have c96630 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96631 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96632 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96633 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96634 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96635 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_6
  simp only [← c96630, ← c96631, ← c96632] at e_1577_6
  have e_1577_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_7
  simp only [← c96633, ← c96634, k2, ← c96635] at e_1577_7
  have hr := e_1577_7
  simp only [e_1577_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15355 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15419)) := by
  have c96636 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96637 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96638 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96639 := (publicBatchWrapper89_copies3019 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96640 := (publicBatchWrapper89_copies3020 a h).1
  have c96641 := (publicBatchWrapper89_copies3020 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_8
  simp only [← c96636, ← c96637, ← c96638] at e_1577_8
  have e_1577_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_9
  simp only [← c96639, ← c96640, k2, ← c96641] at e_1577_9
  have hr := e_1577_9
  simp only [e_1577_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15356 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15420)) := by
  have c96642 := (publicBatchWrapper89_copies3020 a h).2.2.1
  have c96643 := (publicBatchWrapper89_copies3020 a h).2.2.2.1
  have c96644 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.1
  have c96645 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.1
  have c96646 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.1
  have c96647 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_10
  simp only [← c96642, ← c96643, ← c96644] at e_1577_10
  have e_1577_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_11
  simp only [← c96645, ← c96646, k2, ← c96647] at e_1577_11
  have hr := e_1577_11
  simp only [e_1577_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15357 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15421)) := by
  have c96648 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.1
  have c96649 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.1
  have c96650 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96651 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96652 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96653 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_12
  simp only [← c96648, ← c96649, ← c96650] at e_1577_12
  have e_1577_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_13
  simp only [← c96651, ← c96652, k2, ← c96653] at e_1577_13
  have hr := e_1577_13
  simp only [e_1577_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15358 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15422)) := by
  have c96654 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96655 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96656 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96657 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96658 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96659 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_14
  simp only [← c96654, ← c96655, ← c96656] at e_1577_14
  have e_1577_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_15
  simp only [← c96657, ← c96658, k2, ← c96659] at e_1577_15
  have hr := e_1577_15
  simp only [e_1577_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15359 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15423)) := by
  have c96660 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96661 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96662 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96663 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96664 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96665 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_16
  simp only [← c96660, ← c96661, ← c96662] at e_1577_16
  have e_1577_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_17
  simp only [← c96663, ← c96664, k2, ← c96665] at e_1577_17
  have hr := e_1577_17
  simp only [e_1577_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15360 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1577 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15424)) := by
  have c96666 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96667 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96668 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96669 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96670 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96671 := (publicBatchWrapper89_copies3020 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1577_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_18
  simp only [← c96666, ← c96667, ← c96668] at e_1577_18
  have e_1577_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1577 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1577_19
  simp only [← c96669, ← c96670, k2, ← c96671] at e_1577_19
  have hr := e_1577_19
  simp only [e_1577_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15361 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15425)) := by
  have c96672 := (publicBatchWrapper89_copies3021 a h).1
  have c96673 := (publicBatchWrapper89_copies3021 a h).2.1
  have c96674 := (publicBatchWrapper89_copies3021 a h).2.2.1
  have c96675 := (publicBatchWrapper89_copies3021 a h).2.2.2.1
  have c96676 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.1
  have c96677 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_0
  simp only [← c96672, ← c96673, ← c96674] at e_1578_0
  have e_1578_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_1
  simp only [← c96675, ← c96676, k2, ← c96677] at e_1578_1
  have hr := e_1578_1
  simp only [e_1578_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15362 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15426)) := by
  have c96678 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.1
  have c96679 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.1
  have c96680 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.1
  have c96681 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.1
  have c96682 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96683 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_2
  simp only [← c96678, ← c96679, ← c96680] at e_1578_2
  have e_1578_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_3
  simp only [← c96681, ← c96682, k2, ← c96683] at e_1578_3
  have hr := e_1578_3
  simp only [e_1578_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15363 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15427)) := by
  have c96684 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96685 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96686 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96687 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96688 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96689 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_4
  simp only [← c96684, ← c96685, ← c96686] at e_1578_4
  have e_1578_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_5
  simp only [← c96687, ← c96688, k2, ← c96689] at e_1578_5
  have hr := e_1578_5
  simp only [e_1578_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15364 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15428)) := by
  have c96690 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96691 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96692 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96693 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96694 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96695 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_6
  simp only [← c96690, ← c96691, ← c96692] at e_1578_6
  have e_1578_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_7
  simp only [← c96693, ← c96694, k2, ← c96695] at e_1578_7
  have hr := e_1578_7
  simp only [e_1578_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15365 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15429)) := by
  have c96696 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96697 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96698 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96699 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96700 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96701 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_8
  simp only [← c96696, ← c96697, ← c96698] at e_1578_8
  have e_1578_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_9
  simp only [← c96699, ← c96700, k2, ← c96701] at e_1578_9
  have hr := e_1578_9
  simp only [e_1578_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15366 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15430)) := by
  have c96702 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96703 := (publicBatchWrapper89_copies3021 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96704 := (publicBatchWrapper89_copies3022 a h).1
  have c96705 := (publicBatchWrapper89_copies3022 a h).2.1
  have c96706 := (publicBatchWrapper89_copies3022 a h).2.2.1
  have c96707 := (publicBatchWrapper89_copies3022 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_10
  simp only [← c96702, ← c96703, ← c96704] at e_1578_10
  have e_1578_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_11
  simp only [← c96705, ← c96706, k2, ← c96707] at e_1578_11
  have hr := e_1578_11
  simp only [e_1578_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15367 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15431)) := by
  have c96708 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.1
  have c96709 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.1
  have c96710 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.1
  have c96711 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.1
  have c96712 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.1
  have c96713 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_12
  simp only [← c96708, ← c96709, ← c96710] at e_1578_12
  have e_1578_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_13
  simp only [← c96711, ← c96712, k2, ← c96713] at e_1578_13
  have hr := e_1578_13
  simp only [e_1578_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15368 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15432)) := by
  have c96714 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96715 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96716 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96717 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96718 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96719 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_14
  simp only [← c96714, ← c96715, ← c96716] at e_1578_14
  have e_1578_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_15
  simp only [← c96717, ← c96718, k2, ← c96719] at e_1578_15
  have hr := e_1578_15
  simp only [e_1578_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15369 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15433)) := by
  have c96720 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96721 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96722 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96723 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96724 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96725 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_16
  simp only [← c96720, ← c96721, ← c96722] at e_1578_16
  have e_1578_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_17
  simp only [← c96723, ← c96724, k2, ← c96725] at e_1578_17
  have hr := e_1578_17
  simp only [e_1578_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15370 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1578 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15434)) := by
  have c96726 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96727 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96728 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96729 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96730 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96731 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1578_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_18
  simp only [← c96726, ← c96727, ← c96728] at e_1578_18
  have e_1578_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1578 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1578_19
  simp only [← c96729, ← c96730, k2, ← c96731] at e_1578_19
  have hr := e_1578_19
  simp only [e_1578_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15371 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15435)) := by
  have c96732 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96733 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96734 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96735 := (publicBatchWrapper89_copies3022 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96736 := (publicBatchWrapper89_copies3023 a h).1
  have c96737 := (publicBatchWrapper89_copies3023 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_0
  simp only [← c96732, ← c96733, ← c96734] at e_1579_0
  have e_1579_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_1
  simp only [← c96735, ← c96736, k2, ← c96737] at e_1579_1
  have hr := e_1579_1
  simp only [e_1579_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15372 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15436)) := by
  have c96738 := (publicBatchWrapper89_copies3023 a h).2.2.1
  have c96739 := (publicBatchWrapper89_copies3023 a h).2.2.2.1
  have c96740 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.1
  have c96741 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.1
  have c96742 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.1
  have c96743 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_2
  simp only [← c96738, ← c96739, ← c96740] at e_1579_2
  have e_1579_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_3
  simp only [← c96741, ← c96742, k2, ← c96743] at e_1579_3
  have hr := e_1579_3
  simp only [e_1579_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15373 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15437)) := by
  have c96744 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.1
  have c96745 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.1
  have c96746 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96747 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96748 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96749 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_4
  simp only [← c96744, ← c96745, ← c96746] at e_1579_4
  have e_1579_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_5
  simp only [← c96747, ← c96748, k2, ← c96749] at e_1579_5
  have hr := e_1579_5
  simp only [e_1579_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15374 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15438)) := by
  have c96750 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96751 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96752 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96753 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96754 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96755 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_6
  simp only [← c96750, ← c96751, ← c96752] at e_1579_6
  have e_1579_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_7
  simp only [← c96753, ← c96754, k2, ← c96755] at e_1579_7
  have hr := e_1579_7
  simp only [e_1579_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15375 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15439)) := by
  have c96756 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96757 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96758 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96759 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96760 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96761 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_8
  simp only [← c96756, ← c96757, ← c96758] at e_1579_8
  have e_1579_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_9
  simp only [← c96759, ← c96760, k2, ← c96761] at e_1579_9
  have hr := e_1579_9
  simp only [e_1579_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15376 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15440)) := by
  have c96762 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96763 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96764 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96765 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96766 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96767 := (publicBatchWrapper89_copies3023 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_10
  simp only [← c96762, ← c96763, ← c96764] at e_1579_10
  have e_1579_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_11
  simp only [← c96765, ← c96766, k2, ← c96767] at e_1579_11
  have hr := e_1579_11
  simp only [e_1579_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15377 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15441)) := by
  have c96768 := (publicBatchWrapper89_copies3024 a h).1
  have c96769 := (publicBatchWrapper89_copies3024 a h).2.1
  have c96770 := (publicBatchWrapper89_copies3024 a h).2.2.1
  have c96771 := (publicBatchWrapper89_copies3024 a h).2.2.2.1
  have c96772 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.1
  have c96773 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_12
  simp only [← c96768, ← c96769, ← c96770] at e_1579_12
  have e_1579_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_13
  simp only [← c96771, ← c96772, k2, ← c96773] at e_1579_13
  have hr := e_1579_13
  simp only [e_1579_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15378 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15442)) := by
  have c96774 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.1
  have c96775 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.1
  have c96776 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.1
  have c96777 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.1
  have c96778 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96779 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_14
  simp only [← c96774, ← c96775, ← c96776] at e_1579_14
  have e_1579_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_15
  simp only [← c96777, ← c96778, k2, ← c96779] at e_1579_15
  have hr := e_1579_15
  simp only [e_1579_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15379 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15443)) := by
  have c96780 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96781 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96782 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96783 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96784 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96785 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_16
  simp only [← c96780, ← c96781, ← c96782] at e_1579_16
  have e_1579_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_17
  simp only [← c96783, ← c96784, k2, ← c96785] at e_1579_17
  have hr := e_1579_17
  simp only [e_1579_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15380 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1579 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15444)) := by
  have c96786 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96787 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96788 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96789 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96790 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96791 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1579_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_18
  simp only [← c96786, ← c96787, ← c96788] at e_1579_18
  have e_1579_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1579 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1579_19
  simp only [← c96789, ← c96790, k2, ← c96791] at e_1579_19
  have hr := e_1579_19
  simp only [e_1579_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15381 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15445)) := by
  have c96792 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96793 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96794 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96795 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96796 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96797 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_0
  simp only [← c96792, ← c96793, ← c96794] at e_1580_0
  have e_1580_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_1
  simp only [← c96795, ← c96796, k2, ← c96797] at e_1580_1
  have hr := e_1580_1
  simp only [e_1580_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15382 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15446)) := by
  have c96798 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96799 := (publicBatchWrapper89_copies3024 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96800 := (publicBatchWrapper89_copies3025 a h).1
  have c96801 := (publicBatchWrapper89_copies3025 a h).2.1
  have c96802 := (publicBatchWrapper89_copies3025 a h).2.2.1
  have c96803 := (publicBatchWrapper89_copies3025 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_2
  simp only [← c96798, ← c96799, ← c96800] at e_1580_2
  have e_1580_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_3
  simp only [← c96801, ← c96802, k2, ← c96803] at e_1580_3
  have hr := e_1580_3
  simp only [e_1580_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15383 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15447)) := by
  have c96804 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.1
  have c96805 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.1
  have c96806 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.1
  have c96807 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.1
  have c96808 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.1
  have c96809 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_4
  simp only [← c96804, ← c96805, ← c96806] at e_1580_4
  have e_1580_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_5
  simp only [← c96807, ← c96808, k2, ← c96809] at e_1580_5
  have hr := e_1580_5
  simp only [e_1580_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15384 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15448)) := by
  have c96810 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96811 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96812 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96813 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96814 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96815 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_6
  simp only [← c96810, ← c96811, ← c96812] at e_1580_6
  have e_1580_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_7
  simp only [← c96813, ← c96814, k2, ← c96815] at e_1580_7
  have hr := e_1580_7
  simp only [e_1580_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15385 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15449)) := by
  have c96816 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96817 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96818 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96819 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96820 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96821 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_8
  simp only [← c96816, ← c96817, ← c96818] at e_1580_8
  have e_1580_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_9
  simp only [← c96819, ← c96820, k2, ← c96821] at e_1580_9
  have hr := e_1580_9
  simp only [e_1580_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15386 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15450)) := by
  have c96822 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96823 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96824 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96825 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96826 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96827 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_10
  simp only [← c96822, ← c96823, ← c96824] at e_1580_10
  have e_1580_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_11
  simp only [← c96825, ← c96826, k2, ← c96827] at e_1580_11
  have hr := e_1580_11
  simp only [e_1580_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15387 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15451)) := by
  have c96828 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96829 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96830 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96831 := (publicBatchWrapper89_copies3025 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96832 := (publicBatchWrapper89_copies3026 a h).1
  have c96833 := (publicBatchWrapper89_copies3026 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_12
  simp only [← c96828, ← c96829, ← c96830] at e_1580_12
  have e_1580_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_13
  simp only [← c96831, ← c96832, k2, ← c96833] at e_1580_13
  have hr := e_1580_13
  simp only [e_1580_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15388 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15452)) := by
  have c96834 := (publicBatchWrapper89_copies3026 a h).2.2.1
  have c96835 := (publicBatchWrapper89_copies3026 a h).2.2.2.1
  have c96836 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.1
  have c96837 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.1
  have c96838 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.1
  have c96839 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_14
  simp only [← c96834, ← c96835, ← c96836] at e_1580_14
  have e_1580_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_15
  simp only [← c96837, ← c96838, k2, ← c96839] at e_1580_15
  have hr := e_1580_15
  simp only [e_1580_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15389 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15453)) := by
  have c96840 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.1
  have c96841 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.1
  have c96842 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96843 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96844 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96845 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_16
  simp only [← c96840, ← c96841, ← c96842] at e_1580_16
  have e_1580_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_17
  simp only [← c96843, ← c96844, k2, ← c96845] at e_1580_17
  have hr := e_1580_17
  simp only [e_1580_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15390 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1580 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15454)) := by
  have c96846 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96847 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96848 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96849 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96850 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96851 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1580_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_18
  simp only [← c96846, ← c96847, ← c96848] at e_1580_18
  have e_1580_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1580 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1580_19
  simp only [← c96849, ← c96850, k2, ← c96851] at e_1580_19
  have hr := e_1580_19
  simp only [e_1580_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15391 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15455)) := by
  have c96852 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96853 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96854 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96855 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96856 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96857 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_0
  simp only [← c96852, ← c96853, ← c96854] at e_1581_0
  have e_1581_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_1
  simp only [← c96855, ← c96856, k2, ← c96857] at e_1581_1
  have hr := e_1581_1
  simp only [e_1581_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15392 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15456)) := by
  have c96858 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96859 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96860 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96861 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96862 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96863 := (publicBatchWrapper89_copies3026 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_2
  simp only [← c96858, ← c96859, ← c96860] at e_1581_2
  have e_1581_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_3
  simp only [← c96861, ← c96862, k2, ← c96863] at e_1581_3
  have hr := e_1581_3
  simp only [e_1581_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15393 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15457)) := by
  have c96864 := (publicBatchWrapper89_copies3027 a h).1
  have c96865 := (publicBatchWrapper89_copies3027 a h).2.1
  have c96866 := (publicBatchWrapper89_copies3027 a h).2.2.1
  have c96867 := (publicBatchWrapper89_copies3027 a h).2.2.2.1
  have c96868 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.1
  have c96869 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_4
  simp only [← c96864, ← c96865, ← c96866] at e_1581_4
  have e_1581_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_5
  simp only [← c96867, ← c96868, k2, ← c96869] at e_1581_5
  have hr := e_1581_5
  simp only [e_1581_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15394 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15458)) := by
  have c96870 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.1
  have c96871 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.1
  have c96872 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.1
  have c96873 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.1
  have c96874 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96875 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_6
  simp only [← c96870, ← c96871, ← c96872] at e_1581_6
  have e_1581_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_7
  simp only [← c96873, ← c96874, k2, ← c96875] at e_1581_7
  have hr := e_1581_7
  simp only [e_1581_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15395 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15459)) := by
  have c96876 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96877 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96878 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96879 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96880 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96881 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_8
  simp only [← c96876, ← c96877, ← c96878] at e_1581_8
  have e_1581_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_9
  simp only [← c96879, ← c96880, k2, ← c96881] at e_1581_9
  have hr := e_1581_9
  simp only [e_1581_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15396 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15460)) := by
  have c96882 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96883 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96884 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96885 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96886 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96887 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_10
  simp only [← c96882, ← c96883, ← c96884] at e_1581_10
  have e_1581_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_11
  simp only [← c96885, ← c96886, k2, ← c96887] at e_1581_11
  have hr := e_1581_11
  simp only [e_1581_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15397 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15461)) := by
  have c96888 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96889 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96890 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96891 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96892 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96893 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_12
  simp only [← c96888, ← c96889, ← c96890] at e_1581_12
  have e_1581_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_13
  simp only [← c96891, ← c96892, k2, ← c96893] at e_1581_13
  have hr := e_1581_13
  simp only [e_1581_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15398 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15462)) := by
  have c96894 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96895 := (publicBatchWrapper89_copies3027 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96896 := (publicBatchWrapper89_copies3028 a h).1
  have c96897 := (publicBatchWrapper89_copies3028 a h).2.1
  have c96898 := (publicBatchWrapper89_copies3028 a h).2.2.1
  have c96899 := (publicBatchWrapper89_copies3028 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_14
  simp only [← c96894, ← c96895, ← c96896] at e_1581_14
  have e_1581_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_15
  simp only [← c96897, ← c96898, k2, ← c96899] at e_1581_15
  have hr := e_1581_15
  simp only [e_1581_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15399 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15463)) := by
  have c96900 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.1
  have c96901 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.1
  have c96902 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.1
  have c96903 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.1
  have c96904 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.1
  have c96905 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_16
  simp only [← c96900, ← c96901, ← c96902] at e_1581_16
  have e_1581_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_17
  simp only [← c96903, ← c96904, k2, ← c96905] at e_1581_17
  have hr := e_1581_17
  simp only [e_1581_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15400 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1581 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15464)) := by
  have c96906 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96907 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96908 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96909 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96910 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96911 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1581_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_18
  simp only [← c96906, ← c96907, ← c96908] at e_1581_18
  have e_1581_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1581 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1581_19
  simp only [← c96909, ← c96910, k2, ← c96911] at e_1581_19
  have hr := e_1581_19
  simp only [e_1581_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15401 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15465)) := by
  have c96912 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96913 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96914 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96915 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96916 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96917 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_0
  simp only [← c96912, ← c96913, ← c96914] at e_1582_0
  have e_1582_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_1
  simp only [← c96915, ← c96916, k2, ← c96917] at e_1582_1
  have hr := e_1582_1
  simp only [e_1582_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15402 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15466)) := by
  have c96918 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96919 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96920 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96921 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96922 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96923 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_2
  simp only [← c96918, ← c96919, ← c96920] at e_1582_2
  have e_1582_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_3
  simp only [← c96921, ← c96922, k2, ← c96923] at e_1582_3
  have hr := e_1582_3
  simp only [e_1582_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15403 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15467)) := by
  have c96924 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96925 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96926 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96927 := (publicBatchWrapper89_copies3028 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96928 := (publicBatchWrapper89_copies3029 a h).1
  have c96929 := (publicBatchWrapper89_copies3029 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_4
  simp only [← c96924, ← c96925, ← c96926] at e_1582_4
  have e_1582_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_5
  simp only [← c96927, ← c96928, k2, ← c96929] at e_1582_5
  have hr := e_1582_5
  simp only [e_1582_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15404 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15468)) := by
  have c96930 := (publicBatchWrapper89_copies3029 a h).2.2.1
  have c96931 := (publicBatchWrapper89_copies3029 a h).2.2.2.1
  have c96932 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.1
  have c96933 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.1
  have c96934 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.1
  have c96935 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_6
  simp only [← c96930, ← c96931, ← c96932] at e_1582_6
  have e_1582_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_7
  simp only [← c96933, ← c96934, k2, ← c96935] at e_1582_7
  have hr := e_1582_7
  simp only [e_1582_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15405 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15469)) := by
  have c96936 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.1
  have c96937 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.1
  have c96938 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96939 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c96940 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96941 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_8
  simp only [← c96936, ← c96937, ← c96938] at e_1582_8
  have e_1582_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_9
  simp only [← c96939, ← c96940, k2, ← c96941] at e_1582_9
  have hr := e_1582_9
  simp only [e_1582_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15406 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15470)) := by
  have c96942 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96943 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96944 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96945 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96946 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96947 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_10
  simp only [← c96942, ← c96943, ← c96944] at e_1582_10
  have e_1582_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_11
  simp only [← c96945, ← c96946, k2, ← c96947] at e_1582_11
  have hr := e_1582_11
  simp only [e_1582_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15407 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15471)) := by
  have c96948 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96949 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96950 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96951 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96952 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96953 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_12
  simp only [← c96948, ← c96949, ← c96950] at e_1582_12
  have e_1582_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_13
  simp only [← c96951, ← c96952, k2, ← c96953] at e_1582_13
  have hr := e_1582_13
  simp only [e_1582_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15408 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15472)) := by
  have c96954 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96955 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96956 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96957 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96958 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96959 := (publicBatchWrapper89_copies3029 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_14
  simp only [← c96954, ← c96955, ← c96956] at e_1582_14
  have e_1582_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_15
  simp only [← c96957, ← c96958, k2, ← c96959] at e_1582_15
  have hr := e_1582_15
  simp only [e_1582_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15409 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15473)) := by
  have c96960 := (publicBatchWrapper89_copies3030 a h).1
  have c96961 := (publicBatchWrapper89_copies3030 a h).2.1
  have c96962 := (publicBatchWrapper89_copies3030 a h).2.2.1
  have c96963 := (publicBatchWrapper89_copies3030 a h).2.2.2.1
  have c96964 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.1
  have c96965 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_16
  simp only [← c96960, ← c96961, ← c96962] at e_1582_16
  have e_1582_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_17
  simp only [← c96963, ← c96964, k2, ← c96965] at e_1582_17
  have hr := e_1582_17
  simp only [e_1582_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15410 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1582 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15474)) := by
  have c96966 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.1
  have c96967 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.1
  have c96968 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.1
  have c96969 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.1
  have c96970 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.1
  have c96971 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1582_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_18
  simp only [← c96966, ← c96967, ← c96968] at e_1582_18
  have e_1582_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1582 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1582_19
  simp only [← c96969, ← c96970, k2, ← c96971] at e_1582_19
  have hr := e_1582_19
  simp only [e_1582_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15411 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15475)) := by
  have c96972 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96973 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96974 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96975 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96976 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96977 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_0
  simp only [← c96972, ← c96973, ← c96974] at e_1583_0
  have e_1583_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_1
  simp only [← c96975, ← c96976, k2, ← c96977] at e_1583_1
  have hr := e_1583_1
  simp only [e_1583_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15412 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15476)) := by
  have c96978 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96979 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96980 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96981 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96982 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96983 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_2
  simp only [← c96978, ← c96979, ← c96980] at e_1583_2
  have e_1583_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_3
  simp only [← c96981, ← c96982, k2, ← c96983] at e_1583_3
  have hr := e_1583_3
  simp only [e_1583_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15413 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15477)) := by
  have c96984 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96985 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96986 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96987 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96988 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96989 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_4
  simp only [← c96984, ← c96985, ← c96986] at e_1583_4
  have e_1583_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_5
  simp only [← c96987, ← c96988, k2, ← c96989] at e_1583_5
  have hr := e_1583_5
  simp only [e_1583_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15414 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15478)) := by
  have c96990 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c96991 := (publicBatchWrapper89_copies3030 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c96992 := (publicBatchWrapper89_copies3031 a h).1
  have c96993 := (publicBatchWrapper89_copies3031 a h).2.1
  have c96994 := (publicBatchWrapper89_copies3031 a h).2.2.1
  have c96995 := (publicBatchWrapper89_copies3031 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_6
  simp only [← c96990, ← c96991, ← c96992] at e_1583_6
  have e_1583_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_7
  simp only [← c96993, ← c96994, k2, ← c96995] at e_1583_7
  have hr := e_1583_7
  simp only [e_1583_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15415 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15479)) := by
  have c96996 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.1
  have c96997 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.1
  have c96998 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.1
  have c96999 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.1
  have c97000 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.1
  have c97001 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_8
  simp only [← c96996, ← c96997, ← c96998] at e_1583_8
  have e_1583_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_9
  simp only [← c96999, ← c97000, k2, ← c97001] at e_1583_9
  have hr := e_1583_9
  simp only [e_1583_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15416 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15480)) := by
  have c97002 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97003 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97004 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97005 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97006 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97007 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_10
  simp only [← c97002, ← c97003, ← c97004] at e_1583_10
  have e_1583_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_11
  simp only [← c97005, ← c97006, k2, ← c97007] at e_1583_11
  have hr := e_1583_11
  simp only [e_1583_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15417 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15481)) := by
  have c97008 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97009 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97010 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97011 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97012 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97013 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_12
  simp only [← c97008, ← c97009, ← c97010] at e_1583_12
  have e_1583_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_13
  simp only [← c97011, ← c97012, k2, ← c97013] at e_1583_13
  have hr := e_1583_13
  simp only [e_1583_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15418 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15482)) := by
  have c97014 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97015 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97016 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97017 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97018 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97019 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_14
  simp only [← c97014, ← c97015, ← c97016] at e_1583_14
  have e_1583_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_15
  simp only [← c97017, ← c97018, k2, ← c97019] at e_1583_15
  have hr := e_1583_15
  simp only [e_1583_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15419 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15483)) := by
  have c97020 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97021 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97022 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97023 := (publicBatchWrapper89_copies3031 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97024 := (publicBatchWrapper89_copies3032 a h).1
  have c97025 := (publicBatchWrapper89_copies3032 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_16
  simp only [← c97020, ← c97021, ← c97022] at e_1583_16
  have e_1583_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_17
  simp only [← c97023, ← c97024, k2, ← c97025] at e_1583_17
  have hr := e_1583_17
  simp only [e_1583_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15420 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1583 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15484)) := by
  have c97026 := (publicBatchWrapper89_copies3032 a h).2.2.1
  have c97027 := (publicBatchWrapper89_copies3032 a h).2.2.2.1
  have c97028 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.1
  have c97029 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.1
  have c97030 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.1
  have c97031 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1583_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_18
  simp only [← c97026, ← c97027, ← c97028] at e_1583_18
  have e_1583_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1583 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1583_19
  simp only [← c97029, ← c97030, k2, ← c97031] at e_1583_19
  have hr := e_1583_19
  simp only [e_1583_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15421 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15485)) := by
  have c97032 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.1
  have c97033 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.1
  have c97034 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97035 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97036 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97037 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_0
  simp only [← c97032, ← c97033, ← c97034] at e_1584_0
  have e_1584_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_1
  simp only [← c97035, ← c97036, k2, ← c97037] at e_1584_1
  have hr := e_1584_1
  simp only [e_1584_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15422 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15486)) := by
  have c97038 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97039 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97040 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97041 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97042 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97043 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_2
  simp only [← c97038, ← c97039, ← c97040] at e_1584_2
  have e_1584_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_3
  simp only [← c97041, ← c97042, k2, ← c97043] at e_1584_3
  have hr := e_1584_3
  simp only [e_1584_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15423 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15487)) := by
  have c97044 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97045 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97046 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97047 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97048 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97049 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_4
  simp only [← c97044, ← c97045, ← c97046] at e_1584_4
  have e_1584_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_5
  simp only [← c97047, ← c97048, k2, ← c97049] at e_1584_5
  have hr := e_1584_5
  simp only [e_1584_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15424 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15488)) := by
  have c97050 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97051 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97052 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97053 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97054 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97055 := (publicBatchWrapper89_copies3032 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_6
  simp only [← c97050, ← c97051, ← c97052] at e_1584_6
  have e_1584_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_7
  simp only [← c97053, ← c97054, k2, ← c97055] at e_1584_7
  have hr := e_1584_7
  simp only [e_1584_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15425 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15489)) := by
  have c97056 := (publicBatchWrapper89_copies3033 a h).1
  have c97057 := (publicBatchWrapper89_copies3033 a h).2.1
  have c97058 := (publicBatchWrapper89_copies3033 a h).2.2.1
  have c97059 := (publicBatchWrapper89_copies3033 a h).2.2.2.1
  have c97060 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.1
  have c97061 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_8
  simp only [← c97056, ← c97057, ← c97058] at e_1584_8
  have e_1584_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_9
  simp only [← c97059, ← c97060, k2, ← c97061] at e_1584_9
  have hr := e_1584_9
  simp only [e_1584_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15426 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15490)) := by
  have c97062 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.1
  have c97063 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.1
  have c97064 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.1
  have c97065 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.1
  have c97066 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97067 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_10
  simp only [← c97062, ← c97063, ← c97064] at e_1584_10
  have e_1584_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_11
  simp only [← c97065, ← c97066, k2, ← c97067] at e_1584_11
  have hr := e_1584_11
  simp only [e_1584_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15427 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15491)) := by
  have c97068 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97069 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97070 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97071 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97072 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97073 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_12
  simp only [← c97068, ← c97069, ← c97070] at e_1584_12
  have e_1584_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_13
  simp only [← c97071, ← c97072, k2, ← c97073] at e_1584_13
  have hr := e_1584_13
  simp only [e_1584_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15428 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15492)) := by
  have c97074 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97075 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97076 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97077 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97078 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97079 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_14
  simp only [← c97074, ← c97075, ← c97076] at e_1584_14
  have e_1584_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_15
  simp only [← c97077, ← c97078, k2, ← c97079] at e_1584_15
  have hr := e_1584_15
  simp only [e_1584_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15429 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15493)) := by
  have c97080 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97081 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97082 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97083 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97084 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97085 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_16
  simp only [← c97080, ← c97081, ← c97082] at e_1584_16
  have e_1584_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_17
  simp only [← c97083, ← c97084, k2, ← c97085] at e_1584_17
  have hr := e_1584_17
  simp only [e_1584_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15430 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1584 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15494)) := by
  have c97086 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97087 := (publicBatchWrapper89_copies3033 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97088 := (publicBatchWrapper89_copies3034 a h).1
  have c97089 := (publicBatchWrapper89_copies3034 a h).2.1
  have c97090 := (publicBatchWrapper89_copies3034 a h).2.2.1
  have c97091 := (publicBatchWrapper89_copies3034 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1584_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_18
  simp only [← c97086, ← c97087, ← c97088] at e_1584_18
  have e_1584_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1584 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1584_19
  simp only [← c97089, ← c97090, k2, ← c97091] at e_1584_19
  have hr := e_1584_19
  simp only [e_1584_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15431 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1585 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15495)) := by
  have c97092 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.1
  have c97093 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.1
  have c97094 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.1
  have c97095 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.1
  have c97096 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.1
  have c97097 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1585_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_0
  simp only [← c97092, ← c97093, ← c97094] at e_1585_0
  have e_1585_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_1
  simp only [← c97095, ← c97096, k2, ← c97097] at e_1585_1
  have hr := e_1585_1
  simp only [e_1585_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15432 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1585 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15496)) := by
  have c97098 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97099 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97100 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97101 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97102 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97103 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1585_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_2
  simp only [← c97098, ← c97099, ← c97100] at e_1585_2
  have e_1585_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_3
  simp only [← c97101, ← c97102, k2, ← c97103] at e_1585_3
  have hr := e_1585_3
  simp only [e_1585_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15433 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1585 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15497)) := by
  have c97104 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97105 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97106 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97107 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97108 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97109 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1585_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_4
  simp only [← c97104, ← c97105, ← c97106] at e_1585_4
  have e_1585_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_5
  simp only [← c97107, ← c97108, k2, ← c97109] at e_1585_5
  have hr := e_1585_5
  simp only [e_1585_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15434 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1585 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15498)) := by
  have c97110 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97111 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97112 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97113 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97114 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97115 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1585_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_6
  simp only [← c97110, ← c97111, ← c97112] at e_1585_6
  have e_1585_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_7
  simp only [← c97113, ← c97114, k2, ← c97115] at e_1585_7
  have hr := e_1585_7
  simp only [e_1585_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15435 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1585 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15499)) := by
  have c97116 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97117 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97118 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97119 := (publicBatchWrapper89_copies3034 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97120 := (publicBatchWrapper89_copies3035 a h).1
  have c97121 := (publicBatchWrapper89_copies3035 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1585_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_8
  simp only [← c97116, ← c97117, ← c97118] at e_1585_8
  have e_1585_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_9
  simp only [← c97119, ← c97120, k2, ← c97121] at e_1585_9
  have hr := e_1585_9
  simp only [e_1585_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15436 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1585 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15500)) := by
  have c97122 := (publicBatchWrapper89_copies3035 a h).2.2.1
  have c97123 := (publicBatchWrapper89_copies3035 a h).2.2.2.1
  have c97124 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.1
  have c97125 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.1
  have c97126 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.1
  have c97127 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1585_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_10
  simp only [← c97122, ← c97123, ← c97124] at e_1585_10
  have e_1585_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_11
  simp only [← c97125, ← c97126, k2, ← c97127] at e_1585_11
  have hr := e_1585_11
  simp only [e_1585_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15437 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1585 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15501)) := by
  have c97128 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.1
  have c97129 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.1
  have c97130 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97131 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97132 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97133 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1585_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_12
  simp only [← c97128, ← c97129, ← c97130] at e_1585_12
  have e_1585_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_13
  simp only [← c97131, ← c97132, k2, ← c97133] at e_1585_13
  have hr := e_1585_13
  simp only [e_1585_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15438 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1585 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15502)) := by
  have c97134 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97135 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97136 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97137 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97138 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97139 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1585_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_14
  simp only [← c97134, ← c97135, ← c97136] at e_1585_14
  have e_1585_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_15
  simp only [← c97137, ← c97138, k2, ← c97139] at e_1585_15
  have hr := e_1585_15
  simp only [e_1585_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15439 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1585 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15503)) := by
  have c97140 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97141 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97142 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97143 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97144 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97145 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1585_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_16
  simp only [← c97140, ← c97141, ← c97142] at e_1585_16
  have e_1585_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_17
  simp only [← c97143, ← c97144, k2, ← c97145] at e_1585_17
  have hr := e_1585_17
  simp only [e_1585_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15440 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1585 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15504)) := by
  have c97146 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97147 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97148 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97149 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97150 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97151 := (publicBatchWrapper89_copies3035 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1585_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_18
  simp only [← c97146, ← c97147, ← c97148] at e_1585_18
  have e_1585_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1585 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1585_19
  simp only [← c97149, ← c97150, k2, ← c97151] at e_1585_19
  have hr := e_1585_19
  simp only [e_1585_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15441 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1586 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15505)) := by
  have c97152 := (publicBatchWrapper89_copies3036 a h).1
  have c97153 := (publicBatchWrapper89_copies3036 a h).2.1
  have c97154 := (publicBatchWrapper89_copies3036 a h).2.2.1
  have c97155 := (publicBatchWrapper89_copies3036 a h).2.2.2.1
  have c97156 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.1
  have c97157 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1586_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_0
  simp only [← c97152, ← c97153, ← c97154] at e_1586_0
  have e_1586_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_1
  simp only [← c97155, ← c97156, k2, ← c97157] at e_1586_1
  have hr := e_1586_1
  simp only [e_1586_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15442 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1586 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15506)) := by
  have c97158 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.1
  have c97159 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.1
  have c97160 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.1
  have c97161 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.1
  have c97162 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97163 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1586_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_2
  simp only [← c97158, ← c97159, ← c97160] at e_1586_2
  have e_1586_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_3
  simp only [← c97161, ← c97162, k2, ← c97163] at e_1586_3
  have hr := e_1586_3
  simp only [e_1586_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15443 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1586 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15507)) := by
  have c97164 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97165 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97166 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97167 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97168 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97169 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1586_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_4
  simp only [← c97164, ← c97165, ← c97166] at e_1586_4
  have e_1586_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_5
  simp only [← c97167, ← c97168, k2, ← c97169] at e_1586_5
  have hr := e_1586_5
  simp only [e_1586_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15444 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1586 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15508)) := by
  have c97170 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97171 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97172 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97173 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97174 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97175 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1586_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_6
  simp only [← c97170, ← c97171, ← c97172] at e_1586_6
  have e_1586_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_7
  simp only [← c97173, ← c97174, k2, ← c97175] at e_1586_7
  have hr := e_1586_7
  simp only [e_1586_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15445 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1586 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15509)) := by
  have c97176 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97177 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97178 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97179 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97180 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97181 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1586_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_8
  simp only [← c97176, ← c97177, ← c97178] at e_1586_8
  have e_1586_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_9
  simp only [← c97179, ← c97180, k2, ← c97181] at e_1586_9
  have hr := e_1586_9
  simp only [e_1586_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15446 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1586 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15510)) := by
  have c97182 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97183 := (publicBatchWrapper89_copies3036 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97184 := (publicBatchWrapper89_copies3037 a h).1
  have c97185 := (publicBatchWrapper89_copies3037 a h).2.1
  have c97186 := (publicBatchWrapper89_copies3037 a h).2.2.1
  have c97187 := (publicBatchWrapper89_copies3037 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1586_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_10
  simp only [← c97182, ← c97183, ← c97184] at e_1586_10
  have e_1586_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_11
  simp only [← c97185, ← c97186, k2, ← c97187] at e_1586_11
  have hr := e_1586_11
  simp only [e_1586_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15447 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1586 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15511)) := by
  have c97188 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.1
  have c97189 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.1
  have c97190 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.1
  have c97191 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.1
  have c97192 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.1
  have c97193 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1586_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_12
  simp only [← c97188, ← c97189, ← c97190] at e_1586_12
  have e_1586_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_13
  simp only [← c97191, ← c97192, k2, ← c97193] at e_1586_13
  have hr := e_1586_13
  simp only [e_1586_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15448 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1586 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15512)) := by
  have c97194 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97195 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97196 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97197 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97198 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97199 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1586_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_14
  simp only [← c97194, ← c97195, ← c97196] at e_1586_14
  have e_1586_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_15
  simp only [← c97197, ← c97198, k2, ← c97199] at e_1586_15
  have hr := e_1586_15
  simp only [e_1586_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15449 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1586 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15513)) := by
  have c97200 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97201 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97202 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97203 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97204 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97205 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1586_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_16
  simp only [← c97200, ← c97201, ← c97202] at e_1586_16
  have e_1586_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_17
  simp only [← c97203, ← c97204, k2, ← c97205] at e_1586_17
  have hr := e_1586_17
  simp only [e_1586_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15450 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1586 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15514)) := by
  have c97206 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97207 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97208 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97209 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97210 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97211 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1586_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_18
  simp only [← c97206, ← c97207, ← c97208] at e_1586_18
  have e_1586_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1586 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1586_19
  simp only [← c97209, ← c97210, k2, ← c97211] at e_1586_19
  have hr := e_1586_19
  simp only [e_1586_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15451 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1587 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15515)) := by
  have c97212 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97213 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97214 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97215 := (publicBatchWrapper89_copies3037 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97216 := (publicBatchWrapper89_copies3038 a h).1
  have c97217 := (publicBatchWrapper89_copies3038 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1587_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_0
  simp only [← c97212, ← c97213, ← c97214] at e_1587_0
  have e_1587_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_1
  simp only [← c97215, ← c97216, k2, ← c97217] at e_1587_1
  have hr := e_1587_1
  simp only [e_1587_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15452 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1587 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15516)) := by
  have c97218 := (publicBatchWrapper89_copies3038 a h).2.2.1
  have c97219 := (publicBatchWrapper89_copies3038 a h).2.2.2.1
  have c97220 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.1
  have c97221 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.1
  have c97222 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.1
  have c97223 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1587_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_2
  simp only [← c97218, ← c97219, ← c97220] at e_1587_2
  have e_1587_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_3
  simp only [← c97221, ← c97222, k2, ← c97223] at e_1587_3
  have hr := e_1587_3
  simp only [e_1587_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15453 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1587 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15517)) := by
  have c97224 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.1
  have c97225 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.1
  have c97226 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97227 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97228 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97229 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1587_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_4
  simp only [← c97224, ← c97225, ← c97226] at e_1587_4
  have e_1587_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_5
  simp only [← c97227, ← c97228, k2, ← c97229] at e_1587_5
  have hr := e_1587_5
  simp only [e_1587_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15454 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1587 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15518)) := by
  have c97230 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97231 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97232 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97233 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97234 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97235 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1587_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_6
  simp only [← c97230, ← c97231, ← c97232] at e_1587_6
  have e_1587_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_7
  simp only [← c97233, ← c97234, k2, ← c97235] at e_1587_7
  have hr := e_1587_7
  simp only [e_1587_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15455 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1587 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15519)) := by
  have c97236 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97237 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97238 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97239 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97240 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97241 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1587_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_8
  simp only [← c97236, ← c97237, ← c97238] at e_1587_8
  have e_1587_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_9
  simp only [← c97239, ← c97240, k2, ← c97241] at e_1587_9
  have hr := e_1587_9
  simp only [e_1587_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15456 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1587 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15520)) := by
  have c97242 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97243 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97244 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97245 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97246 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97247 := (publicBatchWrapper89_copies3038 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1587_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_10
  simp only [← c97242, ← c97243, ← c97244] at e_1587_10
  have e_1587_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_11
  simp only [← c97245, ← c97246, k2, ← c97247] at e_1587_11
  have hr := e_1587_11
  simp only [e_1587_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15457 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1587 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15521)) := by
  have c97248 := (publicBatchWrapper89_copies3039 a h).1
  have c97249 := (publicBatchWrapper89_copies3039 a h).2.1
  have c97250 := (publicBatchWrapper89_copies3039 a h).2.2.1
  have c97251 := (publicBatchWrapper89_copies3039 a h).2.2.2.1
  have c97252 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.1
  have c97253 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1587_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_12
  simp only [← c97248, ← c97249, ← c97250] at e_1587_12
  have e_1587_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_13
  simp only [← c97251, ← c97252, k2, ← c97253] at e_1587_13
  have hr := e_1587_13
  simp only [e_1587_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15458 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1587 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15522)) := by
  have c97254 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.1
  have c97255 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.1
  have c97256 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.1
  have c97257 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.1
  have c97258 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97259 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1587_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_14
  simp only [← c97254, ← c97255, ← c97256] at e_1587_14
  have e_1587_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_15
  simp only [← c97257, ← c97258, k2, ← c97259] at e_1587_15
  have hr := e_1587_15
  simp only [e_1587_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15459 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1587 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15523)) := by
  have c97260 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97261 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97262 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97263 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97264 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97265 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1587_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_16
  simp only [← c97260, ← c97261, ← c97262] at e_1587_16
  have e_1587_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_17
  simp only [← c97263, ← c97264, k2, ← c97265] at e_1587_17
  have hr := e_1587_17
  simp only [e_1587_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15460 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1587 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15524)) := by
  have c97266 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97267 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97268 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97269 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97270 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97271 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1587_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_18
  simp only [← c97266, ← c97267, ← c97268] at e_1587_18
  have e_1587_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1587 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1587_19
  simp only [← c97269, ← c97270, k2, ← c97271] at e_1587_19
  have hr := e_1587_19
  simp only [e_1587_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15461 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1588 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15525)) := by
  have c97272 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97273 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97274 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97275 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97276 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97277 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1588_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_0
  simp only [← c97272, ← c97273, ← c97274] at e_1588_0
  have e_1588_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_1
  simp only [← c97275, ← c97276, k2, ← c97277] at e_1588_1
  have hr := e_1588_1
  simp only [e_1588_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15462 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1588 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15526)) := by
  have c97278 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97279 := (publicBatchWrapper89_copies3039 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97280 := (publicBatchWrapper89_copies3040 a h).1
  have c97281 := (publicBatchWrapper89_copies3040 a h).2.1
  have c97282 := (publicBatchWrapper89_copies3040 a h).2.2.1
  have c97283 := (publicBatchWrapper89_copies3040 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1588_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_2
  simp only [← c97278, ← c97279, ← c97280] at e_1588_2
  have e_1588_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_3
  simp only [← c97281, ← c97282, k2, ← c97283] at e_1588_3
  have hr := e_1588_3
  simp only [e_1588_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15463 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1588 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15527)) := by
  have c97284 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.1
  have c97285 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.1
  have c97286 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.1
  have c97287 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.1
  have c97288 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.1
  have c97289 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1588_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_4
  simp only [← c97284, ← c97285, ← c97286] at e_1588_4
  have e_1588_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_5
  simp only [← c97287, ← c97288, k2, ← c97289] at e_1588_5
  have hr := e_1588_5
  simp only [e_1588_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15464 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1588 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15528)) := by
  have c97290 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97291 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97292 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97293 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97294 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97295 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1588_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_6
  simp only [← c97290, ← c97291, ← c97292] at e_1588_6
  have e_1588_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_7
  simp only [← c97293, ← c97294, k2, ← c97295] at e_1588_7
  have hr := e_1588_7
  simp only [e_1588_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15465 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1588 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15529)) := by
  have c97296 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97297 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97298 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97299 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97300 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97301 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1588_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_8
  simp only [← c97296, ← c97297, ← c97298] at e_1588_8
  have e_1588_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_9
  simp only [← c97299, ← c97300, k2, ← c97301] at e_1588_9
  have hr := e_1588_9
  simp only [e_1588_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15466 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1588 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15530)) := by
  have c97302 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97303 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97304 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97305 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97306 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97307 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1588_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_10
  simp only [← c97302, ← c97303, ← c97304] at e_1588_10
  have e_1588_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_11
  simp only [← c97305, ← c97306, k2, ← c97307] at e_1588_11
  have hr := e_1588_11
  simp only [e_1588_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15467 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1588 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15531)) := by
  have c97308 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97309 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97310 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97311 := (publicBatchWrapper89_copies3040 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97312 := (publicBatchWrapper89_copies3041 a h).1
  have c97313 := (publicBatchWrapper89_copies3041 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1588_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_12
  simp only [← c97308, ← c97309, ← c97310] at e_1588_12
  have e_1588_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_13
  simp only [← c97311, ← c97312, k2, ← c97313] at e_1588_13
  have hr := e_1588_13
  simp only [e_1588_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15468 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1588 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15532)) := by
  have c97314 := (publicBatchWrapper89_copies3041 a h).2.2.1
  have c97315 := (publicBatchWrapper89_copies3041 a h).2.2.2.1
  have c97316 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.1
  have c97317 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.1
  have c97318 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.1
  have c97319 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1588_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_14
  simp only [← c97314, ← c97315, ← c97316] at e_1588_14
  have e_1588_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_15
  simp only [← c97317, ← c97318, k2, ← c97319] at e_1588_15
  have hr := e_1588_15
  simp only [e_1588_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15469 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1588 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15533)) := by
  have c97320 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.1
  have c97321 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.1
  have c97322 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97323 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97324 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97325 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1588_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_16
  simp only [← c97320, ← c97321, ← c97322] at e_1588_16
  have e_1588_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_17
  simp only [← c97323, ← c97324, k2, ← c97325] at e_1588_17
  have hr := e_1588_17
  simp only [e_1588_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15470 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1588 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15534)) := by
  have c97326 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97327 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97328 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97329 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97330 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97331 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1588_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_18
  simp only [← c97326, ← c97327, ← c97328] at e_1588_18
  have e_1588_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1588 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1588_19
  simp only [← c97329, ← c97330, k2, ← c97331] at e_1588_19
  have hr := e_1588_19
  simp only [e_1588_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15471 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1589 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15535)) := by
  have c97332 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97333 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97334 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97335 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97336 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97337 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1589_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_0
  simp only [← c97332, ← c97333, ← c97334] at e_1589_0
  have e_1589_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_1
  simp only [← c97335, ← c97336, k2, ← c97337] at e_1589_1
  have hr := e_1589_1
  simp only [e_1589_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15472 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1589 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15536)) := by
  have c97338 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97339 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97340 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97341 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97342 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97343 := (publicBatchWrapper89_copies3041 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1589_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_2
  simp only [← c97338, ← c97339, ← c97340] at e_1589_2
  have e_1589_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_3
  simp only [← c97341, ← c97342, k2, ← c97343] at e_1589_3
  have hr := e_1589_3
  simp only [e_1589_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15473 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1589 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15537)) := by
  have c97344 := (publicBatchWrapper89_copies3042 a h).1
  have c97345 := (publicBatchWrapper89_copies3042 a h).2.1
  have c97346 := (publicBatchWrapper89_copies3042 a h).2.2.1
  have c97347 := (publicBatchWrapper89_copies3042 a h).2.2.2.1
  have c97348 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.1
  have c97349 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1589_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_4
  simp only [← c97344, ← c97345, ← c97346] at e_1589_4
  have e_1589_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_5
  simp only [← c97347, ← c97348, k2, ← c97349] at e_1589_5
  have hr := e_1589_5
  simp only [e_1589_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15474 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1589 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15538)) := by
  have c97350 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.1
  have c97351 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.1
  have c97352 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.1
  have c97353 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.1
  have c97354 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97355 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1589_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_6
  simp only [← c97350, ← c97351, ← c97352] at e_1589_6
  have e_1589_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_7
  simp only [← c97353, ← c97354, k2, ← c97355] at e_1589_7
  have hr := e_1589_7
  simp only [e_1589_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15475 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1589 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15539)) := by
  have c97356 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97357 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97358 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97359 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97360 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97361 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1589_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_8
  simp only [← c97356, ← c97357, ← c97358] at e_1589_8
  have e_1589_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_9
  simp only [← c97359, ← c97360, k2, ← c97361] at e_1589_9
  have hr := e_1589_9
  simp only [e_1589_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15476 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1589 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15540)) := by
  have c97362 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97363 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97364 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97365 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97366 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97367 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1589_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_10
  simp only [← c97362, ← c97363, ← c97364] at e_1589_10
  have e_1589_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_11
  simp only [← c97365, ← c97366, k2, ← c97367] at e_1589_11
  have hr := e_1589_11
  simp only [e_1589_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15477 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1589 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15541)) := by
  have c97368 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97369 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97370 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97371 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97372 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97373 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1589_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_12
  simp only [← c97368, ← c97369, ← c97370] at e_1589_12
  have e_1589_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_13
  simp only [← c97371, ← c97372, k2, ← c97373] at e_1589_13
  have hr := e_1589_13
  simp only [e_1589_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15478 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1589 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15542)) := by
  have c97374 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97375 := (publicBatchWrapper89_copies3042 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97376 := (publicBatchWrapper89_copies3043 a h).1
  have c97377 := (publicBatchWrapper89_copies3043 a h).2.1
  have c97378 := (publicBatchWrapper89_copies3043 a h).2.2.1
  have c97379 := (publicBatchWrapper89_copies3043 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1589_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_14
  simp only [← c97374, ← c97375, ← c97376] at e_1589_14
  have e_1589_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_15
  simp only [← c97377, ← c97378, k2, ← c97379] at e_1589_15
  have hr := e_1589_15
  simp only [e_1589_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15479 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1589 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15543)) := by
  have c97380 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.1
  have c97381 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.1
  have c97382 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.1
  have c97383 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.1
  have c97384 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.1
  have c97385 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1589_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_16
  simp only [← c97380, ← c97381, ← c97382] at e_1589_16
  have e_1589_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_17
  simp only [← c97383, ← c97384, k2, ← c97385] at e_1589_17
  have hr := e_1589_17
  simp only [e_1589_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15480 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1589 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15544)) := by
  have c97386 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97387 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97388 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97389 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97390 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97391 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1589_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_18
  simp only [← c97386, ← c97387, ← c97388] at e_1589_18
  have e_1589_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1589 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1589_19
  simp only [← c97389, ← c97390, k2, ← c97391] at e_1589_19
  have hr := e_1589_19
  simp only [e_1589_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15481 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1590 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15545)) := by
  have c97392 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97393 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97394 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97395 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97396 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97397 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1590_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_0
  simp only [← c97392, ← c97393, ← c97394] at e_1590_0
  have e_1590_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_1
  simp only [← c97395, ← c97396, k2, ← c97397] at e_1590_1
  have hr := e_1590_1
  simp only [e_1590_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15482 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1590 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15546)) := by
  have c97398 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97399 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97400 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97401 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97402 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97403 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1590_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_2
  simp only [← c97398, ← c97399, ← c97400] at e_1590_2
  have e_1590_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_3
  simp only [← c97401, ← c97402, k2, ← c97403] at e_1590_3
  have hr := e_1590_3
  simp only [e_1590_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15483 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1590 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15547)) := by
  have c97404 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97405 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97406 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97407 := (publicBatchWrapper89_copies3043 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97408 := (publicBatchWrapper89_copies3044 a h).1
  have c97409 := (publicBatchWrapper89_copies3044 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1590_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_4
  simp only [← c97404, ← c97405, ← c97406] at e_1590_4
  have e_1590_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_5
  simp only [← c97407, ← c97408, k2, ← c97409] at e_1590_5
  have hr := e_1590_5
  simp only [e_1590_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15484 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1590 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15548)) := by
  have c97410 := (publicBatchWrapper89_copies3044 a h).2.2.1
  have c97411 := (publicBatchWrapper89_copies3044 a h).2.2.2.1
  have c97412 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.1
  have c97413 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.1
  have c97414 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.1
  have c97415 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1590_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_6
  simp only [← c97410, ← c97411, ← c97412] at e_1590_6
  have e_1590_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_7
  simp only [← c97413, ← c97414, k2, ← c97415] at e_1590_7
  have hr := e_1590_7
  simp only [e_1590_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15485 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1590 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15549)) := by
  have c97416 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.1
  have c97417 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.1
  have c97418 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97419 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97420 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97421 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1590_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_8
  simp only [← c97416, ← c97417, ← c97418] at e_1590_8
  have e_1590_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_9
  simp only [← c97419, ← c97420, k2, ← c97421] at e_1590_9
  have hr := e_1590_9
  simp only [e_1590_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15486 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1590 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15550)) := by
  have c97422 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97423 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97424 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97425 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97426 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97427 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1590_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_10
  simp only [← c97422, ← c97423, ← c97424] at e_1590_10
  have e_1590_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_11
  simp only [← c97425, ← c97426, k2, ← c97427] at e_1590_11
  have hr := e_1590_11
  simp only [e_1590_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15487 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1590 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15551)) := by
  have c97428 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97429 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97430 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97431 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97432 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97433 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1590_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_12
  simp only [← c97428, ← c97429, ← c97430] at e_1590_12
  have e_1590_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_13
  simp only [← c97431, ← c97432, k2, ← c97433] at e_1590_13
  have hr := e_1590_13
  simp only [e_1590_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15488 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1590 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15552)) := by
  have c97434 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97435 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97436 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97437 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97438 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97439 := (publicBatchWrapper89_copies3044 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1590_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_14
  simp only [← c97434, ← c97435, ← c97436] at e_1590_14
  have e_1590_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_15
  simp only [← c97437, ← c97438, k2, ← c97439] at e_1590_15
  have hr := e_1590_15
  simp only [e_1590_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15489 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1590 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15553)) := by
  have c97440 := (publicBatchWrapper89_copies3045 a h).1
  have c97441 := (publicBatchWrapper89_copies3045 a h).2.1
  have c97442 := (publicBatchWrapper89_copies3045 a h).2.2.1
  have c97443 := (publicBatchWrapper89_copies3045 a h).2.2.2.1
  have c97444 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.1
  have c97445 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1590_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_16
  simp only [← c97440, ← c97441, ← c97442] at e_1590_16
  have e_1590_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_17
  simp only [← c97443, ← c97444, k2, ← c97445] at e_1590_17
  have hr := e_1590_17
  simp only [e_1590_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15490 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1590 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15554)) := by
  have c97446 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.1
  have c97447 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.1
  have c97448 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.1
  have c97449 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.1
  have c97450 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97451 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1590_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_18
  simp only [← c97446, ← c97447, ← c97448] at e_1590_18
  have e_1590_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1590 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1590_19
  simp only [← c97449, ← c97450, k2, ← c97451] at e_1590_19
  have hr := e_1590_19
  simp only [e_1590_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15491 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1591 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15555)) := by
  have c97452 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97453 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97454 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97455 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97456 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97457 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1591_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_0
  simp only [← c97452, ← c97453, ← c97454] at e_1591_0
  have e_1591_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_1
  simp only [← c97455, ← c97456, k2, ← c97457] at e_1591_1
  have hr := e_1591_1
  simp only [e_1591_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15492 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1591 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15556)) := by
  have c97458 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97459 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97460 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97461 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97462 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97463 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1591_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_2
  simp only [← c97458, ← c97459, ← c97460] at e_1591_2
  have e_1591_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_3
  simp only [← c97461, ← c97462, k2, ← c97463] at e_1591_3
  have hr := e_1591_3
  simp only [e_1591_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15493 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1591 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15557)) := by
  have c97464 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97465 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97466 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97467 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97468 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97469 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1591_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_4
  simp only [← c97464, ← c97465, ← c97466] at e_1591_4
  have e_1591_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_5
  simp only [← c97467, ← c97468, k2, ← c97469] at e_1591_5
  have hr := e_1591_5
  simp only [e_1591_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15494 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1591 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15558)) := by
  have c97470 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97471 := (publicBatchWrapper89_copies3045 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97472 := (publicBatchWrapper89_copies3046 a h).1
  have c97473 := (publicBatchWrapper89_copies3046 a h).2.1
  have c97474 := (publicBatchWrapper89_copies3046 a h).2.2.1
  have c97475 := (publicBatchWrapper89_copies3046 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1591_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_6
  simp only [← c97470, ← c97471, ← c97472] at e_1591_6
  have e_1591_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_7
  simp only [← c97473, ← c97474, k2, ← c97475] at e_1591_7
  have hr := e_1591_7
  simp only [e_1591_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15495 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1591 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15559)) := by
  have c97476 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.1
  have c97477 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.1
  have c97478 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.1
  have c97479 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.1
  have c97480 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.1
  have c97481 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1591_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_8
  simp only [← c97476, ← c97477, ← c97478] at e_1591_8
  have e_1591_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_9
  simp only [← c97479, ← c97480, k2, ← c97481] at e_1591_9
  have hr := e_1591_9
  simp only [e_1591_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15496 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1591 47) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15560)) := by
  have c97482 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97483 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97484 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97485 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97486 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97487 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1591_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_10
  simp only [← c97482, ← c97483, ← c97484] at e_1591_10
  have e_1591_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_11
  simp only [← c97485, ← c97486, k2, ← c97487] at e_1591_11
  have hr := e_1591_11
  simp only [e_1591_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15497 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1591 55) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15561)) := by
  have c97488 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97489 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97490 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97491 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97492 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97493 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1591_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_12
  simp only [← c97488, ← c97489, ← c97490] at e_1591_12
  have e_1591_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_13
  simp only [← c97491, ← c97492, k2, ← c97493] at e_1591_13
  have hr := e_1591_13
  simp only [e_1591_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15498 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1591 63) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15562)) := by
  have c97494 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97495 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97496 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97497 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97498 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97499 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1591_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_14
  simp only [← c97494, ← c97495, ← c97496] at e_1591_14
  have e_1591_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_15
  simp only [← c97497, ← c97498, k2, ← c97499] at e_1591_15
  have hr := e_1591_15
  simp only [e_1591_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15499 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1591 71) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15563)) := by
  have c97500 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97501 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97502 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97503 := (publicBatchWrapper89_copies3046 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97504 := (publicBatchWrapper89_copies3047 a h).1
  have c97505 := (publicBatchWrapper89_copies3047 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1591_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_16
  simp only [← c97500, ← c97501, ← c97502] at e_1591_16
  have e_1591_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_17
  simp only [← c97503, ← c97504, k2, ← c97505] at e_1591_17
  have hr := e_1591_17
  simp only [e_1591_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15500 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1591 79) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15564)) := by
  have c97506 := (publicBatchWrapper89_copies3047 a h).2.2.1
  have c97507 := (publicBatchWrapper89_copies3047 a h).2.2.2.1
  have c97508 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.1
  have c97509 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.1
  have c97510 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.1
  have c97511 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1591_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_18
  simp only [← c97506, ← c97507, ← c97508] at e_1591_18
  have e_1591_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1591 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1591_19
  simp only [← c97509, ← c97510, k2, ← c97511] at e_1591_19
  have hr := e_1591_19
  simp only [e_1591_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15501 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1592 7) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15565)) := by
  have c97512 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.1
  have c97513 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.1
  have c97514 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97515 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97516 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97517 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1592_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_0
  simp only [← c97512, ← c97513, ← c97514] at e_1592_0
  have e_1592_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_1
  simp only [← c97515, ← c97516, k2, ← c97517] at e_1592_1
  have hr := e_1592_1
  simp only [e_1592_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15502 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1592 15) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15566)) := by
  have c97518 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97519 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97520 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97521 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97522 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97523 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1592_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_2
  simp only [← c97518, ← c97519, ← c97520] at e_1592_2
  have e_1592_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_3
  simp only [← c97521, ← c97522, k2, ← c97523] at e_1592_3
  have hr := e_1592_3
  simp only [e_1592_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15503 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1592 23) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15567)) := by
  have c97524 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97525 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97526 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97527 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97528 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97529 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1592_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_4
  simp only [← c97524, ← c97525, ← c97526] at e_1592_4
  have e_1592_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_5
  simp only [← c97527, ← c97528, k2, ← c97529] at e_1592_5
  have hr := e_1592_5
  simp only [e_1592_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15504 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1592 31) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15568)) := by
  have c97530 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97531 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97532 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97533 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97534 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97535 := (publicBatchWrapper89_copies3047 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1592_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_6
  simp only [← c97530, ← c97531, ← c97532] at e_1592_6
  have e_1592_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_7
  simp only [← c97533, ← c97534, k2, ← c97535] at e_1592_7
  have hr := e_1592_7
  simp only [e_1592_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15505 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1592 39) = bselect (a (.wire 76 7)) (a (.virt 200)) (a (.virt 15569)) := by
  have c97536 := (publicBatchWrapper89_copies3048 a h).1
  have c97537 := (publicBatchWrapper89_copies3048 a h).2.1
  have c97538 := (publicBatchWrapper89_copies3048 a h).2.2.1
  have c97539 := (publicBatchWrapper89_copies3048 a h).2.2.2.1
  have c97540 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.1
  have c97541 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1592_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_8
  simp only [← c97536, ← c97537, ← c97538] at e_1592_8
  have e_1592_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_9
  simp only [← c97539, ← c97540, k2, ← c97541] at e_1592_9
  have hr := e_1592_9
  simp only [e_1592_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15506 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1592 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15610)) := by
  have c97542 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.1
  have c97543 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.1
  have c97544 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.1
  have c97545 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.1
  have c97546 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97547 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1592_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_10
  simp only [← c97542, ← c97543, ← c97544] at e_1592_10
  have e_1592_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_11
  simp only [← c97545, ← c97546, k2, ← c97547] at e_1592_11
  have hr := e_1592_11
  simp only [e_1592_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15507 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1592 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15611)) := by
  have c97548 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97549 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97550 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97551 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97552 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97553 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1592_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_12
  simp only [← c97548, ← c97549, ← c97550] at e_1592_12
  have e_1592_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_13
  simp only [← c97551, ← c97552, k2, ← c97553] at e_1592_13
  have hr := e_1592_13
  simp only [e_1592_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15508 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1592 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15612)) := by
  have c97554 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97555 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97556 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97557 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97558 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97559 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1592_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_14
  simp only [← c97554, ← c97555, ← c97556] at e_1592_14
  have e_1592_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_15
  simp only [← c97557, ← c97558, k2, ← c97559] at e_1592_15
  have hr := e_1592_15
  simp only [e_1592_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15509 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1592 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15613)) := by
  have c97560 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97561 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97562 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97563 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97564 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97565 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1592_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_16
  simp only [← c97560, ← c97561, ← c97562] at e_1592_16
  have e_1592_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_17
  simp only [← c97563, ← c97564, k2, ← c97565] at e_1592_17
  have hr := e_1592_17
  simp only [e_1592_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15510 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1592 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15614)) := by
  have c97566 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97567 := (publicBatchWrapper89_copies3048 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97568 := (publicBatchWrapper89_copies3049 a h).1
  have c97569 := (publicBatchWrapper89_copies3049 a h).2.1
  have c97570 := (publicBatchWrapper89_copies3049 a h).2.2.1
  have c97571 := (publicBatchWrapper89_copies3049 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1592_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_18
  simp only [← c97566, ← c97567, ← c97568] at e_1592_18
  have e_1592_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1592 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1592_19
  simp only [← c97569, ← c97570, k2, ← c97571] at e_1592_19
  have hr := e_1592_19
  simp only [e_1592_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15511 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1593 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15615)) := by
  have c97572 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.1
  have c97573 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.1
  have c97574 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.1
  have c97575 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.1
  have c97576 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.1
  have c97577 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1593_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_0
  simp only [← c97572, ← c97573, ← c97574] at e_1593_0
  have e_1593_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_1
  simp only [← c97575, ← c97576, k2, ← c97577] at e_1593_1
  have hr := e_1593_1
  simp only [e_1593_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15512 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1593 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15616)) := by
  have c97578 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97579 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97580 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97581 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97582 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97583 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1593_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_2
  simp only [← c97578, ← c97579, ← c97580] at e_1593_2
  have e_1593_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_3
  simp only [← c97581, ← c97582, k2, ← c97583] at e_1593_3
  have hr := e_1593_3
  simp only [e_1593_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15513 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1593 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15617)) := by
  have c97584 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97585 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97586 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97587 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97588 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97589 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1593_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_4
  simp only [← c97584, ← c97585, ← c97586] at e_1593_4
  have e_1593_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_5
  simp only [← c97587, ← c97588, k2, ← c97589] at e_1593_5
  have hr := e_1593_5
  simp only [e_1593_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15514 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1593 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15618)) := by
  have c97590 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97591 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97592 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97593 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97594 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97595 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1593_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_6
  simp only [← c97590, ← c97591, ← c97592] at e_1593_6
  have e_1593_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_7
  simp only [← c97593, ← c97594, k2, ← c97595] at e_1593_7
  have hr := e_1593_7
  simp only [e_1593_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15515 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1593 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15619)) := by
  have c97596 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97597 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97598 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97599 := (publicBatchWrapper89_copies3049 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97600 := (publicBatchWrapper89_copies3050 a h).1
  have c97601 := (publicBatchWrapper89_copies3050 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1593_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_8
  simp only [← c97596, ← c97597, ← c97598] at e_1593_8
  have e_1593_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_9
  simp only [← c97599, ← c97600, k2, ← c97601] at e_1593_9
  have hr := e_1593_9
  simp only [e_1593_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15516 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1593 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15620)) := by
  have c97602 := (publicBatchWrapper89_copies3050 a h).2.2.1
  have c97603 := (publicBatchWrapper89_copies3050 a h).2.2.2.1
  have c97604 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.1
  have c97605 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.1
  have c97606 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.1
  have c97607 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1593_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_10
  simp only [← c97602, ← c97603, ← c97604] at e_1593_10
  have e_1593_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_11
  simp only [← c97605, ← c97606, k2, ← c97607] at e_1593_11
  have hr := e_1593_11
  simp only [e_1593_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15517 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1593 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15621)) := by
  have c97608 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.1
  have c97609 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.1
  have c97610 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97611 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97612 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97613 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1593_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_12
  simp only [← c97608, ← c97609, ← c97610] at e_1593_12
  have e_1593_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_13
  simp only [← c97611, ← c97612, k2, ← c97613] at e_1593_13
  have hr := e_1593_13
  simp only [e_1593_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15518 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1593 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15622)) := by
  have c97614 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97615 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97616 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97617 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97618 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97619 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1593_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_14
  simp only [← c97614, ← c97615, ← c97616] at e_1593_14
  have e_1593_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_15
  simp only [← c97617, ← c97618, k2, ← c97619] at e_1593_15
  have hr := e_1593_15
  simp only [e_1593_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15519 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1593 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15623)) := by
  have c97620 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97621 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97622 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97623 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97624 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97625 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1593_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_16
  simp only [← c97620, ← c97621, ← c97622] at e_1593_16
  have e_1593_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_17
  simp only [← c97623, ← c97624, k2, ← c97625] at e_1593_17
  have hr := e_1593_17
  simp only [e_1593_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15520 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1593 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15624)) := by
  have c97626 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97627 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97628 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97629 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97630 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97631 := (publicBatchWrapper89_copies3050 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1593_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_18
  simp only [← c97626, ← c97627, ← c97628] at e_1593_18
  have e_1593_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1593 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1593_19
  simp only [← c97629, ← c97630, k2, ← c97631] at e_1593_19
  have hr := e_1593_19
  simp only [e_1593_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15521 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1594 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15625)) := by
  have c97632 := (publicBatchWrapper89_copies3051 a h).1
  have c97633 := (publicBatchWrapper89_copies3051 a h).2.1
  have c97634 := (publicBatchWrapper89_copies3051 a h).2.2.1
  have c97635 := (publicBatchWrapper89_copies3051 a h).2.2.2.1
  have c97636 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.1
  have c97637 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1594_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_0
  simp only [← c97632, ← c97633, ← c97634] at e_1594_0
  have e_1594_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_1
  simp only [← c97635, ← c97636, k2, ← c97637] at e_1594_1
  have hr := e_1594_1
  simp only [e_1594_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15522 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1594 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15626)) := by
  have c97638 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.1
  have c97639 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.1
  have c97640 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.1
  have c97641 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.1
  have c97642 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97643 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1594_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_2
  simp only [← c97638, ← c97639, ← c97640] at e_1594_2
  have e_1594_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_3
  simp only [← c97641, ← c97642, k2, ← c97643] at e_1594_3
  have hr := e_1594_3
  simp only [e_1594_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15523 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1594 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15627)) := by
  have c97644 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97645 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97646 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97647 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97648 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97649 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1594_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_4
  simp only [← c97644, ← c97645, ← c97646] at e_1594_4
  have e_1594_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_5
  simp only [← c97647, ← c97648, k2, ← c97649] at e_1594_5
  have hr := e_1594_5
  simp only [e_1594_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15524 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1594 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15628)) := by
  have c97650 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97651 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97652 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97653 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97654 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97655 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1594_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_6
  simp only [← c97650, ← c97651, ← c97652] at e_1594_6
  have e_1594_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_7
  simp only [← c97653, ← c97654, k2, ← c97655] at e_1594_7
  have hr := e_1594_7
  simp only [e_1594_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15525 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1594 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15629)) := by
  have c97656 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97657 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97658 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97659 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97660 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97661 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1594_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_8
  simp only [← c97656, ← c97657, ← c97658] at e_1594_8
  have e_1594_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_9
  simp only [← c97659, ← c97660, k2, ← c97661] at e_1594_9
  have hr := e_1594_9
  simp only [e_1594_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15526 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1594 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15630)) := by
  have c97662 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97663 := (publicBatchWrapper89_copies3051 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97664 := (publicBatchWrapper89_copies3052 a h).1
  have c97665 := (publicBatchWrapper89_copies3052 a h).2.1
  have c97666 := (publicBatchWrapper89_copies3052 a h).2.2.1
  have c97667 := (publicBatchWrapper89_copies3052 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1594_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_10
  simp only [← c97662, ← c97663, ← c97664] at e_1594_10
  have e_1594_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_11
  simp only [← c97665, ← c97666, k2, ← c97667] at e_1594_11
  have hr := e_1594_11
  simp only [e_1594_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15527 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1594 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15631)) := by
  have c97668 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.1
  have c97669 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.1
  have c97670 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.1
  have c97671 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.1
  have c97672 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.1
  have c97673 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1594_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_12
  simp only [← c97668, ← c97669, ← c97670] at e_1594_12
  have e_1594_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_13
  simp only [← c97671, ← c97672, k2, ← c97673] at e_1594_13
  have hr := e_1594_13
  simp only [e_1594_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15528 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1594 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15632)) := by
  have c97674 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97675 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97676 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97677 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97678 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97679 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1594_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_14
  simp only [← c97674, ← c97675, ← c97676] at e_1594_14
  have e_1594_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_15
  simp only [← c97677, ← c97678, k2, ← c97679] at e_1594_15
  have hr := e_1594_15
  simp only [e_1594_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15529 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1594 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15633)) := by
  have c97680 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97681 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97682 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97683 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97684 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97685 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1594_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_16
  simp only [← c97680, ← c97681, ← c97682] at e_1594_16
  have e_1594_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_17
  simp only [← c97683, ← c97684, k2, ← c97685] at e_1594_17
  have hr := e_1594_17
  simp only [e_1594_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15530 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1594 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15634)) := by
  have c97686 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97687 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97688 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97689 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97690 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97691 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1594_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_18
  simp only [← c97686, ← c97687, ← c97688] at e_1594_18
  have e_1594_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1594 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1594_19
  simp only [← c97689, ← c97690, k2, ← c97691] at e_1594_19
  have hr := e_1594_19
  simp only [e_1594_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15531 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1595 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15635)) := by
  have c97692 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97693 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97694 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97695 := (publicBatchWrapper89_copies3052 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97696 := (publicBatchWrapper89_copies3053 a h).1
  have c97697 := (publicBatchWrapper89_copies3053 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1595_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_0
  simp only [← c97692, ← c97693, ← c97694] at e_1595_0
  have e_1595_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_1
  simp only [← c97695, ← c97696, k2, ← c97697] at e_1595_1
  have hr := e_1595_1
  simp only [e_1595_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15532 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1595 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15636)) := by
  have c97698 := (publicBatchWrapper89_copies3053 a h).2.2.1
  have c97699 := (publicBatchWrapper89_copies3053 a h).2.2.2.1
  have c97700 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.1
  have c97701 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.1
  have c97702 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.1
  have c97703 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1595_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_2
  simp only [← c97698, ← c97699, ← c97700] at e_1595_2
  have e_1595_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_3
  simp only [← c97701, ← c97702, k2, ← c97703] at e_1595_3
  have hr := e_1595_3
  simp only [e_1595_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15533 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1595 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15637)) := by
  have c97704 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.1
  have c97705 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.1
  have c97706 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97707 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97708 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97709 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1595_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_4
  simp only [← c97704, ← c97705, ← c97706] at e_1595_4
  have e_1595_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_5
  simp only [← c97707, ← c97708, k2, ← c97709] at e_1595_5
  have hr := e_1595_5
  simp only [e_1595_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15534 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1595 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15638)) := by
  have c97710 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97711 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97712 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97713 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97714 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97715 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1595_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_6
  simp only [← c97710, ← c97711, ← c97712] at e_1595_6
  have e_1595_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_7
  simp only [← c97713, ← c97714, k2, ← c97715] at e_1595_7
  have hr := e_1595_7
  simp only [e_1595_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15535 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1595 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15639)) := by
  have c97716 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97717 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97718 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97719 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97720 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97721 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1595_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_8
  simp only [← c97716, ← c97717, ← c97718] at e_1595_8
  have e_1595_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_9
  simp only [← c97719, ← c97720, k2, ← c97721] at e_1595_9
  have hr := e_1595_9
  simp only [e_1595_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15536 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1595 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15640)) := by
  have c97722 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97723 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97724 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97725 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97726 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97727 := (publicBatchWrapper89_copies3053 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1595_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_10
  simp only [← c97722, ← c97723, ← c97724] at e_1595_10
  have e_1595_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_11
  simp only [← c97725, ← c97726, k2, ← c97727] at e_1595_11
  have hr := e_1595_11
  simp only [e_1595_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15537 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1595 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15641)) := by
  have c97728 := (publicBatchWrapper89_copies3054 a h).1
  have c97729 := (publicBatchWrapper89_copies3054 a h).2.1
  have c97730 := (publicBatchWrapper89_copies3054 a h).2.2.1
  have c97731 := (publicBatchWrapper89_copies3054 a h).2.2.2.1
  have c97732 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.1
  have c97733 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1595_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_12
  simp only [← c97728, ← c97729, ← c97730] at e_1595_12
  have e_1595_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_13
  simp only [← c97731, ← c97732, k2, ← c97733] at e_1595_13
  have hr := e_1595_13
  simp only [e_1595_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15538 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1595 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15642)) := by
  have c97734 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.1
  have c97735 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.1
  have c97736 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.1
  have c97737 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.1
  have c97738 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97739 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1595_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_14
  simp only [← c97734, ← c97735, ← c97736] at e_1595_14
  have e_1595_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_15
  simp only [← c97737, ← c97738, k2, ← c97739] at e_1595_15
  have hr := e_1595_15
  simp only [e_1595_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15539 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1595 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15643)) := by
  have c97740 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97741 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97742 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97743 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97744 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97745 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1595_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_16
  simp only [← c97740, ← c97741, ← c97742] at e_1595_16
  have e_1595_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_17
  simp only [← c97743, ← c97744, k2, ← c97745] at e_1595_17
  have hr := e_1595_17
  simp only [e_1595_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15540 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1595 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15644)) := by
  have c97746 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97747 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97748 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97749 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97750 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97751 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1595_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_18
  simp only [← c97746, ← c97747, ← c97748] at e_1595_18
  have e_1595_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1595 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1595_19
  simp only [← c97749, ← c97750, k2, ← c97751] at e_1595_19
  have hr := e_1595_19
  simp only [e_1595_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15541 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1596 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15645)) := by
  have c97752 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97753 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97754 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97755 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97756 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97757 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1596_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_0
  simp only [← c97752, ← c97753, ← c97754] at e_1596_0
  have e_1596_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_1
  simp only [← c97755, ← c97756, k2, ← c97757] at e_1596_1
  have hr := e_1596_1
  simp only [e_1596_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15542 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1596 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15646)) := by
  have c97758 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97759 := (publicBatchWrapper89_copies3054 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97760 := (publicBatchWrapper89_copies3055 a h).1
  have c97761 := (publicBatchWrapper89_copies3055 a h).2.1
  have c97762 := (publicBatchWrapper89_copies3055 a h).2.2.1
  have c97763 := (publicBatchWrapper89_copies3055 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1596_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_2
  simp only [← c97758, ← c97759, ← c97760] at e_1596_2
  have e_1596_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_3
  simp only [← c97761, ← c97762, k2, ← c97763] at e_1596_3
  have hr := e_1596_3
  simp only [e_1596_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1596 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15647)) := by
  have c97764 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.1
  have c97765 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.1
  have c97766 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.1
  have c97767 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.1
  have c97768 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.1
  have c97769 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1596_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_4
  simp only [← c97764, ← c97765, ← c97766] at e_1596_4
  have e_1596_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_5
  simp only [← c97767, ← c97768, k2, ← c97769] at e_1596_5
  have hr := e_1596_5
  simp only [e_1596_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1596 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15648)) := by
  have c97770 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97771 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97772 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97773 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97774 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97775 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1596_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_6
  simp only [← c97770, ← c97771, ← c97772] at e_1596_6
  have e_1596_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_7
  simp only [← c97773, ← c97774, k2, ← c97775] at e_1596_7
  have hr := e_1596_7
  simp only [e_1596_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1596 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15649)) := by
  have c97776 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97777 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97778 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97779 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97780 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97781 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1596_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_8
  simp only [← c97776, ← c97777, ← c97778] at e_1596_8
  have e_1596_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_9
  simp only [← c97779, ← c97780, k2, ← c97781] at e_1596_9
  have hr := e_1596_9
  simp only [e_1596_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1596 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15650)) := by
  have c97782 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97783 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97784 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97785 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97786 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97787 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1596_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_10
  simp only [← c97782, ← c97783, ← c97784] at e_1596_10
  have e_1596_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_11
  simp only [← c97785, ← c97786, k2, ← c97787] at e_1596_11
  have hr := e_1596_11
  simp only [e_1596_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1596 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15651)) := by
  have c97788 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97789 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97790 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97791 := (publicBatchWrapper89_copies3055 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97792 := (publicBatchWrapper89_copies3056 a h).1
  have c97793 := (publicBatchWrapper89_copies3056 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1596_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_12
  simp only [← c97788, ← c97789, ← c97790] at e_1596_12
  have e_1596_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_13
  simp only [← c97791, ← c97792, k2, ← c97793] at e_1596_13
  have hr := e_1596_13
  simp only [e_1596_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1596 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15652)) := by
  have c97794 := (publicBatchWrapper89_copies3056 a h).2.2.1
  have c97795 := (publicBatchWrapper89_copies3056 a h).2.2.2.1
  have c97796 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.1
  have c97797 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.1
  have c97798 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.1
  have c97799 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1596_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_14
  simp only [← c97794, ← c97795, ← c97796] at e_1596_14
  have e_1596_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_15
  simp only [← c97797, ← c97798, k2, ← c97799] at e_1596_15
  have hr := e_1596_15
  simp only [e_1596_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1596 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15653)) := by
  have c97800 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.1
  have c97801 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.1
  have c97802 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97803 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97804 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97805 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1596_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_16
  simp only [← c97800, ← c97801, ← c97802] at e_1596_16
  have e_1596_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_17
  simp only [← c97803, ← c97804, k2, ← c97805] at e_1596_17
  have hr := e_1596_17
  simp only [e_1596_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1596 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15654)) := by
  have c97806 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97807 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97808 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97809 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97810 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97811 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1596_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_18
  simp only [← c97806, ← c97807, ← c97808] at e_1596_18
  have e_1596_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1596 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1596_19
  simp only [← c97809, ← c97810, k2, ← c97811] at e_1596_19
  have hr := e_1596_19
  simp only [e_1596_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1597 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15655)) := by
  have c97812 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97813 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97814 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97815 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97816 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97817 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1597_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_0
  simp only [← c97812, ← c97813, ← c97814] at e_1597_0
  have e_1597_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_1
  simp only [← c97815, ← c97816, k2, ← c97817] at e_1597_1
  have hr := e_1597_1
  simp only [e_1597_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1597 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15656)) := by
  have c97818 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97819 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97820 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97821 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97822 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97823 := (publicBatchWrapper89_copies3056 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1597_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_2
  simp only [← c97818, ← c97819, ← c97820] at e_1597_2
  have e_1597_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_3
  simp only [← c97821, ← c97822, k2, ← c97823] at e_1597_3
  have hr := e_1597_3
  simp only [e_1597_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1597 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15657)) := by
  have c97824 := (publicBatchWrapper89_copies3057 a h).1
  have c97825 := (publicBatchWrapper89_copies3057 a h).2.1
  have c97826 := (publicBatchWrapper89_copies3057 a h).2.2.1
  have c97827 := (publicBatchWrapper89_copies3057 a h).2.2.2.1
  have c97828 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.1
  have c97829 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1597_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_4
  simp only [← c97824, ← c97825, ← c97826] at e_1597_4
  have e_1597_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_5
  simp only [← c97827, ← c97828, k2, ← c97829] at e_1597_5
  have hr := e_1597_5
  simp only [e_1597_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1597 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15658)) := by
  have c97830 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.1
  have c97831 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.1
  have c97832 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.1
  have c97833 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.1
  have c97834 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97835 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1597_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_6
  simp only [← c97830, ← c97831, ← c97832] at e_1597_6
  have e_1597_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_7
  simp only [← c97833, ← c97834, k2, ← c97835] at e_1597_7
  have hr := e_1597_7
  simp only [e_1597_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1597 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15659)) := by
  have c97836 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97837 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97838 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97839 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97840 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97841 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1597_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_8
  simp only [← c97836, ← c97837, ← c97838] at e_1597_8
  have e_1597_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_9
  simp only [← c97839, ← c97840, k2, ← c97841] at e_1597_9
  have hr := e_1597_9
  simp only [e_1597_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1597 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15660)) := by
  have c97842 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97843 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97844 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97845 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97846 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97847 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1597_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_10
  simp only [← c97842, ← c97843, ← c97844] at e_1597_10
  have e_1597_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_11
  simp only [← c97845, ← c97846, k2, ← c97847] at e_1597_11
  have hr := e_1597_11
  simp only [e_1597_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1597 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15661)) := by
  have c97848 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97849 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97850 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97851 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97852 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97853 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1597_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_12
  simp only [← c97848, ← c97849, ← c97850] at e_1597_12
  have e_1597_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_13
  simp only [← c97851, ← c97852, k2, ← c97853] at e_1597_13
  have hr := e_1597_13
  simp only [e_1597_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1597 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15662)) := by
  have c97854 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97855 := (publicBatchWrapper89_copies3057 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97856 := (publicBatchWrapper89_copies3058 a h).1
  have c97857 := (publicBatchWrapper89_copies3058 a h).2.1
  have c97858 := (publicBatchWrapper89_copies3058 a h).2.2.1
  have c97859 := (publicBatchWrapper89_copies3058 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1597_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_14
  simp only [← c97854, ← c97855, ← c97856] at e_1597_14
  have e_1597_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_15
  simp only [← c97857, ← c97858, k2, ← c97859] at e_1597_15
  have hr := e_1597_15
  simp only [e_1597_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1597 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15663)) := by
  have c97860 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.1
  have c97861 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.1
  have c97862 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.1
  have c97863 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.1
  have c97864 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.1
  have c97865 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1597_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_16
  simp only [← c97860, ← c97861, ← c97862] at e_1597_16
  have e_1597_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_17
  simp only [← c97863, ← c97864, k2, ← c97865] at e_1597_17
  have hr := e_1597_17
  simp only [e_1597_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1597 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15664)) := by
  have c97866 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97867 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97868 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97869 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97870 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97871 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1597_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_18
  simp only [← c97866, ← c97867, ← c97868] at e_1597_18
  have e_1597_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1597 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1597_19
  simp only [← c97869, ← c97870, k2, ← c97871] at e_1597_19
  have hr := e_1597_19
  simp only [e_1597_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1598 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15665)) := by
  have c97872 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97873 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97874 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97875 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97876 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97877 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1598_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_0
  simp only [← c97872, ← c97873, ← c97874] at e_1598_0
  have e_1598_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_1
  simp only [← c97875, ← c97876, k2, ← c97877] at e_1598_1
  have hr := e_1598_1
  simp only [e_1598_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1598 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15666)) := by
  have c97878 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97879 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97880 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97881 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97882 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97883 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1598_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_2
  simp only [← c97878, ← c97879, ← c97880] at e_1598_2
  have e_1598_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_3
  simp only [← c97881, ← c97882, k2, ← c97883] at e_1598_3
  have hr := e_1598_3
  simp only [e_1598_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1598 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15667)) := by
  have c97884 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97885 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97886 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97887 := (publicBatchWrapper89_copies3058 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97888 := (publicBatchWrapper89_copies3059 a h).1
  have c97889 := (publicBatchWrapper89_copies3059 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1598_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_4
  simp only [← c97884, ← c97885, ← c97886] at e_1598_4
  have e_1598_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_5
  simp only [← c97887, ← c97888, k2, ← c97889] at e_1598_5
  have hr := e_1598_5
  simp only [e_1598_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1598 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15668)) := by
  have c97890 := (publicBatchWrapper89_copies3059 a h).2.2.1
  have c97891 := (publicBatchWrapper89_copies3059 a h).2.2.2.1
  have c97892 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.1
  have c97893 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.1
  have c97894 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.1
  have c97895 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1598_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_6
  simp only [← c97890, ← c97891, ← c97892] at e_1598_6
  have e_1598_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_7
  simp only [← c97893, ← c97894, k2, ← c97895] at e_1598_7
  have hr := e_1598_7
  simp only [e_1598_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1598 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15669)) := by
  have c97896 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.1
  have c97897 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.1
  have c97898 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97899 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97900 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97901 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1598_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_8
  simp only [← c97896, ← c97897, ← c97898] at e_1598_8
  have e_1598_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_9
  simp only [← c97899, ← c97900, k2, ← c97901] at e_1598_9
  have hr := e_1598_9
  simp only [e_1598_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1598 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15670)) := by
  have c97902 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97903 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97904 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97905 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97906 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97907 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1598_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_10
  simp only [← c97902, ← c97903, ← c97904] at e_1598_10
  have e_1598_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_11
  simp only [← c97905, ← c97906, k2, ← c97907] at e_1598_11
  have hr := e_1598_11
  simp only [e_1598_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1598 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15671)) := by
  have c97908 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97909 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97910 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97911 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97912 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97913 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1598_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_12
  simp only [← c97908, ← c97909, ← c97910] at e_1598_12
  have e_1598_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_13
  simp only [← c97911, ← c97912, k2, ← c97913] at e_1598_13
  have hr := e_1598_13
  simp only [e_1598_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1598 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15672)) := by
  have c97914 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97915 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97916 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97917 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97918 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97919 := (publicBatchWrapper89_copies3059 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1598_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_14
  simp only [← c97914, ← c97915, ← c97916] at e_1598_14
  have e_1598_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_15
  simp only [← c97917, ← c97918, k2, ← c97919] at e_1598_15
  have hr := e_1598_15
  simp only [e_1598_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1598 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15673)) := by
  have c97920 := (publicBatchWrapper89_copies3060 a h).1
  have c97921 := (publicBatchWrapper89_copies3060 a h).2.1
  have c97922 := (publicBatchWrapper89_copies3060 a h).2.2.1
  have c97923 := (publicBatchWrapper89_copies3060 a h).2.2.2.1
  have c97924 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.1
  have c97925 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1598_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_16
  simp only [← c97920, ← c97921, ← c97922] at e_1598_16
  have e_1598_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_17
  simp only [← c97923, ← c97924, k2, ← c97925] at e_1598_17
  have hr := e_1598_17
  simp only [e_1598_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1598 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15674)) := by
  have c97926 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.1
  have c97927 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.1
  have c97928 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.1
  have c97929 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.1
  have c97930 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97931 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1598_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_18
  simp only [← c97926, ← c97927, ← c97928] at e_1598_18
  have e_1598_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1598 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1598_19
  simp only [← c97929, ← c97930, k2, ← c97931] at e_1598_19
  have hr := e_1598_19
  simp only [e_1598_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1599 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15675)) := by
  have c97932 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97933 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97934 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97935 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97936 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97937 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1599_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_0
  simp only [← c97932, ← c97933, ← c97934] at e_1599_0
  have e_1599_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_1
  simp only [← c97935, ← c97936, k2, ← c97937] at e_1599_1
  have hr := e_1599_1
  simp only [e_1599_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1599 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15676)) := by
  have c97938 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97939 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97940 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97941 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97942 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97943 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1599_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_2
  simp only [← c97938, ← c97939, ← c97940] at e_1599_2
  have e_1599_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_3
  simp only [← c97941, ← c97942, k2, ← c97943] at e_1599_3
  have hr := e_1599_3
  simp only [e_1599_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1599 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15677)) := by
  have c97944 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97945 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97946 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97947 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97948 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97949 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1599_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_4
  simp only [← c97944, ← c97945, ← c97946] at e_1599_4
  have e_1599_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_5
  simp only [← c97947, ← c97948, k2, ← c97949] at e_1599_5
  have hr := e_1599_5
  simp only [e_1599_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1599 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15678)) := by
  have c97950 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97951 := (publicBatchWrapper89_copies3060 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97952 := (publicBatchWrapper89_copies3061 a h).1
  have c97953 := (publicBatchWrapper89_copies3061 a h).2.1
  have c97954 := (publicBatchWrapper89_copies3061 a h).2.2.1
  have c97955 := (publicBatchWrapper89_copies3061 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1599_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_6
  simp only [← c97950, ← c97951, ← c97952] at e_1599_6
  have e_1599_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_7
  simp only [← c97953, ← c97954, k2, ← c97955] at e_1599_7
  have hr := e_1599_7
  simp only [e_1599_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1599 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15679)) := by
  have c97956 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.1
  have c97957 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.1
  have c97958 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.1
  have c97959 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.1
  have c97960 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.1
  have c97961 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1599_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_8
  simp only [← c97956, ← c97957, ← c97958] at e_1599_8
  have e_1599_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_9
  simp only [← c97959, ← c97960, k2, ← c97961] at e_1599_9
  have hr := e_1599_9
  simp only [e_1599_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1599 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15680)) := by
  have c97962 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97963 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97964 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97965 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97966 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97967 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1599_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_10
  simp only [← c97962, ← c97963, ← c97964] at e_1599_10
  have e_1599_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_11
  simp only [← c97965, ← c97966, k2, ← c97967] at e_1599_11
  have hr := e_1599_11
  simp only [e_1599_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1599 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15681)) := by
  have c97968 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97969 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97970 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97971 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97972 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97973 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1599_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_12
  simp only [← c97968, ← c97969, ← c97970] at e_1599_12
  have e_1599_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_13
  simp only [← c97971, ← c97972, k2, ← c97973] at e_1599_13
  have hr := e_1599_13
  simp only [e_1599_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1599 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15682)) := by
  have c97974 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97975 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97976 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97977 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97978 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97979 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1599_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_14
  simp only [← c97974, ← c97975, ← c97976] at e_1599_14
  have e_1599_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_15
  simp only [← c97977, ← c97978, k2, ← c97979] at e_1599_15
  have hr := e_1599_15
  simp only [e_1599_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1599 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15683)) := by
  have c97980 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97981 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97982 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97983 := (publicBatchWrapper89_copies3061 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c97984 := (publicBatchWrapper89_copies3062 a h).1
  have c97985 := (publicBatchWrapper89_copies3062 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1599_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_16
  simp only [← c97980, ← c97981, ← c97982] at e_1599_16
  have e_1599_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_17
  simp only [← c97983, ← c97984, k2, ← c97985] at e_1599_17
  have hr := e_1599_17
  simp only [e_1599_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1599 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15684)) := by
  have c97986 := (publicBatchWrapper89_copies3062 a h).2.2.1
  have c97987 := (publicBatchWrapper89_copies3062 a h).2.2.2.1
  have c97988 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.1
  have c97989 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.1
  have c97990 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.1
  have c97991 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1599_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_18
  simp only [← c97986, ← c97987, ← c97988] at e_1599_18
  have e_1599_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1599 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1599_19
  simp only [← c97989, ← c97990, k2, ← c97991] at e_1599_19
  have hr := e_1599_19
  simp only [e_1599_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1600 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15685)) := by
  have c97992 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.1
  have c97993 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.1
  have c97994 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.1
  have c97995 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c97996 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97997 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1600_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_0
  simp only [← c97992, ← c97993, ← c97994] at e_1600_0
  have e_1600_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_1
  simp only [← c97995, ← c97996, k2, ← c97997] at e_1600_1
  have hr := e_1600_1
  simp only [e_1600_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1600 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15686)) := by
  have c97998 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c97999 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98000 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98001 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98002 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98003 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1600_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_2
  simp only [← c97998, ← c97999, ← c98000] at e_1600_2
  have e_1600_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_3
  simp only [← c98001, ← c98002, k2, ← c98003] at e_1600_3
  have hr := e_1600_3
  simp only [e_1600_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1600 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15687)) := by
  have c98004 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98005 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98006 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98007 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98008 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98009 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1600_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_4
  simp only [← c98004, ← c98005, ← c98006] at e_1600_4
  have e_1600_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_5
  simp only [← c98007, ← c98008, k2, ← c98009] at e_1600_5
  have hr := e_1600_5
  simp only [e_1600_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1600 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15688)) := by
  have c98010 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98011 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98012 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98013 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98014 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98015 := (publicBatchWrapper89_copies3062 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1600_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_6
  simp only [← c98010, ← c98011, ← c98012] at e_1600_6
  have e_1600_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_7
  simp only [← c98013, ← c98014, k2, ← c98015] at e_1600_7
  have hr := e_1600_7
  simp only [e_1600_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1600 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15689)) := by
  have c98016 := (publicBatchWrapper89_copies3063 a h).1
  have c98017 := (publicBatchWrapper89_copies3063 a h).2.1
  have c98018 := (publicBatchWrapper89_copies3063 a h).2.2.1
  have c98019 := (publicBatchWrapper89_copies3063 a h).2.2.2.1
  have c98020 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.1
  have c98021 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1600_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_8
  simp only [← c98016, ← c98017, ← c98018] at e_1600_8
  have e_1600_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_9
  simp only [← c98019, ← c98020, k2, ← c98021] at e_1600_9
  have hr := e_1600_9
  simp only [e_1600_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1600 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15690)) := by
  have c98022 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.1
  have c98023 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.1
  have c98024 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.1
  have c98025 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.1
  have c98026 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98027 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1600_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_10
  simp only [← c98022, ← c98023, ← c98024] at e_1600_10
  have e_1600_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_11
  simp only [← c98025, ← c98026, k2, ← c98027] at e_1600_11
  have hr := e_1600_11
  simp only [e_1600_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1600 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15691)) := by
  have c98028 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98029 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98030 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98031 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98032 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98033 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1600_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_12
  simp only [← c98028, ← c98029, ← c98030] at e_1600_12
  have e_1600_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_13
  simp only [← c98031, ← c98032, k2, ← c98033] at e_1600_13
  have hr := e_1600_13
  simp only [e_1600_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1600 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15692)) := by
  have c98034 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98035 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98036 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98037 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98038 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98039 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1600_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_14
  simp only [← c98034, ← c98035, ← c98036] at e_1600_14
  have e_1600_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_15
  simp only [← c98037, ← c98038, k2, ← c98039] at e_1600_15
  have hr := e_1600_15
  simp only [e_1600_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1600 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15693)) := by
  have c98040 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98041 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98042 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98043 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98044 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98045 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1600_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_16
  simp only [← c98040, ← c98041, ← c98042] at e_1600_16
  have e_1600_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_17
  simp only [← c98043, ← c98044, k2, ← c98045] at e_1600_17
  have hr := e_1600_17
  simp only [e_1600_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1600 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15694)) := by
  have c98046 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98047 := (publicBatchWrapper89_copies3063 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98048 := (publicBatchWrapper89_copies3064 a h).1
  have c98049 := (publicBatchWrapper89_copies3064 a h).2.1
  have c98050 := (publicBatchWrapper89_copies3064 a h).2.2.1
  have c98051 := (publicBatchWrapper89_copies3064 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1600_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_18
  simp only [← c98046, ← c98047, ← c98048] at e_1600_18
  have e_1600_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1600 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1600_19
  simp only [← c98049, ← c98050, k2, ← c98051] at e_1600_19
  have hr := e_1600_19
  simp only [e_1600_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1601 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15695)) := by
  have c98052 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.1
  have c98053 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.1
  have c98054 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.1
  have c98055 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.1
  have c98056 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.1
  have c98057 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1601_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_0
  simp only [← c98052, ← c98053, ← c98054] at e_1601_0
  have e_1601_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_1
  simp only [← c98055, ← c98056, k2, ← c98057] at e_1601_1
  have hr := e_1601_1
  simp only [e_1601_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1601 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15696)) := by
  have c98058 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98059 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98060 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98061 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98062 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98063 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1601_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_2
  simp only [← c98058, ← c98059, ← c98060] at e_1601_2
  have e_1601_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_3
  simp only [← c98061, ← c98062, k2, ← c98063] at e_1601_3
  have hr := e_1601_3
  simp only [e_1601_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1601 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15697)) := by
  have c98064 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98065 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98066 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98067 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98068 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98069 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1601_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_4
  simp only [← c98064, ← c98065, ← c98066] at e_1601_4
  have e_1601_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_5
  simp only [← c98067, ← c98068, k2, ← c98069] at e_1601_5
  have hr := e_1601_5
  simp only [e_1601_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1601 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15698)) := by
  have c98070 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98071 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98072 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98073 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98074 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98075 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1601_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_6
  simp only [← c98070, ← c98071, ← c98072] at e_1601_6
  have e_1601_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_7
  simp only [← c98073, ← c98074, k2, ← c98075] at e_1601_7
  have hr := e_1601_7
  simp only [e_1601_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1601 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15699)) := by
  have c98076 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98077 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98078 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98079 := (publicBatchWrapper89_copies3064 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98080 := (publicBatchWrapper89_copies3065 a h).1
  have c98081 := (publicBatchWrapper89_copies3065 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1601_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_8
  simp only [← c98076, ← c98077, ← c98078] at e_1601_8
  have e_1601_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_9
  simp only [← c98079, ← c98080, k2, ← c98081] at e_1601_9
  have hr := e_1601_9
  simp only [e_1601_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1601 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15700)) := by
  have c98082 := (publicBatchWrapper89_copies3065 a h).2.2.1
  have c98083 := (publicBatchWrapper89_copies3065 a h).2.2.2.1
  have c98084 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.1
  have c98085 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.1
  have c98086 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.1
  have c98087 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1601_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_10
  simp only [← c98082, ← c98083, ← c98084] at e_1601_10
  have e_1601_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_11
  simp only [← c98085, ← c98086, k2, ← c98087] at e_1601_11
  have hr := e_1601_11
  simp only [e_1601_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1601 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15701)) := by
  have c98088 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.1
  have c98089 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.1
  have c98090 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98091 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98092 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98093 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1601_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_12
  simp only [← c98088, ← c98089, ← c98090] at e_1601_12
  have e_1601_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_13
  simp only [← c98091, ← c98092, k2, ← c98093] at e_1601_13
  have hr := e_1601_13
  simp only [e_1601_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1601 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15702)) := by
  have c98094 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98095 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98096 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98097 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98098 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98099 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1601_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_14
  simp only [← c98094, ← c98095, ← c98096] at e_1601_14
  have e_1601_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_15
  simp only [← c98097, ← c98098, k2, ← c98099] at e_1601_15
  have hr := e_1601_15
  simp only [e_1601_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1601 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15703)) := by
  have c98100 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98101 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98102 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98103 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98104 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98105 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1601_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_16
  simp only [← c98100, ← c98101, ← c98102] at e_1601_16
  have e_1601_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_17
  simp only [← c98103, ← c98104, k2, ← c98105] at e_1601_17
  have hr := e_1601_17
  simp only [e_1601_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1601 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15704)) := by
  have c98106 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98107 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98108 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98109 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98110 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98111 := (publicBatchWrapper89_copies3065 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1601_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_18
  simp only [← c98106, ← c98107, ← c98108] at e_1601_18
  have e_1601_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1601 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1601_19
  simp only [← c98109, ← c98110, k2, ← c98111] at e_1601_19
  have hr := e_1601_19
  simp only [e_1601_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1602 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15705)) := by
  have c98112 := (publicBatchWrapper89_copies3066 a h).1
  have c98113 := (publicBatchWrapper89_copies3066 a h).2.1
  have c98114 := (publicBatchWrapper89_copies3066 a h).2.2.1
  have c98115 := (publicBatchWrapper89_copies3066 a h).2.2.2.1
  have c98116 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.1
  have c98117 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1602_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_0
  simp only [← c98112, ← c98113, ← c98114] at e_1602_0
  have e_1602_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_1
  simp only [← c98115, ← c98116, k2, ← c98117] at e_1602_1
  have hr := e_1602_1
  simp only [e_1602_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1602 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15706)) := by
  have c98118 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.1
  have c98119 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.1
  have c98120 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.1
  have c98121 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.1
  have c98122 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98123 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1602_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_2
  simp only [← c98118, ← c98119, ← c98120] at e_1602_2
  have e_1602_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_3
  simp only [← c98121, ← c98122, k2, ← c98123] at e_1602_3
  have hr := e_1602_3
  simp only [e_1602_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1602 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15707)) := by
  have c98124 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98125 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98126 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98127 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98128 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98129 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1602_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_4
  simp only [← c98124, ← c98125, ← c98126] at e_1602_4
  have e_1602_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_5
  simp only [← c98127, ← c98128, k2, ← c98129] at e_1602_5
  have hr := e_1602_5
  simp only [e_1602_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1602 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15708)) := by
  have c98130 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98131 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98132 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98133 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98134 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98135 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1602_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_6
  simp only [← c98130, ← c98131, ← c98132] at e_1602_6
  have e_1602_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_7
  simp only [← c98133, ← c98134, k2, ← c98135] at e_1602_7
  have hr := e_1602_7
  simp only [e_1602_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1602 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15709)) := by
  have c98136 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98137 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98138 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98139 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98140 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98141 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1602_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_8
  simp only [← c98136, ← c98137, ← c98138] at e_1602_8
  have e_1602_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_9
  simp only [← c98139, ← c98140, k2, ← c98141] at e_1602_9
  have hr := e_1602_9
  simp only [e_1602_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1602 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15710)) := by
  have c98142 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98143 := (publicBatchWrapper89_copies3066 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98144 := (publicBatchWrapper89_copies3067 a h).1
  have c98145 := (publicBatchWrapper89_copies3067 a h).2.1
  have c98146 := (publicBatchWrapper89_copies3067 a h).2.2.1
  have c98147 := (publicBatchWrapper89_copies3067 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1602_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_10
  simp only [← c98142, ← c98143, ← c98144] at e_1602_10
  have e_1602_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_11
  simp only [← c98145, ← c98146, k2, ← c98147] at e_1602_11
  have hr := e_1602_11
  simp only [e_1602_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1602 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15711)) := by
  have c98148 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.1
  have c98149 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.1
  have c98150 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.1
  have c98151 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.1
  have c98152 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.1
  have c98153 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1602_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_12
  simp only [← c98148, ← c98149, ← c98150] at e_1602_12
  have e_1602_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_13
  simp only [← c98151, ← c98152, k2, ← c98153] at e_1602_13
  have hr := e_1602_13
  simp only [e_1602_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1602 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15712)) := by
  have c98154 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98155 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98156 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98157 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98158 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98159 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1602_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_14
  simp only [← c98154, ← c98155, ← c98156] at e_1602_14
  have e_1602_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_15
  simp only [← c98157, ← c98158, k2, ← c98159] at e_1602_15
  have hr := e_1602_15
  simp only [e_1602_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1602 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15713)) := by
  have c98160 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98161 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98162 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98163 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98164 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98165 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1602_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_16
  simp only [← c98160, ← c98161, ← c98162] at e_1602_16
  have e_1602_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_17
  simp only [← c98163, ← c98164, k2, ← c98165] at e_1602_17
  have hr := e_1602_17
  simp only [e_1602_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1602 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15714)) := by
  have c98166 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98167 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98168 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98169 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98170 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98171 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1602_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_18
  simp only [← c98166, ← c98167, ← c98168] at e_1602_18
  have e_1602_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1602 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1602_19
  simp only [← c98169, ← c98170, k2, ← c98171] at e_1602_19
  have hr := e_1602_19
  simp only [e_1602_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1603 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15715)) := by
  have c98172 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98173 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98174 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98175 := (publicBatchWrapper89_copies3067 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98176 := (publicBatchWrapper89_copies3068 a h).1
  have c98177 := (publicBatchWrapper89_copies3068 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1603_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_0
  simp only [← c98172, ← c98173, ← c98174] at e_1603_0
  have e_1603_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_1
  simp only [← c98175, ← c98176, k2, ← c98177] at e_1603_1
  have hr := e_1603_1
  simp only [e_1603_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1603 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15716)) := by
  have c98178 := (publicBatchWrapper89_copies3068 a h).2.2.1
  have c98179 := (publicBatchWrapper89_copies3068 a h).2.2.2.1
  have c98180 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.1
  have c98181 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.1
  have c98182 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.1
  have c98183 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1603_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_2
  simp only [← c98178, ← c98179, ← c98180] at e_1603_2
  have e_1603_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_3
  simp only [← c98181, ← c98182, k2, ← c98183] at e_1603_3
  have hr := e_1603_3
  simp only [e_1603_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1603 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15717)) := by
  have c98184 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.1
  have c98185 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.1
  have c98186 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98187 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98188 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98189 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1603_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_4
  simp only [← c98184, ← c98185, ← c98186] at e_1603_4
  have e_1603_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_5
  simp only [← c98187, ← c98188, k2, ← c98189] at e_1603_5
  have hr := e_1603_5
  simp only [e_1603_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1603 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15718)) := by
  have c98190 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98191 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98192 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98193 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98194 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98195 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1603_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_6
  simp only [← c98190, ← c98191, ← c98192] at e_1603_6
  have e_1603_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_7
  simp only [← c98193, ← c98194, k2, ← c98195] at e_1603_7
  have hr := e_1603_7
  simp only [e_1603_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1603 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15719)) := by
  have c98196 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98197 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98198 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98199 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98200 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98201 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1603_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_8
  simp only [← c98196, ← c98197, ← c98198] at e_1603_8
  have e_1603_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_9
  simp only [← c98199, ← c98200, k2, ← c98201] at e_1603_9
  have hr := e_1603_9
  simp only [e_1603_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1603 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15720)) := by
  have c98202 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98203 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98204 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98205 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98206 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98207 := (publicBatchWrapper89_copies3068 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1603_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_10
  simp only [← c98202, ← c98203, ← c98204] at e_1603_10
  have e_1603_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_11
  simp only [← c98205, ← c98206, k2, ← c98207] at e_1603_11
  have hr := e_1603_11
  simp only [e_1603_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1603 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15721)) := by
  have c98208 := (publicBatchWrapper89_copies3069 a h).1
  have c98209 := (publicBatchWrapper89_copies3069 a h).2.1
  have c98210 := (publicBatchWrapper89_copies3069 a h).2.2.1
  have c98211 := (publicBatchWrapper89_copies3069 a h).2.2.2.1
  have c98212 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.1
  have c98213 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1603_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_12
  simp only [← c98208, ← c98209, ← c98210] at e_1603_12
  have e_1603_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_13
  simp only [← c98211, ← c98212, k2, ← c98213] at e_1603_13
  have hr := e_1603_13
  simp only [e_1603_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1603 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15722)) := by
  have c98214 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.1
  have c98215 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.1
  have c98216 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.1
  have c98217 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.1
  have c98218 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98219 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1603_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_14
  simp only [← c98214, ← c98215, ← c98216] at e_1603_14
  have e_1603_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_15
  simp only [← c98217, ← c98218, k2, ← c98219] at e_1603_15
  have hr := e_1603_15
  simp only [e_1603_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1603 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15723)) := by
  have c98220 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98221 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98222 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98223 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98224 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98225 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1603_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_16
  simp only [← c98220, ← c98221, ← c98222] at e_1603_16
  have e_1603_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_17
  simp only [← c98223, ← c98224, k2, ← c98225] at e_1603_17
  have hr := e_1603_17
  simp only [e_1603_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1603 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15724)) := by
  have c98226 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98227 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98228 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98229 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98230 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98231 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1603_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_18
  simp only [← c98226, ← c98227, ← c98228] at e_1603_18
  have e_1603_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1603 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1603_19
  simp only [← c98229, ← c98230, k2, ← c98231] at e_1603_19
  have hr := e_1603_19
  simp only [e_1603_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1604 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15725)) := by
  have c98232 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98233 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98234 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98235 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98236 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98237 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1604_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_0
  simp only [← c98232, ← c98233, ← c98234] at e_1604_0
  have e_1604_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_1
  simp only [← c98235, ← c98236, k2, ← c98237] at e_1604_1
  have hr := e_1604_1
  simp only [e_1604_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1604 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15726)) := by
  have c98238 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98239 := (publicBatchWrapper89_copies3069 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98240 := (publicBatchWrapper89_copies3070 a h).1
  have c98241 := (publicBatchWrapper89_copies3070 a h).2.1
  have c98242 := (publicBatchWrapper89_copies3070 a h).2.2.1
  have c98243 := (publicBatchWrapper89_copies3070 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1604_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_2
  simp only [← c98238, ← c98239, ← c98240] at e_1604_2
  have e_1604_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_3
  simp only [← c98241, ← c98242, k2, ← c98243] at e_1604_3
  have hr := e_1604_3
  simp only [e_1604_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1604 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15727)) := by
  have c98244 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.1
  have c98245 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.1
  have c98246 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.1
  have c98247 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.1
  have c98248 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.1
  have c98249 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1604_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_4
  simp only [← c98244, ← c98245, ← c98246] at e_1604_4
  have e_1604_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_5
  simp only [← c98247, ← c98248, k2, ← c98249] at e_1604_5
  have hr := e_1604_5
  simp only [e_1604_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1604 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15728)) := by
  have c98250 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98251 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98252 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98253 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98254 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98255 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1604_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_6
  simp only [← c98250, ← c98251, ← c98252] at e_1604_6
  have e_1604_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_7
  simp only [← c98253, ← c98254, k2, ← c98255] at e_1604_7
  have hr := e_1604_7
  simp only [e_1604_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1604 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15729)) := by
  have c98256 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98257 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98258 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98259 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98260 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98261 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1604_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_8
  simp only [← c98256, ← c98257, ← c98258] at e_1604_8
  have e_1604_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_9
  simp only [← c98259, ← c98260, k2, ← c98261] at e_1604_9
  have hr := e_1604_9
  simp only [e_1604_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1604 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15730)) := by
  have c98262 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98263 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98264 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98265 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98266 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98267 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1604_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_10
  simp only [← c98262, ← c98263, ← c98264] at e_1604_10
  have e_1604_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_11
  simp only [← c98265, ← c98266, k2, ← c98267] at e_1604_11
  have hr := e_1604_11
  simp only [e_1604_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1604 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15731)) := by
  have c98268 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98269 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98270 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98271 := (publicBatchWrapper89_copies3070 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98272 := (publicBatchWrapper89_copies3071 a h).1
  have c98273 := (publicBatchWrapper89_copies3071 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1604_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_12
  simp only [← c98268, ← c98269, ← c98270] at e_1604_12
  have e_1604_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_13
  simp only [← c98271, ← c98272, k2, ← c98273] at e_1604_13
  have hr := e_1604_13
  simp only [e_1604_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1604 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15732)) := by
  have c98274 := (publicBatchWrapper89_copies3071 a h).2.2.1
  have c98275 := (publicBatchWrapper89_copies3071 a h).2.2.2.1
  have c98276 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.1
  have c98277 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.1
  have c98278 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.1
  have c98279 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1604_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_14
  simp only [← c98274, ← c98275, ← c98276] at e_1604_14
  have e_1604_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_15
  simp only [← c98277, ← c98278, k2, ← c98279] at e_1604_15
  have hr := e_1604_15
  simp only [e_1604_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1604 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15733)) := by
  have c98280 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.1
  have c98281 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.1
  have c98282 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98283 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98284 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98285 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1604_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_16
  simp only [← c98280, ← c98281, ← c98282] at e_1604_16
  have e_1604_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_17
  simp only [← c98283, ← c98284, k2, ← c98285] at e_1604_17
  have hr := e_1604_17
  simp only [e_1604_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1604 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15734)) := by
  have c98286 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98287 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98288 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98289 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98290 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98291 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1604_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_18
  simp only [← c98286, ← c98287, ← c98288] at e_1604_18
  have e_1604_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1604 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1604_19
  simp only [← c98289, ← c98290, k2, ← c98291] at e_1604_19
  have hr := e_1604_19
  simp only [e_1604_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1605 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15735)) := by
  have c98292 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98293 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98294 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98295 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98296 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98297 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1605_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_0
  simp only [← c98292, ← c98293, ← c98294] at e_1605_0
  have e_1605_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_1
  simp only [← c98295, ← c98296, k2, ← c98297] at e_1605_1
  have hr := e_1605_1
  simp only [e_1605_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1605 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15736)) := by
  have c98298 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98299 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98300 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98301 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98302 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98303 := (publicBatchWrapper89_copies3071 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1605_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_2
  simp only [← c98298, ← c98299, ← c98300] at e_1605_2
  have e_1605_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_3
  simp only [← c98301, ← c98302, k2, ← c98303] at e_1605_3
  have hr := e_1605_3
  simp only [e_1605_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1605 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15737)) := by
  have c98304 := (publicBatchWrapper89_copies3072 a h).1
  have c98305 := (publicBatchWrapper89_copies3072 a h).2.1
  have c98306 := (publicBatchWrapper89_copies3072 a h).2.2.1
  have c98307 := (publicBatchWrapper89_copies3072 a h).2.2.2.1
  have c98308 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.1
  have c98309 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1605_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_4
  simp only [← c98304, ← c98305, ← c98306] at e_1605_4
  have e_1605_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_5
  simp only [← c98307, ← c98308, k2, ← c98309] at e_1605_5
  have hr := e_1605_5
  simp only [e_1605_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1605 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15738)) := by
  have c98310 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.1
  have c98311 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.1
  have c98312 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.1
  have c98313 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.1
  have c98314 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98315 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1605_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_6
  simp only [← c98310, ← c98311, ← c98312] at e_1605_6
  have e_1605_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_7
  simp only [← c98313, ← c98314, k2, ← c98315] at e_1605_7
  have hr := e_1605_7
  simp only [e_1605_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1605 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15739)) := by
  have c98316 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98317 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98318 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98319 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98320 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98321 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1605_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_8
  simp only [← c98316, ← c98317, ← c98318] at e_1605_8
  have e_1605_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_9
  simp only [← c98319, ← c98320, k2, ← c98321] at e_1605_9
  have hr := e_1605_9
  simp only [e_1605_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1605 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15740)) := by
  have c98322 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98323 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98324 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98325 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98326 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98327 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1605_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_10
  simp only [← c98322, ← c98323, ← c98324] at e_1605_10
  have e_1605_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_11
  simp only [← c98325, ← c98326, k2, ← c98327] at e_1605_11
  have hr := e_1605_11
  simp only [e_1605_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1605 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15741)) := by
  have c98328 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98329 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98330 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98331 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98332 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98333 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1605_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_12
  simp only [← c98328, ← c98329, ← c98330] at e_1605_12
  have e_1605_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_13
  simp only [← c98331, ← c98332, k2, ← c98333] at e_1605_13
  have hr := e_1605_13
  simp only [e_1605_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1605 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15742)) := by
  have c98334 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98335 := (publicBatchWrapper89_copies3072 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98336 := (publicBatchWrapper89_copies3073 a h).1
  have c98337 := (publicBatchWrapper89_copies3073 a h).2.1
  have c98338 := (publicBatchWrapper89_copies3073 a h).2.2.1
  have c98339 := (publicBatchWrapper89_copies3073 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1605_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_14
  simp only [← c98334, ← c98335, ← c98336] at e_1605_14
  have e_1605_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_15
  simp only [← c98337, ← c98338, k2, ← c98339] at e_1605_15
  have hr := e_1605_15
  simp only [e_1605_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1605 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15743)) := by
  have c98340 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.1
  have c98341 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.1
  have c98342 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.1
  have c98343 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.1
  have c98344 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.1
  have c98345 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1605_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_16
  simp only [← c98340, ← c98341, ← c98342] at e_1605_16
  have e_1605_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_17
  simp only [← c98343, ← c98344, k2, ← c98345] at e_1605_17
  have hr := e_1605_17
  simp only [e_1605_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1605 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15744)) := by
  have c98346 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98347 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98348 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98349 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98350 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98351 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1605_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_18
  simp only [← c98346, ← c98347, ← c98348] at e_1605_18
  have e_1605_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1605 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1605_19
  simp only [← c98349, ← c98350, k2, ← c98351] at e_1605_19
  have hr := e_1605_19
  simp only [e_1605_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1606 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15745)) := by
  have c98352 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98353 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98354 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98355 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98356 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98357 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1606_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_0
  simp only [← c98352, ← c98353, ← c98354] at e_1606_0
  have e_1606_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_1
  simp only [← c98355, ← c98356, k2, ← c98357] at e_1606_1
  have hr := e_1606_1
  simp only [e_1606_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1606 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15746)) := by
  have c98358 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98359 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98360 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98361 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98362 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98363 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1606_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_2
  simp only [← c98358, ← c98359, ← c98360] at e_1606_2
  have e_1606_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_3
  simp only [← c98361, ← c98362, k2, ← c98363] at e_1606_3
  have hr := e_1606_3
  simp only [e_1606_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1606 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15747)) := by
  have c98364 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98365 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98366 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98367 := (publicBatchWrapper89_copies3073 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98368 := (publicBatchWrapper89_copies3074 a h).1
  have c98369 := (publicBatchWrapper89_copies3074 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1606_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_4
  simp only [← c98364, ← c98365, ← c98366] at e_1606_4
  have e_1606_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_5
  simp only [← c98367, ← c98368, k2, ← c98369] at e_1606_5
  have hr := e_1606_5
  simp only [e_1606_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1606 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15748)) := by
  have c98370 := (publicBatchWrapper89_copies3074 a h).2.2.1
  have c98371 := (publicBatchWrapper89_copies3074 a h).2.2.2.1
  have c98372 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.1
  have c98373 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.1
  have c98374 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.1
  have c98375 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1606_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_6
  simp only [← c98370, ← c98371, ← c98372] at e_1606_6
  have e_1606_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_7
  simp only [← c98373, ← c98374, k2, ← c98375] at e_1606_7
  have hr := e_1606_7
  simp only [e_1606_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1606 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15749)) := by
  have c98376 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.1
  have c98377 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.1
  have c98378 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98379 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98380 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98381 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1606_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_8
  simp only [← c98376, ← c98377, ← c98378] at e_1606_8
  have e_1606_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_9
  simp only [← c98379, ← c98380, k2, ← c98381] at e_1606_9
  have hr := e_1606_9
  simp only [e_1606_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1606 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15750)) := by
  have c98382 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98383 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98384 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98385 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98386 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98387 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1606_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_10
  simp only [← c98382, ← c98383, ← c98384] at e_1606_10
  have e_1606_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_11
  simp only [← c98385, ← c98386, k2, ← c98387] at e_1606_11
  have hr := e_1606_11
  simp only [e_1606_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1606 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15751)) := by
  have c98388 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98389 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98390 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98391 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98392 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98393 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1606_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_12
  simp only [← c98388, ← c98389, ← c98390] at e_1606_12
  have e_1606_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_13
  simp only [← c98391, ← c98392, k2, ← c98393] at e_1606_13
  have hr := e_1606_13
  simp only [e_1606_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1606 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15752)) := by
  have c98394 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98395 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98396 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98397 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98398 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98399 := (publicBatchWrapper89_copies3074 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1606_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_14
  simp only [← c98394, ← c98395, ← c98396] at e_1606_14
  have e_1606_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_15
  simp only [← c98397, ← c98398, k2, ← c98399] at e_1606_15
  have hr := e_1606_15
  simp only [e_1606_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1606 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15753)) := by
  have c98400 := (publicBatchWrapper89_copies3075 a h).1
  have c98401 := (publicBatchWrapper89_copies3075 a h).2.1
  have c98402 := (publicBatchWrapper89_copies3075 a h).2.2.1
  have c98403 := (publicBatchWrapper89_copies3075 a h).2.2.2.1
  have c98404 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.1
  have c98405 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1606_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_16
  simp only [← c98400, ← c98401, ← c98402] at e_1606_16
  have e_1606_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_17
  simp only [← c98403, ← c98404, k2, ← c98405] at e_1606_17
  have hr := e_1606_17
  simp only [e_1606_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1606 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15754)) := by
  have c98406 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.1
  have c98407 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.1
  have c98408 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.1
  have c98409 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.1
  have c98410 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98411 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1606_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_18
  simp only [← c98406, ← c98407, ← c98408] at e_1606_18
  have e_1606_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1606 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1606_19
  simp only [← c98409, ← c98410, k2, ← c98411] at e_1606_19
  have hr := e_1606_19
  simp only [e_1606_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1607 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15755)) := by
  have c98412 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98413 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98414 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98415 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98416 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98417 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1607_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_0
  simp only [← c98412, ← c98413, ← c98414] at e_1607_0
  have e_1607_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_1
  simp only [← c98415, ← c98416, k2, ← c98417] at e_1607_1
  have hr := e_1607_1
  simp only [e_1607_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1607 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15756)) := by
  have c98418 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98419 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98420 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98421 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98422 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98423 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1607_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_2
  simp only [← c98418, ← c98419, ← c98420] at e_1607_2
  have e_1607_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_3
  simp only [← c98421, ← c98422, k2, ← c98423] at e_1607_3
  have hr := e_1607_3
  simp only [e_1607_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1607 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15757)) := by
  have c98424 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98425 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98426 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98427 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98428 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98429 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1607_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_4
  simp only [← c98424, ← c98425, ← c98426] at e_1607_4
  have e_1607_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_5
  simp only [← c98427, ← c98428, k2, ← c98429] at e_1607_5
  have hr := e_1607_5
  simp only [e_1607_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1607 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15758)) := by
  have c98430 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98431 := (publicBatchWrapper89_copies3075 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98432 := (publicBatchWrapper89_copies3076 a h).1
  have c98433 := (publicBatchWrapper89_copies3076 a h).2.1
  have c98434 := (publicBatchWrapper89_copies3076 a h).2.2.1
  have c98435 := (publicBatchWrapper89_copies3076 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1607_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_6
  simp only [← c98430, ← c98431, ← c98432] at e_1607_6
  have e_1607_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_7
  simp only [← c98433, ← c98434, k2, ← c98435] at e_1607_7
  have hr := e_1607_7
  simp only [e_1607_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1607 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15759)) := by
  have c98436 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.1
  have c98437 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.1
  have c98438 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.1
  have c98439 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.1
  have c98440 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.1
  have c98441 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1607_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_8
  simp only [← c98436, ← c98437, ← c98438] at e_1607_8
  have e_1607_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_9
  simp only [← c98439, ← c98440, k2, ← c98441] at e_1607_9
  have hr := e_1607_9
  simp only [e_1607_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1607 47) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15760)) := by
  have c98442 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98443 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98444 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98445 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98446 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98447 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1607_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_10
  simp only [← c98442, ← c98443, ← c98444] at e_1607_10
  have e_1607_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_11
  simp only [← c98445, ← c98446, k2, ← c98447] at e_1607_11
  have hr := e_1607_11
  simp only [e_1607_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1607 55) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15761)) := by
  have c98448 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98449 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98450 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98451 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98452 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98453 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1607_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_12
  simp only [← c98448, ← c98449, ← c98450] at e_1607_12
  have e_1607_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_13
  simp only [← c98451, ← c98452, k2, ← c98453] at e_1607_13
  have hr := e_1607_13
  simp only [e_1607_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1607 63) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15762)) := by
  have c98454 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98455 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98456 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98457 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98458 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98459 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1607_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_14
  simp only [← c98454, ← c98455, ← c98456] at e_1607_14
  have e_1607_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_15
  simp only [← c98457, ← c98458, k2, ← c98459] at e_1607_15
  have hr := e_1607_15
  simp only [e_1607_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1607 71) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15763)) := by
  have c98460 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98461 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98462 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98463 := (publicBatchWrapper89_copies3076 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98464 := (publicBatchWrapper89_copies3077 a h).1
  have c98465 := (publicBatchWrapper89_copies3077 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1607_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_16
  simp only [← c98460, ← c98461, ← c98462] at e_1607_16
  have e_1607_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_17
  simp only [← c98463, ← c98464, k2, ← c98465] at e_1607_17
  have hr := e_1607_17
  simp only [e_1607_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1607 79) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15764)) := by
  have c98466 := (publicBatchWrapper89_copies3077 a h).2.2.1
  have c98467 := (publicBatchWrapper89_copies3077 a h).2.2.2.1
  have c98468 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.1
  have c98469 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.1
  have c98470 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.1
  have c98471 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1607_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_18
  simp only [← c98466, ← c98467, ← c98468] at e_1607_18
  have e_1607_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1607 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1607_19
  simp only [← c98469, ← c98470, k2, ← c98471] at e_1607_19
  have hr := e_1607_19
  simp only [e_1607_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1608 7) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15765)) := by
  have c98472 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.1
  have c98473 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.1
  have c98474 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98475 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98476 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98477 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1608_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_0
  simp only [← c98472, ← c98473, ← c98474] at e_1608_0
  have e_1608_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_1
  simp only [← c98475, ← c98476, k2, ← c98477] at e_1608_1
  have hr := e_1608_1
  simp only [e_1608_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1608 15) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15766)) := by
  have c98478 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98479 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98480 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98481 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98482 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98483 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1608_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_2
  simp only [← c98478, ← c98479, ← c98480] at e_1608_2
  have e_1608_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_3
  simp only [← c98481, ← c98482, k2, ← c98483] at e_1608_3
  have hr := e_1608_3
  simp only [e_1608_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1608 23) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15767)) := by
  have c98484 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98485 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98486 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98487 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98488 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98489 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1608_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_4
  simp only [← c98484, ← c98485, ← c98486] at e_1608_4
  have e_1608_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_5
  simp only [← c98487, ← c98488, k2, ← c98489] at e_1608_5
  have hr := e_1608_5
  simp only [e_1608_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1608 31) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15768)) := by
  have c98490 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98491 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98492 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98493 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98494 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98495 := (publicBatchWrapper89_copies3077 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1608_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_6
  simp only [← c98490, ← c98491, ← c98492] at e_1608_6
  have e_1608_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_7
  simp only [← c98493, ← c98494, k2, ← c98495] at e_1608_7
  have hr := e_1608_7
  simp only [e_1608_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1608 39) = bselect (a (.wire 76 51)) (a (.virt 200)) (a (.virt 15769)) := by
  have c98496 := (publicBatchWrapper89_copies3078 a h).1
  have c98497 := (publicBatchWrapper89_copies3078 a h).2.1
  have c98498 := (publicBatchWrapper89_copies3078 a h).2.2.1
  have c98499 := (publicBatchWrapper89_copies3078 a h).2.2.2.1
  have c98500 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.1
  have c98501 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1608_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_8
  simp only [← c98496, ← c98497, ← c98498] at e_1608_8
  have e_1608_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_9
  simp only [← c98499, ← c98500, k2, ← c98501] at e_1608_9
  have hr := e_1608_9
  simp only [e_1608_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1608 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15810)) := by
  have c98502 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.1
  have c98503 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.1
  have c98504 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.1
  have c98505 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.1
  have c98506 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98507 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1608_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_10
  simp only [← c98502, ← c98503, ← c98504] at e_1608_10
  have e_1608_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_11
  simp only [← c98505, ← c98506, k2, ← c98507] at e_1608_11
  have hr := e_1608_11
  simp only [e_1608_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1608 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15811)) := by
  have c98508 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98509 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98510 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98511 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98512 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98513 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1608_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_12
  simp only [← c98508, ← c98509, ← c98510] at e_1608_12
  have e_1608_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_13
  simp only [← c98511, ← c98512, k2, ← c98513] at e_1608_13
  have hr := e_1608_13
  simp only [e_1608_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1608 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15812)) := by
  have c98514 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98515 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98516 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98517 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98518 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98519 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1608_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_14
  simp only [← c98514, ← c98515, ← c98516] at e_1608_14
  have e_1608_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_15
  simp only [← c98517, ← c98518, k2, ← c98519] at e_1608_15
  have hr := e_1608_15
  simp only [e_1608_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1608 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15813)) := by
  have c98520 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98521 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98522 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98523 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98524 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98525 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1608_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_16
  simp only [← c98520, ← c98521, ← c98522] at e_1608_16
  have e_1608_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_17
  simp only [← c98523, ← c98524, k2, ← c98525] at e_1608_17
  have hr := e_1608_17
  simp only [e_1608_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1608 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15814)) := by
  have c98526 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98527 := (publicBatchWrapper89_copies3078 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98528 := (publicBatchWrapper89_copies3079 a h).1
  have c98529 := (publicBatchWrapper89_copies3079 a h).2.1
  have c98530 := (publicBatchWrapper89_copies3079 a h).2.2.1
  have c98531 := (publicBatchWrapper89_copies3079 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1608_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_18
  simp only [← c98526, ← c98527, ← c98528] at e_1608_18
  have e_1608_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1608 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1608_19
  simp only [← c98529, ← c98530, k2, ← c98531] at e_1608_19
  have hr := e_1608_19
  simp only [e_1608_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1609 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15815)) := by
  have c98532 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.1
  have c98533 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.1
  have c98534 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.1
  have c98535 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.1
  have c98536 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.1
  have c98537 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1609_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_0
  simp only [← c98532, ← c98533, ← c98534] at e_1609_0
  have e_1609_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_1
  simp only [← c98535, ← c98536, k2, ← c98537] at e_1609_1
  have hr := e_1609_1
  simp only [e_1609_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1609 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15816)) := by
  have c98538 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98539 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98540 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98541 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98542 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98543 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1609_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_2
  simp only [← c98538, ← c98539, ← c98540] at e_1609_2
  have e_1609_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_3
  simp only [← c98541, ← c98542, k2, ← c98543] at e_1609_3
  have hr := e_1609_3
  simp only [e_1609_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1609 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15817)) := by
  have c98544 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98545 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98546 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98547 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98548 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98549 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1609_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_4
  simp only [← c98544, ← c98545, ← c98546] at e_1609_4
  have e_1609_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_5
  simp only [← c98547, ← c98548, k2, ← c98549] at e_1609_5
  have hr := e_1609_5
  simp only [e_1609_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1609 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15818)) := by
  have c98550 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98551 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98552 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98553 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98554 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98555 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1609_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_6
  simp only [← c98550, ← c98551, ← c98552] at e_1609_6
  have e_1609_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_7
  simp only [← c98553, ← c98554, k2, ← c98555] at e_1609_7
  have hr := e_1609_7
  simp only [e_1609_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1609 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15819)) := by
  have c98556 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98557 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98558 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98559 := (publicBatchWrapper89_copies3079 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98560 := (publicBatchWrapper89_copies3080 a h).1
  have c98561 := (publicBatchWrapper89_copies3080 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1609_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_8
  simp only [← c98556, ← c98557, ← c98558] at e_1609_8
  have e_1609_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_9
  simp only [← c98559, ← c98560, k2, ← c98561] at e_1609_9
  have hr := e_1609_9
  simp only [e_1609_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1609 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15820)) := by
  have c98562 := (publicBatchWrapper89_copies3080 a h).2.2.1
  have c98563 := (publicBatchWrapper89_copies3080 a h).2.2.2.1
  have c98564 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.1
  have c98565 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.1
  have c98566 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.1
  have c98567 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1609_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_10
  simp only [← c98562, ← c98563, ← c98564] at e_1609_10
  have e_1609_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_11
  simp only [← c98565, ← c98566, k2, ← c98567] at e_1609_11
  have hr := e_1609_11
  simp only [e_1609_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1609 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15821)) := by
  have c98568 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.1
  have c98569 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.1
  have c98570 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98571 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98572 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98573 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1609_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_12
  simp only [← c98568, ← c98569, ← c98570] at e_1609_12
  have e_1609_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_13
  simp only [← c98571, ← c98572, k2, ← c98573] at e_1609_13
  have hr := e_1609_13
  simp only [e_1609_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1609 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15822)) := by
  have c98574 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98575 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98576 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98577 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98578 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98579 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1609_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_14
  simp only [← c98574, ← c98575, ← c98576] at e_1609_14
  have e_1609_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_15
  simp only [← c98577, ← c98578, k2, ← c98579] at e_1609_15
  have hr := e_1609_15
  simp only [e_1609_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1609 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15823)) := by
  have c98580 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98581 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98582 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98583 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98584 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98585 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1609_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_16
  simp only [← c98580, ← c98581, ← c98582] at e_1609_16
  have e_1609_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_17
  simp only [← c98583, ← c98584, k2, ← c98585] at e_1609_17
  have hr := e_1609_17
  simp only [e_1609_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1609 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15824)) := by
  have c98586 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98587 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98588 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98589 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98590 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98591 := (publicBatchWrapper89_copies3080 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1609_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_18
  simp only [← c98586, ← c98587, ← c98588] at e_1609_18
  have e_1609_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1609 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1609_19
  simp only [← c98589, ← c98590, k2, ← c98591] at e_1609_19
  have hr := e_1609_19
  simp only [e_1609_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1610 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15825)) := by
  have c98592 := (publicBatchWrapper89_copies3081 a h).1
  have c98593 := (publicBatchWrapper89_copies3081 a h).2.1
  have c98594 := (publicBatchWrapper89_copies3081 a h).2.2.1
  have c98595 := (publicBatchWrapper89_copies3081 a h).2.2.2.1
  have c98596 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.1
  have c98597 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1610_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_0
  simp only [← c98592, ← c98593, ← c98594] at e_1610_0
  have e_1610_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_1
  simp only [← c98595, ← c98596, k2, ← c98597] at e_1610_1
  have hr := e_1610_1
  simp only [e_1610_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1610 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15826)) := by
  have c98598 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.1
  have c98599 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.1
  have c98600 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.1
  have c98601 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.1
  have c98602 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98603 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1610_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_2
  simp only [← c98598, ← c98599, ← c98600] at e_1610_2
  have e_1610_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_3
  simp only [← c98601, ← c98602, k2, ← c98603] at e_1610_3
  have hr := e_1610_3
  simp only [e_1610_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1610 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15827)) := by
  have c98604 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98605 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98606 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98607 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98608 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98609 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1610_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_4
  simp only [← c98604, ← c98605, ← c98606] at e_1610_4
  have e_1610_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_5
  simp only [← c98607, ← c98608, k2, ← c98609] at e_1610_5
  have hr := e_1610_5
  simp only [e_1610_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1610 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15828)) := by
  have c98610 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98611 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98612 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98613 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98614 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98615 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1610_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_6
  simp only [← c98610, ← c98611, ← c98612] at e_1610_6
  have e_1610_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_7
  simp only [← c98613, ← c98614, k2, ← c98615] at e_1610_7
  have hr := e_1610_7
  simp only [e_1610_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1610 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15829)) := by
  have c98616 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98617 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98618 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98619 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98620 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98621 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1610_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_8
  simp only [← c98616, ← c98617, ← c98618] at e_1610_8
  have e_1610_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_9
  simp only [← c98619, ← c98620, k2, ← c98621] at e_1610_9
  have hr := e_1610_9
  simp only [e_1610_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1610 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15830)) := by
  have c98622 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98623 := (publicBatchWrapper89_copies3081 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98624 := (publicBatchWrapper89_copies3082 a h).1
  have c98625 := (publicBatchWrapper89_copies3082 a h).2.1
  have c98626 := (publicBatchWrapper89_copies3082 a h).2.2.1
  have c98627 := (publicBatchWrapper89_copies3082 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1610_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_10
  simp only [← c98622, ← c98623, ← c98624] at e_1610_10
  have e_1610_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_11
  simp only [← c98625, ← c98626, k2, ← c98627] at e_1610_11
  have hr := e_1610_11
  simp only [e_1610_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1610 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15831)) := by
  have c98628 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.1
  have c98629 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.1
  have c98630 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.1
  have c98631 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.1
  have c98632 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.1
  have c98633 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1610_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_12
  simp only [← c98628, ← c98629, ← c98630] at e_1610_12
  have e_1610_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_13
  simp only [← c98631, ← c98632, k2, ← c98633] at e_1610_13
  have hr := e_1610_13
  simp only [e_1610_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1610 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15832)) := by
  have c98634 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98635 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98636 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98637 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98638 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98639 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1610_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_14
  simp only [← c98634, ← c98635, ← c98636] at e_1610_14
  have e_1610_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_15
  simp only [← c98637, ← c98638, k2, ← c98639] at e_1610_15
  have hr := e_1610_15
  simp only [e_1610_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1610 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15833)) := by
  have c98640 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98641 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98642 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98643 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98644 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98645 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1610_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_16
  simp only [← c98640, ← c98641, ← c98642] at e_1610_16
  have e_1610_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_17
  simp only [← c98643, ← c98644, k2, ← c98645] at e_1610_17
  have hr := e_1610_17
  simp only [e_1610_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1610 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15834)) := by
  have c98646 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98647 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98648 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98649 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98650 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98651 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1610_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_18
  simp only [← c98646, ← c98647, ← c98648] at e_1610_18
  have e_1610_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1610 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1610_19
  simp only [← c98649, ← c98650, k2, ← c98651] at e_1610_19
  have hr := e_1610_19
  simp only [e_1610_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1611 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15835)) := by
  have c98652 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98653 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98654 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98655 := (publicBatchWrapper89_copies3082 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98656 := (publicBatchWrapper89_copies3083 a h).1
  have c98657 := (publicBatchWrapper89_copies3083 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1611_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_0
  simp only [← c98652, ← c98653, ← c98654] at e_1611_0
  have e_1611_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_1
  simp only [← c98655, ← c98656, k2, ← c98657] at e_1611_1
  have hr := e_1611_1
  simp only [e_1611_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1611 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15836)) := by
  have c98658 := (publicBatchWrapper89_copies3083 a h).2.2.1
  have c98659 := (publicBatchWrapper89_copies3083 a h).2.2.2.1
  have c98660 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.1
  have c98661 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.1
  have c98662 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.1
  have c98663 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1611_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_2
  simp only [← c98658, ← c98659, ← c98660] at e_1611_2
  have e_1611_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_3
  simp only [← c98661, ← c98662, k2, ← c98663] at e_1611_3
  have hr := e_1611_3
  simp only [e_1611_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1611 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15837)) := by
  have c98664 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.1
  have c98665 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.1
  have c98666 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98667 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98668 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98669 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1611_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_4
  simp only [← c98664, ← c98665, ← c98666] at e_1611_4
  have e_1611_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_5
  simp only [← c98667, ← c98668, k2, ← c98669] at e_1611_5
  have hr := e_1611_5
  simp only [e_1611_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1611 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15838)) := by
  have c98670 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98671 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98672 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98673 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98674 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98675 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1611_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_6
  simp only [← c98670, ← c98671, ← c98672] at e_1611_6
  have e_1611_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_7
  simp only [← c98673, ← c98674, k2, ← c98675] at e_1611_7
  have hr := e_1611_7
  simp only [e_1611_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1611 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15839)) := by
  have c98676 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98677 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98678 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98679 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98680 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98681 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1611_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_8
  simp only [← c98676, ← c98677, ← c98678] at e_1611_8
  have e_1611_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_9
  simp only [← c98679, ← c98680, k2, ← c98681] at e_1611_9
  have hr := e_1611_9
  simp only [e_1611_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1611 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15840)) := by
  have c98682 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98683 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98684 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98685 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98686 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98687 := (publicBatchWrapper89_copies3083 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1611_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_10
  simp only [← c98682, ← c98683, ← c98684] at e_1611_10
  have e_1611_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_11
  simp only [← c98685, ← c98686, k2, ← c98687] at e_1611_11
  have hr := e_1611_11
  simp only [e_1611_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1611 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15841)) := by
  have c98688 := (publicBatchWrapper89_copies3084 a h).1
  have c98689 := (publicBatchWrapper89_copies3084 a h).2.1
  have c98690 := (publicBatchWrapper89_copies3084 a h).2.2.1
  have c98691 := (publicBatchWrapper89_copies3084 a h).2.2.2.1
  have c98692 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.1
  have c98693 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1611_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_12
  simp only [← c98688, ← c98689, ← c98690] at e_1611_12
  have e_1611_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_13
  simp only [← c98691, ← c98692, k2, ← c98693] at e_1611_13
  have hr := e_1611_13
  simp only [e_1611_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1611 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15842)) := by
  have c98694 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.1
  have c98695 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.1
  have c98696 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.1
  have c98697 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.1
  have c98698 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98699 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1611_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_14
  simp only [← c98694, ← c98695, ← c98696] at e_1611_14
  have e_1611_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_15
  simp only [← c98697, ← c98698, k2, ← c98699] at e_1611_15
  have hr := e_1611_15
  simp only [e_1611_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1611 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15843)) := by
  have c98700 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98701 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98702 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98703 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98704 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98705 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1611_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_16
  simp only [← c98700, ← c98701, ← c98702] at e_1611_16
  have e_1611_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_17
  simp only [← c98703, ← c98704, k2, ← c98705] at e_1611_17
  have hr := e_1611_17
  simp only [e_1611_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1611 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15844)) := by
  have c98706 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98707 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98708 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98709 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98710 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98711 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1611_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_18
  simp only [← c98706, ← c98707, ← c98708] at e_1611_18
  have e_1611_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1611 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1611_19
  simp only [← c98709, ← c98710, k2, ← c98711] at e_1611_19
  have hr := e_1611_19
  simp only [e_1611_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1612 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15845)) := by
  have c98712 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98713 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98714 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98715 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98716 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98717 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1612_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_0
  simp only [← c98712, ← c98713, ← c98714] at e_1612_0
  have e_1612_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_1
  simp only [← c98715, ← c98716, k2, ← c98717] at e_1612_1
  have hr := e_1612_1
  simp only [e_1612_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1612 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15846)) := by
  have c98718 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98719 := (publicBatchWrapper89_copies3084 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98720 := (publicBatchWrapper89_copies3085 a h).1
  have c98721 := (publicBatchWrapper89_copies3085 a h).2.1
  have c98722 := (publicBatchWrapper89_copies3085 a h).2.2.1
  have c98723 := (publicBatchWrapper89_copies3085 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1612_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_2
  simp only [← c98718, ← c98719, ← c98720] at e_1612_2
  have e_1612_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_3
  simp only [← c98721, ← c98722, k2, ← c98723] at e_1612_3
  have hr := e_1612_3
  simp only [e_1612_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1612 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15847)) := by
  have c98724 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.1
  have c98725 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.1
  have c98726 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.1
  have c98727 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.1
  have c98728 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.1
  have c98729 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1612_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_4
  simp only [← c98724, ← c98725, ← c98726] at e_1612_4
  have e_1612_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_5
  simp only [← c98727, ← c98728, k2, ← c98729] at e_1612_5
  have hr := e_1612_5
  simp only [e_1612_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1612 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15848)) := by
  have c98730 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98731 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98732 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98733 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98734 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98735 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1612_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_6
  simp only [← c98730, ← c98731, ← c98732] at e_1612_6
  have e_1612_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_7
  simp only [← c98733, ← c98734, k2, ← c98735] at e_1612_7
  have hr := e_1612_7
  simp only [e_1612_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1612 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15849)) := by
  have c98736 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98737 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98738 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98739 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98740 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98741 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1612_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_8
  simp only [← c98736, ← c98737, ← c98738] at e_1612_8
  have e_1612_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_9
  simp only [← c98739, ← c98740, k2, ← c98741] at e_1612_9
  have hr := e_1612_9
  simp only [e_1612_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1612 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15850)) := by
  have c98742 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98743 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98744 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98745 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98746 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98747 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1612_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_10
  simp only [← c98742, ← c98743, ← c98744] at e_1612_10
  have e_1612_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_11
  simp only [← c98745, ← c98746, k2, ← c98747] at e_1612_11
  have hr := e_1612_11
  simp only [e_1612_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1612 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15851)) := by
  have c98748 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98749 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98750 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98751 := (publicBatchWrapper89_copies3085 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98752 := (publicBatchWrapper89_copies3086 a h).1
  have c98753 := (publicBatchWrapper89_copies3086 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1612_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_12
  simp only [← c98748, ← c98749, ← c98750] at e_1612_12
  have e_1612_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_13
  simp only [← c98751, ← c98752, k2, ← c98753] at e_1612_13
  have hr := e_1612_13
  simp only [e_1612_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1612 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15852)) := by
  have c98754 := (publicBatchWrapper89_copies3086 a h).2.2.1
  have c98755 := (publicBatchWrapper89_copies3086 a h).2.2.2.1
  have c98756 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.1
  have c98757 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.1
  have c98758 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.1
  have c98759 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1612_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_14
  simp only [← c98754, ← c98755, ← c98756] at e_1612_14
  have e_1612_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_15
  simp only [← c98757, ← c98758, k2, ← c98759] at e_1612_15
  have hr := e_1612_15
  simp only [e_1612_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1612 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15853)) := by
  have c98760 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.1
  have c98761 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.1
  have c98762 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98763 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98764 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98765 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1612_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_16
  simp only [← c98760, ← c98761, ← c98762] at e_1612_16
  have e_1612_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_17
  simp only [← c98763, ← c98764, k2, ← c98765] at e_1612_17
  have hr := e_1612_17
  simp only [e_1612_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1612 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15854)) := by
  have c98766 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98767 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98768 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98769 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98770 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98771 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1612_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_18
  simp only [← c98766, ← c98767, ← c98768] at e_1612_18
  have e_1612_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1612 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1612_19
  simp only [← c98769, ← c98770, k2, ← c98771] at e_1612_19
  have hr := e_1612_19
  simp only [e_1612_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1613 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15855)) := by
  have c98772 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98773 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98774 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98775 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98776 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98777 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1613_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_0
  simp only [← c98772, ← c98773, ← c98774] at e_1613_0
  have e_1613_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_1
  simp only [← c98775, ← c98776, k2, ← c98777] at e_1613_1
  have hr := e_1613_1
  simp only [e_1613_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1613 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15856)) := by
  have c98778 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98779 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98780 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98781 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98782 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98783 := (publicBatchWrapper89_copies3086 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1613_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_2
  simp only [← c98778, ← c98779, ← c98780] at e_1613_2
  have e_1613_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_3
  simp only [← c98781, ← c98782, k2, ← c98783] at e_1613_3
  have hr := e_1613_3
  simp only [e_1613_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1613 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15857)) := by
  have c98784 := (publicBatchWrapper89_copies3087 a h).1
  have c98785 := (publicBatchWrapper89_copies3087 a h).2.1
  have c98786 := (publicBatchWrapper89_copies3087 a h).2.2.1
  have c98787 := (publicBatchWrapper89_copies3087 a h).2.2.2.1
  have c98788 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.1
  have c98789 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1613_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_4
  simp only [← c98784, ← c98785, ← c98786] at e_1613_4
  have e_1613_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_5
  simp only [← c98787, ← c98788, k2, ← c98789] at e_1613_5
  have hr := e_1613_5
  simp only [e_1613_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1613 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15858)) := by
  have c98790 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.1
  have c98791 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.1
  have c98792 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.1
  have c98793 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.1
  have c98794 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98795 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1613_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_6
  simp only [← c98790, ← c98791, ← c98792] at e_1613_6
  have e_1613_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_7
  simp only [← c98793, ← c98794, k2, ← c98795] at e_1613_7
  have hr := e_1613_7
  simp only [e_1613_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1613 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15859)) := by
  have c98796 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98797 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98798 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98799 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98800 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98801 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1613_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_8
  simp only [← c98796, ← c98797, ← c98798] at e_1613_8
  have e_1613_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_9
  simp only [← c98799, ← c98800, k2, ← c98801] at e_1613_9
  have hr := e_1613_9
  simp only [e_1613_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1613 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15860)) := by
  have c98802 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98803 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98804 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98805 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98806 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98807 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1613_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_10
  simp only [← c98802, ← c98803, ← c98804] at e_1613_10
  have e_1613_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_11
  simp only [← c98805, ← c98806, k2, ← c98807] at e_1613_11
  have hr := e_1613_11
  simp only [e_1613_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1613 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15861)) := by
  have c98808 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98809 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98810 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98811 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98812 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98813 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1613_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_12
  simp only [← c98808, ← c98809, ← c98810] at e_1613_12
  have e_1613_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_13
  simp only [← c98811, ← c98812, k2, ← c98813] at e_1613_13
  have hr := e_1613_13
  simp only [e_1613_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1613 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15862)) := by
  have c98814 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98815 := (publicBatchWrapper89_copies3087 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98816 := (publicBatchWrapper89_copies3088 a h).1
  have c98817 := (publicBatchWrapper89_copies3088 a h).2.1
  have c98818 := (publicBatchWrapper89_copies3088 a h).2.2.1
  have c98819 := (publicBatchWrapper89_copies3088 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1613_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_14
  simp only [← c98814, ← c98815, ← c98816] at e_1613_14
  have e_1613_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_15
  simp only [← c98817, ← c98818, k2, ← c98819] at e_1613_15
  have hr := e_1613_15
  simp only [e_1613_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1613 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15863)) := by
  have c98820 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.1
  have c98821 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.1
  have c98822 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.1
  have c98823 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.1
  have c98824 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.1
  have c98825 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1613_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_16
  simp only [← c98820, ← c98821, ← c98822] at e_1613_16
  have e_1613_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_17
  simp only [← c98823, ← c98824, k2, ← c98825] at e_1613_17
  have hr := e_1613_17
  simp only [e_1613_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1613 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15864)) := by
  have c98826 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98827 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98828 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98829 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98830 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98831 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1613_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_18
  simp only [← c98826, ← c98827, ← c98828] at e_1613_18
  have e_1613_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1613 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1613_19
  simp only [← c98829, ← c98830, k2, ← c98831] at e_1613_19
  have hr := e_1613_19
  simp only [e_1613_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1614 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15865)) := by
  have c98832 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98833 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98834 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98835 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98836 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98837 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1614_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_0
  simp only [← c98832, ← c98833, ← c98834] at e_1614_0
  have e_1614_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_1
  simp only [← c98835, ← c98836, k2, ← c98837] at e_1614_1
  have hr := e_1614_1
  simp only [e_1614_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1614 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15866)) := by
  have c98838 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98839 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98840 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98841 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98842 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98843 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1614_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_2
  simp only [← c98838, ← c98839, ← c98840] at e_1614_2
  have e_1614_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_3
  simp only [← c98841, ← c98842, k2, ← c98843] at e_1614_3
  have hr := e_1614_3
  simp only [e_1614_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1614 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15867)) := by
  have c98844 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98845 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98846 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98847 := (publicBatchWrapper89_copies3088 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98848 := (publicBatchWrapper89_copies3089 a h).1
  have c98849 := (publicBatchWrapper89_copies3089 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1614_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_4
  simp only [← c98844, ← c98845, ← c98846] at e_1614_4
  have e_1614_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_5
  simp only [← c98847, ← c98848, k2, ← c98849] at e_1614_5
  have hr := e_1614_5
  simp only [e_1614_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1614 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15868)) := by
  have c98850 := (publicBatchWrapper89_copies3089 a h).2.2.1
  have c98851 := (publicBatchWrapper89_copies3089 a h).2.2.2.1
  have c98852 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.1
  have c98853 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.1
  have c98854 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.1
  have c98855 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1614_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_6
  simp only [← c98850, ← c98851, ← c98852] at e_1614_6
  have e_1614_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_7
  simp only [← c98853, ← c98854, k2, ← c98855] at e_1614_7
  have hr := e_1614_7
  simp only [e_1614_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1614 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15869)) := by
  have c98856 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.1
  have c98857 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.1
  have c98858 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98859 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98860 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98861 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1614_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_8
  simp only [← c98856, ← c98857, ← c98858] at e_1614_8
  have e_1614_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_9
  simp only [← c98859, ← c98860, k2, ← c98861] at e_1614_9
  have hr := e_1614_9
  simp only [e_1614_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1614 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15870)) := by
  have c98862 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98863 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98864 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98865 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98866 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98867 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1614_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_10
  simp only [← c98862, ← c98863, ← c98864] at e_1614_10
  have e_1614_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_11
  simp only [← c98865, ← c98866, k2, ← c98867] at e_1614_11
  have hr := e_1614_11
  simp only [e_1614_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1614 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15871)) := by
  have c98868 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98869 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98870 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98871 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98872 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98873 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1614_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_12
  simp only [← c98868, ← c98869, ← c98870] at e_1614_12
  have e_1614_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_13
  simp only [← c98871, ← c98872, k2, ← c98873] at e_1614_13
  have hr := e_1614_13
  simp only [e_1614_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1614 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15872)) := by
  have c98874 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98875 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98876 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98877 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98878 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98879 := (publicBatchWrapper89_copies3089 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1614_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_14
  simp only [← c98874, ← c98875, ← c98876] at e_1614_14
  have e_1614_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_15
  simp only [← c98877, ← c98878, k2, ← c98879] at e_1614_15
  have hr := e_1614_15
  simp only [e_1614_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1614 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15873)) := by
  have c98880 := (publicBatchWrapper89_copies3090 a h).1
  have c98881 := (publicBatchWrapper89_copies3090 a h).2.1
  have c98882 := (publicBatchWrapper89_copies3090 a h).2.2.1
  have c98883 := (publicBatchWrapper89_copies3090 a h).2.2.2.1
  have c98884 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.1
  have c98885 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1614_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_16
  simp only [← c98880, ← c98881, ← c98882] at e_1614_16
  have e_1614_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_17
  simp only [← c98883, ← c98884, k2, ← c98885] at e_1614_17
  have hr := e_1614_17
  simp only [e_1614_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1614 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15874)) := by
  have c98886 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.1
  have c98887 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.1
  have c98888 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.1
  have c98889 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.1
  have c98890 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98891 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1614_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_18
  simp only [← c98886, ← c98887, ← c98888] at e_1614_18
  have e_1614_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1614 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1614_19
  simp only [← c98889, ← c98890, k2, ← c98891] at e_1614_19
  have hr := e_1614_19
  simp only [e_1614_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1615 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15875)) := by
  have c98892 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98893 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98894 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98895 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98896 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98897 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1615_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_0
  simp only [← c98892, ← c98893, ← c98894] at e_1615_0
  have e_1615_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_1
  simp only [← c98895, ← c98896, k2, ← c98897] at e_1615_1
  have hr := e_1615_1
  simp only [e_1615_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1615 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15876)) := by
  have c98898 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98899 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98900 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98901 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98902 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98903 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1615_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_2
  simp only [← c98898, ← c98899, ← c98900] at e_1615_2
  have e_1615_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_3
  simp only [← c98901, ← c98902, k2, ← c98903] at e_1615_3
  have hr := e_1615_3
  simp only [e_1615_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1615 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15877)) := by
  have c98904 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98905 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98906 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98907 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98908 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98909 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1615_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_4
  simp only [← c98904, ← c98905, ← c98906] at e_1615_4
  have e_1615_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_5
  simp only [← c98907, ← c98908, k2, ← c98909] at e_1615_5
  have hr := e_1615_5
  simp only [e_1615_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1615 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15878)) := by
  have c98910 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98911 := (publicBatchWrapper89_copies3090 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98912 := (publicBatchWrapper89_copies3091 a h).1
  have c98913 := (publicBatchWrapper89_copies3091 a h).2.1
  have c98914 := (publicBatchWrapper89_copies3091 a h).2.2.1
  have c98915 := (publicBatchWrapper89_copies3091 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1615_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_6
  simp only [← c98910, ← c98911, ← c98912] at e_1615_6
  have e_1615_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_7
  simp only [← c98913, ← c98914, k2, ← c98915] at e_1615_7
  have hr := e_1615_7
  simp only [e_1615_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1615 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15879)) := by
  have c98916 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.1
  have c98917 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.1
  have c98918 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.1
  have c98919 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.1
  have c98920 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.1
  have c98921 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1615_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_8
  simp only [← c98916, ← c98917, ← c98918] at e_1615_8
  have e_1615_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_9
  simp only [← c98919, ← c98920, k2, ← c98921] at e_1615_9
  have hr := e_1615_9
  simp only [e_1615_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1615 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15880)) := by
  have c98922 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98923 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98924 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98925 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98926 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98927 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1615_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_10
  simp only [← c98922, ← c98923, ← c98924] at e_1615_10
  have e_1615_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_11
  simp only [← c98925, ← c98926, k2, ← c98927] at e_1615_11
  have hr := e_1615_11
  simp only [e_1615_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1615 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15881)) := by
  have c98928 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98929 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98930 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98931 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98932 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98933 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1615_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_12
  simp only [← c98928, ← c98929, ← c98930] at e_1615_12
  have e_1615_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_13
  simp only [← c98931, ← c98932, k2, ← c98933] at e_1615_13
  have hr := e_1615_13
  simp only [e_1615_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1615 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15882)) := by
  have c98934 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98935 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98936 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98937 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98938 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98939 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1615_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_14
  simp only [← c98934, ← c98935, ← c98936] at e_1615_14
  have e_1615_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_15
  simp only [← c98937, ← c98938, k2, ← c98939] at e_1615_15
  have hr := e_1615_15
  simp only [e_1615_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1615 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15883)) := by
  have c98940 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98941 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98942 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98943 := (publicBatchWrapper89_copies3091 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c98944 := (publicBatchWrapper89_copies3092 a h).1
  have c98945 := (publicBatchWrapper89_copies3092 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1615_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_16
  simp only [← c98940, ← c98941, ← c98942] at e_1615_16
  have e_1615_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_17
  simp only [← c98943, ← c98944, k2, ← c98945] at e_1615_17
  have hr := e_1615_17
  simp only [e_1615_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1615 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15884)) := by
  have c98946 := (publicBatchWrapper89_copies3092 a h).2.2.1
  have c98947 := (publicBatchWrapper89_copies3092 a h).2.2.2.1
  have c98948 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.1
  have c98949 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.1
  have c98950 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.1
  have c98951 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1615_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_18
  simp only [← c98946, ← c98947, ← c98948] at e_1615_18
  have e_1615_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1615 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1615_19
  simp only [← c98949, ← c98950, k2, ← c98951] at e_1615_19
  have hr := e_1615_19
  simp only [e_1615_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1616 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15885)) := by
  have c98952 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.1
  have c98953 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.1
  have c98954 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98955 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c98956 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98957 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1616_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_0
  simp only [← c98952, ← c98953, ← c98954] at e_1616_0
  have e_1616_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_1
  simp only [← c98955, ← c98956, k2, ← c98957] at e_1616_1
  have hr := e_1616_1
  simp only [e_1616_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1616 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15886)) := by
  have c98958 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98959 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98960 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98961 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98962 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98963 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1616_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_2
  simp only [← c98958, ← c98959, ← c98960] at e_1616_2
  have e_1616_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_3
  simp only [← c98961, ← c98962, k2, ← c98963] at e_1616_3
  have hr := e_1616_3
  simp only [e_1616_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1616 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15887)) := by
  have c98964 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98965 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98966 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98967 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98968 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98969 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1616_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_4
  simp only [← c98964, ← c98965, ← c98966] at e_1616_4
  have e_1616_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_5
  simp only [← c98967, ← c98968, k2, ← c98969] at e_1616_5
  have hr := e_1616_5
  simp only [e_1616_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1616 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15888)) := by
  have c98970 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98971 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98972 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98973 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98974 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98975 := (publicBatchWrapper89_copies3092 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1616_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_6
  simp only [← c98970, ← c98971, ← c98972] at e_1616_6
  have e_1616_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_7
  simp only [← c98973, ← c98974, k2, ← c98975] at e_1616_7
  have hr := e_1616_7
  simp only [e_1616_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1616 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15889)) := by
  have c98976 := (publicBatchWrapper89_copies3093 a h).1
  have c98977 := (publicBatchWrapper89_copies3093 a h).2.1
  have c98978 := (publicBatchWrapper89_copies3093 a h).2.2.1
  have c98979 := (publicBatchWrapper89_copies3093 a h).2.2.2.1
  have c98980 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.1
  have c98981 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1616_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_8
  simp only [← c98976, ← c98977, ← c98978] at e_1616_8
  have e_1616_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_9
  simp only [← c98979, ← c98980, k2, ← c98981] at e_1616_9
  have hr := e_1616_9
  simp only [e_1616_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1616 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15890)) := by
  have c98982 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.1
  have c98983 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.1
  have c98984 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.1
  have c98985 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.1
  have c98986 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.1
  have c98987 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1616_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_10
  simp only [← c98982, ← c98983, ← c98984] at e_1616_10
  have e_1616_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_11
  simp only [← c98985, ← c98986, k2, ← c98987] at e_1616_11
  have hr := e_1616_11
  simp only [e_1616_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1616 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15891)) := by
  have c98988 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98989 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98990 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98991 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98992 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98993 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1616_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_12
  simp only [← c98988, ← c98989, ← c98990] at e_1616_12
  have e_1616_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_13
  simp only [← c98991, ← c98992, k2, ← c98993] at e_1616_13
  have hr := e_1616_13
  simp only [e_1616_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1616 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15892)) := by
  have c98994 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98995 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98996 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98997 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98998 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c98999 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1616_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_14
  simp only [← c98994, ← c98995, ← c98996] at e_1616_14
  have e_1616_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_15
  simp only [← c98997, ← c98998, k2, ← c98999] at e_1616_15
  have hr := e_1616_15
  simp only [e_1616_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1616 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15893)) := by
  have c99000 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99001 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99002 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99003 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99004 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99005 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1616_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_16
  simp only [← c99000, ← c99001, ← c99002] at e_1616_16
  have e_1616_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_17
  simp only [← c99003, ← c99004, k2, ← c99005] at e_1616_17
  have hr := e_1616_17
  simp only [e_1616_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1616 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15894)) := by
  have c99006 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99007 := (publicBatchWrapper89_copies3093 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99008 := (publicBatchWrapper89_copies3094 a h).1
  have c99009 := (publicBatchWrapper89_copies3094 a h).2.1
  have c99010 := (publicBatchWrapper89_copies3094 a h).2.2.1
  have c99011 := (publicBatchWrapper89_copies3094 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1616_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_18
  simp only [← c99006, ← c99007, ← c99008] at e_1616_18
  have e_1616_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1616 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1616_19
  simp only [← c99009, ← c99010, k2, ← c99011] at e_1616_19
  have hr := e_1616_19
  simp only [e_1616_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1617 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15895)) := by
  have c99012 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.1
  have c99013 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.1
  have c99014 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.1
  have c99015 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.1
  have c99016 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.1
  have c99017 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1617_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_0
  simp only [← c99012, ← c99013, ← c99014] at e_1617_0
  have e_1617_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_1
  simp only [← c99015, ← c99016, k2, ← c99017] at e_1617_1
  have hr := e_1617_1
  simp only [e_1617_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1617 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15896)) := by
  have c99018 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99019 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99020 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99021 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99022 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99023 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1617_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_2
  simp only [← c99018, ← c99019, ← c99020] at e_1617_2
  have e_1617_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_3
  simp only [← c99021, ← c99022, k2, ← c99023] at e_1617_3
  have hr := e_1617_3
  simp only [e_1617_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1617 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15897)) := by
  have c99024 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99025 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99026 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99027 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99028 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99029 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1617_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_4
  simp only [← c99024, ← c99025, ← c99026] at e_1617_4
  have e_1617_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_5
  simp only [← c99027, ← c99028, k2, ← c99029] at e_1617_5
  have hr := e_1617_5
  simp only [e_1617_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1617 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15898)) := by
  have c99030 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99031 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99032 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99033 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99034 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99035 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1617_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_6
  simp only [← c99030, ← c99031, ← c99032] at e_1617_6
  have e_1617_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_7
  simp only [← c99033, ← c99034, k2, ← c99035] at e_1617_7
  have hr := e_1617_7
  simp only [e_1617_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1617 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15899)) := by
  have c99036 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99037 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99038 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99039 := (publicBatchWrapper89_copies3094 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99040 := (publicBatchWrapper89_copies3095 a h).1
  have c99041 := (publicBatchWrapper89_copies3095 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1617_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_8
  simp only [← c99036, ← c99037, ← c99038] at e_1617_8
  have e_1617_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_9
  simp only [← c99039, ← c99040, k2, ← c99041] at e_1617_9
  have hr := e_1617_9
  simp only [e_1617_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1617 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15900)) := by
  have c99042 := (publicBatchWrapper89_copies3095 a h).2.2.1
  have c99043 := (publicBatchWrapper89_copies3095 a h).2.2.2.1
  have c99044 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.1
  have c99045 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.1
  have c99046 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.1
  have c99047 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1617_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_10
  simp only [← c99042, ← c99043, ← c99044] at e_1617_10
  have e_1617_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_11
  simp only [← c99045, ← c99046, k2, ← c99047] at e_1617_11
  have hr := e_1617_11
  simp only [e_1617_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1617 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15901)) := by
  have c99048 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.1
  have c99049 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.1
  have c99050 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99051 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99052 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99053 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1617_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_12
  simp only [← c99048, ← c99049, ← c99050] at e_1617_12
  have e_1617_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_13
  simp only [← c99051, ← c99052, k2, ← c99053] at e_1617_13
  have hr := e_1617_13
  simp only [e_1617_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1617 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15902)) := by
  have c99054 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99055 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99056 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99057 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99058 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99059 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1617_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_14
  simp only [← c99054, ← c99055, ← c99056] at e_1617_14
  have e_1617_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_15
  simp only [← c99057, ← c99058, k2, ← c99059] at e_1617_15
  have hr := e_1617_15
  simp only [e_1617_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1617 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15903)) := by
  have c99060 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99061 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99062 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99063 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99064 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99065 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1617_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_16
  simp only [← c99060, ← c99061, ← c99062] at e_1617_16
  have e_1617_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_17
  simp only [← c99063, ← c99064, k2, ← c99065] at e_1617_17
  have hr := e_1617_17
  simp only [e_1617_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1617 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15904)) := by
  have c99066 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99067 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99068 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99069 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99070 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99071 := (publicBatchWrapper89_copies3095 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1617_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_18
  simp only [← c99066, ← c99067, ← c99068] at e_1617_18
  have e_1617_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1617 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1617_19
  simp only [← c99069, ← c99070, k2, ← c99071] at e_1617_19
  have hr := e_1617_19
  simp only [e_1617_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1618 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15905)) := by
  have c99072 := (publicBatchWrapper89_copies3096 a h).1
  have c99073 := (publicBatchWrapper89_copies3096 a h).2.1
  have c99074 := (publicBatchWrapper89_copies3096 a h).2.2.1
  have c99075 := (publicBatchWrapper89_copies3096 a h).2.2.2.1
  have c99076 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.1
  have c99077 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1618_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_0
  simp only [← c99072, ← c99073, ← c99074] at e_1618_0
  have e_1618_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_1
  simp only [← c99075, ← c99076, k2, ← c99077] at e_1618_1
  have hr := e_1618_1
  simp only [e_1618_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1618 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15906)) := by
  have c99078 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.1
  have c99079 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.1
  have c99080 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.1
  have c99081 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.1
  have c99082 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99083 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1618_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_2
  simp only [← c99078, ← c99079, ← c99080] at e_1618_2
  have e_1618_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_3
  simp only [← c99081, ← c99082, k2, ← c99083] at e_1618_3
  have hr := e_1618_3
  simp only [e_1618_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1618 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15907)) := by
  have c99084 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99085 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99086 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99087 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99088 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99089 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1618_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_4
  simp only [← c99084, ← c99085, ← c99086] at e_1618_4
  have e_1618_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_5
  simp only [← c99087, ← c99088, k2, ← c99089] at e_1618_5
  have hr := e_1618_5
  simp only [e_1618_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1618 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15908)) := by
  have c99090 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99091 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99092 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99093 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99094 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99095 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1618_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_6
  simp only [← c99090, ← c99091, ← c99092] at e_1618_6
  have e_1618_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_7
  simp only [← c99093, ← c99094, k2, ← c99095] at e_1618_7
  have hr := e_1618_7
  simp only [e_1618_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1618 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15909)) := by
  have c99096 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99097 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99098 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99099 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99100 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99101 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1618_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_8
  simp only [← c99096, ← c99097, ← c99098] at e_1618_8
  have e_1618_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_9
  simp only [← c99099, ← c99100, k2, ← c99101] at e_1618_9
  have hr := e_1618_9
  simp only [e_1618_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1618 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15910)) := by
  have c99102 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99103 := (publicBatchWrapper89_copies3096 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99104 := (publicBatchWrapper89_copies3097 a h).1
  have c99105 := (publicBatchWrapper89_copies3097 a h).2.1
  have c99106 := (publicBatchWrapper89_copies3097 a h).2.2.1
  have c99107 := (publicBatchWrapper89_copies3097 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1618_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_10
  simp only [← c99102, ← c99103, ← c99104] at e_1618_10
  have e_1618_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_11
  simp only [← c99105, ← c99106, k2, ← c99107] at e_1618_11
  have hr := e_1618_11
  simp only [e_1618_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1618 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15911)) := by
  have c99108 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.1
  have c99109 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.1
  have c99110 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.1
  have c99111 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.1
  have c99112 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.1
  have c99113 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1618_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_12
  simp only [← c99108, ← c99109, ← c99110] at e_1618_12
  have e_1618_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_13
  simp only [← c99111, ← c99112, k2, ← c99113] at e_1618_13
  have hr := e_1618_13
  simp only [e_1618_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1618 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15912)) := by
  have c99114 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99115 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99116 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99117 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99118 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99119 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1618_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_14
  simp only [← c99114, ← c99115, ← c99116] at e_1618_14
  have e_1618_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_15
  simp only [← c99117, ← c99118, k2, ← c99119] at e_1618_15
  have hr := e_1618_15
  simp only [e_1618_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1618 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15913)) := by
  have c99120 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99121 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99122 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99123 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99124 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99125 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1618_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_16
  simp only [← c99120, ← c99121, ← c99122] at e_1618_16
  have e_1618_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_17
  simp only [← c99123, ← c99124, k2, ← c99125] at e_1618_17
  have hr := e_1618_17
  simp only [e_1618_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1618 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15914)) := by
  have c99126 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99127 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99128 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99129 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99130 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99131 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1618_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_18
  simp only [← c99126, ← c99127, ← c99128] at e_1618_18
  have e_1618_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1618 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1618_19
  simp only [← c99129, ← c99130, k2, ← c99131] at e_1618_19
  have hr := e_1618_19
  simp only [e_1618_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1619 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15915)) := by
  have c99132 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99133 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99134 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99135 := (publicBatchWrapper89_copies3097 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99136 := (publicBatchWrapper89_copies3098 a h).1
  have c99137 := (publicBatchWrapper89_copies3098 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1619_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_0
  simp only [← c99132, ← c99133, ← c99134] at e_1619_0
  have e_1619_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_1
  simp only [← c99135, ← c99136, k2, ← c99137] at e_1619_1
  have hr := e_1619_1
  simp only [e_1619_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1619 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15916)) := by
  have c99138 := (publicBatchWrapper89_copies3098 a h).2.2.1
  have c99139 := (publicBatchWrapper89_copies3098 a h).2.2.2.1
  have c99140 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.1
  have c99141 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.1
  have c99142 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.1
  have c99143 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1619_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_2
  simp only [← c99138, ← c99139, ← c99140] at e_1619_2
  have e_1619_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_3
  simp only [← c99141, ← c99142, k2, ← c99143] at e_1619_3
  have hr := e_1619_3
  simp only [e_1619_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1619 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15917)) := by
  have c99144 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.1
  have c99145 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.1
  have c99146 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99147 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99148 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99149 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1619_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_4
  simp only [← c99144, ← c99145, ← c99146] at e_1619_4
  have e_1619_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_5
  simp only [← c99147, ← c99148, k2, ← c99149] at e_1619_5
  have hr := e_1619_5
  simp only [e_1619_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1619 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15918)) := by
  have c99150 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99151 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99152 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99153 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99154 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99155 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1619_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_6
  simp only [← c99150, ← c99151, ← c99152] at e_1619_6
  have e_1619_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_7
  simp only [← c99153, ← c99154, k2, ← c99155] at e_1619_7
  have hr := e_1619_7
  simp only [e_1619_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1619 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15919)) := by
  have c99156 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99157 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99158 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99159 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99160 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99161 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1619_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_8
  simp only [← c99156, ← c99157, ← c99158] at e_1619_8
  have e_1619_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_9
  simp only [← c99159, ← c99160, k2, ← c99161] at e_1619_9
  have hr := e_1619_9
  simp only [e_1619_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1619 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15920)) := by
  have c99162 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99163 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99164 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99165 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99166 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99167 := (publicBatchWrapper89_copies3098 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1619_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_10
  simp only [← c99162, ← c99163, ← c99164] at e_1619_10
  have e_1619_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_11
  simp only [← c99165, ← c99166, k2, ← c99167] at e_1619_11
  have hr := e_1619_11
  simp only [e_1619_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1619 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15921)) := by
  have c99168 := (publicBatchWrapper89_copies3099 a h).1
  have c99169 := (publicBatchWrapper89_copies3099 a h).2.1
  have c99170 := (publicBatchWrapper89_copies3099 a h).2.2.1
  have c99171 := (publicBatchWrapper89_copies3099 a h).2.2.2.1
  have c99172 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.1
  have c99173 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1619_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_12
  simp only [← c99168, ← c99169, ← c99170] at e_1619_12
  have e_1619_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_13
  simp only [← c99171, ← c99172, k2, ← c99173] at e_1619_13
  have hr := e_1619_13
  simp only [e_1619_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1619 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15922)) := by
  have c99174 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.1
  have c99175 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.1
  have c99176 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.1
  have c99177 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.1
  have c99178 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99179 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1619_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_14
  simp only [← c99174, ← c99175, ← c99176] at e_1619_14
  have e_1619_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_15
  simp only [← c99177, ← c99178, k2, ← c99179] at e_1619_15
  have hr := e_1619_15
  simp only [e_1619_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1619 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15923)) := by
  have c99180 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99181 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99182 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99183 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99184 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99185 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1619_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_16
  simp only [← c99180, ← c99181, ← c99182] at e_1619_16
  have e_1619_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_17
  simp only [← c99183, ← c99184, k2, ← c99185] at e_1619_17
  have hr := e_1619_17
  simp only [e_1619_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1619 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15924)) := by
  have c99186 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99187 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99188 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99189 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99190 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99191 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1619_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_18
  simp only [← c99186, ← c99187, ← c99188] at e_1619_18
  have e_1619_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1619 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1619_19
  simp only [← c99189, ← c99190, k2, ← c99191] at e_1619_19
  have hr := e_1619_19
  simp only [e_1619_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1620 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15925)) := by
  have c99192 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99193 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99194 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99195 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99196 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99197 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1620_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_0
  simp only [← c99192, ← c99193, ← c99194] at e_1620_0
  have e_1620_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_1
  simp only [← c99195, ← c99196, k2, ← c99197] at e_1620_1
  have hr := e_1620_1
  simp only [e_1620_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1620 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15926)) := by
  have c99198 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99199 := (publicBatchWrapper89_copies3099 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99200 := (publicBatchWrapper89_copies3100 a h).1
  have c99201 := (publicBatchWrapper89_copies3100 a h).2.1
  have c99202 := (publicBatchWrapper89_copies3100 a h).2.2.1
  have c99203 := (publicBatchWrapper89_copies3100 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1620_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_2
  simp only [← c99198, ← c99199, ← c99200] at e_1620_2
  have e_1620_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_3
  simp only [← c99201, ← c99202, k2, ← c99203] at e_1620_3
  have hr := e_1620_3
  simp only [e_1620_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1620 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15927)) := by
  have c99204 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.1
  have c99205 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.1
  have c99206 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.1
  have c99207 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.1
  have c99208 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.1
  have c99209 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1620_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_4
  simp only [← c99204, ← c99205, ← c99206] at e_1620_4
  have e_1620_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_5
  simp only [← c99207, ← c99208, k2, ← c99209] at e_1620_5
  have hr := e_1620_5
  simp only [e_1620_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1620 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15928)) := by
  have c99210 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99211 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99212 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99213 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99214 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99215 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1620_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_6
  simp only [← c99210, ← c99211, ← c99212] at e_1620_6
  have e_1620_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_7
  simp only [← c99213, ← c99214, k2, ← c99215] at e_1620_7
  have hr := e_1620_7
  simp only [e_1620_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1620 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15929)) := by
  have c99216 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99217 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99218 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99219 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99220 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99221 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1620_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_8
  simp only [← c99216, ← c99217, ← c99218] at e_1620_8
  have e_1620_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_9
  simp only [← c99219, ← c99220, k2, ← c99221] at e_1620_9
  have hr := e_1620_9
  simp only [e_1620_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1620 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15930)) := by
  have c99222 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99223 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99224 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99225 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99226 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99227 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1620_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_10
  simp only [← c99222, ← c99223, ← c99224] at e_1620_10
  have e_1620_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_11
  simp only [← c99225, ← c99226, k2, ← c99227] at e_1620_11
  have hr := e_1620_11
  simp only [e_1620_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1620 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15931)) := by
  have c99228 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99229 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99230 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99231 := (publicBatchWrapper89_copies3100 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99232 := (publicBatchWrapper89_copies3101 a h).1
  have c99233 := (publicBatchWrapper89_copies3101 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1620_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_12
  simp only [← c99228, ← c99229, ← c99230] at e_1620_12
  have e_1620_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_13
  simp only [← c99231, ← c99232, k2, ← c99233] at e_1620_13
  have hr := e_1620_13
  simp only [e_1620_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1620 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15932)) := by
  have c99234 := (publicBatchWrapper89_copies3101 a h).2.2.1
  have c99235 := (publicBatchWrapper89_copies3101 a h).2.2.2.1
  have c99236 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.1
  have c99237 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.1
  have c99238 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.1
  have c99239 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1620_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_14
  simp only [← c99234, ← c99235, ← c99236] at e_1620_14
  have e_1620_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_15
  simp only [← c99237, ← c99238, k2, ← c99239] at e_1620_15
  have hr := e_1620_15
  simp only [e_1620_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1620 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15933)) := by
  have c99240 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.1
  have c99241 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.1
  have c99242 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99243 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99244 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99245 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1620_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_16
  simp only [← c99240, ← c99241, ← c99242] at e_1620_16
  have e_1620_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_17
  simp only [← c99243, ← c99244, k2, ← c99245] at e_1620_17
  have hr := e_1620_17
  simp only [e_1620_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1620 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15934)) := by
  have c99246 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99247 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99248 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99249 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99250 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99251 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1620_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_18
  simp only [← c99246, ← c99247, ← c99248] at e_1620_18
  have e_1620_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1620 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1620_19
  simp only [← c99249, ← c99250, k2, ← c99251] at e_1620_19
  have hr := e_1620_19
  simp only [e_1620_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1621 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15935)) := by
  have c99252 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99253 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99254 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99255 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99256 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99257 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1621_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_0
  simp only [← c99252, ← c99253, ← c99254] at e_1621_0
  have e_1621_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_1
  simp only [← c99255, ← c99256, k2, ← c99257] at e_1621_1
  have hr := e_1621_1
  simp only [e_1621_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1621 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15936)) := by
  have c99258 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99259 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99260 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99261 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99262 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99263 := (publicBatchWrapper89_copies3101 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1621_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_2
  simp only [← c99258, ← c99259, ← c99260] at e_1621_2
  have e_1621_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_3
  simp only [← c99261, ← c99262, k2, ← c99263] at e_1621_3
  have hr := e_1621_3
  simp only [e_1621_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1621 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15937)) := by
  have c99264 := (publicBatchWrapper89_copies3102 a h).1
  have c99265 := (publicBatchWrapper89_copies3102 a h).2.1
  have c99266 := (publicBatchWrapper89_copies3102 a h).2.2.1
  have c99267 := (publicBatchWrapper89_copies3102 a h).2.2.2.1
  have c99268 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.1
  have c99269 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1621_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_4
  simp only [← c99264, ← c99265, ← c99266] at e_1621_4
  have e_1621_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_5
  simp only [← c99267, ← c99268, k2, ← c99269] at e_1621_5
  have hr := e_1621_5
  simp only [e_1621_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1621 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15938)) := by
  have c99270 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.1
  have c99271 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.1
  have c99272 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.1
  have c99273 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.1
  have c99274 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99275 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1621_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_6
  simp only [← c99270, ← c99271, ← c99272] at e_1621_6
  have e_1621_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_7
  simp only [← c99273, ← c99274, k2, ← c99275] at e_1621_7
  have hr := e_1621_7
  simp only [e_1621_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1621 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15939)) := by
  have c99276 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99277 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99278 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99279 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99280 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99281 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1621_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_8
  simp only [← c99276, ← c99277, ← c99278] at e_1621_8
  have e_1621_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_9
  simp only [← c99279, ← c99280, k2, ← c99281] at e_1621_9
  have hr := e_1621_9
  simp only [e_1621_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1621 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15940)) := by
  have c99282 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99283 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99284 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99285 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99286 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99287 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1621_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_10
  simp only [← c99282, ← c99283, ← c99284] at e_1621_10
  have e_1621_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_11
  simp only [← c99285, ← c99286, k2, ← c99287] at e_1621_11
  have hr := e_1621_11
  simp only [e_1621_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1621 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15941)) := by
  have c99288 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99289 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99290 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99291 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99292 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99293 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1621_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_12
  simp only [← c99288, ← c99289, ← c99290] at e_1621_12
  have e_1621_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_13
  simp only [← c99291, ← c99292, k2, ← c99293] at e_1621_13
  have hr := e_1621_13
  simp only [e_1621_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1621 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15942)) := by
  have c99294 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99295 := (publicBatchWrapper89_copies3102 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99296 := (publicBatchWrapper89_copies3103 a h).1
  have c99297 := (publicBatchWrapper89_copies3103 a h).2.1
  have c99298 := (publicBatchWrapper89_copies3103 a h).2.2.1
  have c99299 := (publicBatchWrapper89_copies3103 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1621_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_14
  simp only [← c99294, ← c99295, ← c99296] at e_1621_14
  have e_1621_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_15
  simp only [← c99297, ← c99298, k2, ← c99299] at e_1621_15
  have hr := e_1621_15
  simp only [e_1621_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1621 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15943)) := by
  have c99300 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.1
  have c99301 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.1
  have c99302 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.1
  have c99303 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.1
  have c99304 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.1
  have c99305 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1621_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_16
  simp only [← c99300, ← c99301, ← c99302] at e_1621_16
  have e_1621_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_17
  simp only [← c99303, ← c99304, k2, ← c99305] at e_1621_17
  have hr := e_1621_17
  simp only [e_1621_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1621 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15944)) := by
  have c99306 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99307 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99308 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99309 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99310 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99311 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1621_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_18
  simp only [← c99306, ← c99307, ← c99308] at e_1621_18
  have e_1621_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1621 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1621_19
  simp only [← c99309, ← c99310, k2, ← c99311] at e_1621_19
  have hr := e_1621_19
  simp only [e_1621_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1622 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15945)) := by
  have c99312 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99313 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99314 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99315 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99316 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99317 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1622_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_0
  simp only [← c99312, ← c99313, ← c99314] at e_1622_0
  have e_1622_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_1
  simp only [← c99315, ← c99316, k2, ← c99317] at e_1622_1
  have hr := e_1622_1
  simp only [e_1622_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1622 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15946)) := by
  have c99318 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99319 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99320 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99321 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99322 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99323 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1622_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_2
  simp only [← c99318, ← c99319, ← c99320] at e_1622_2
  have e_1622_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_3
  simp only [← c99321, ← c99322, k2, ← c99323] at e_1622_3
  have hr := e_1622_3
  simp only [e_1622_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1622 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15947)) := by
  have c99324 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99325 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99326 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99327 := (publicBatchWrapper89_copies3103 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99328 := (publicBatchWrapper89_copies3104 a h).1
  have c99329 := (publicBatchWrapper89_copies3104 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1622_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_4
  simp only [← c99324, ← c99325, ← c99326] at e_1622_4
  have e_1622_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_5
  simp only [← c99327, ← c99328, k2, ← c99329] at e_1622_5
  have hr := e_1622_5
  simp only [e_1622_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1622 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15948)) := by
  have c99330 := (publicBatchWrapper89_copies3104 a h).2.2.1
  have c99331 := (publicBatchWrapper89_copies3104 a h).2.2.2.1
  have c99332 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.1
  have c99333 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.1
  have c99334 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.1
  have c99335 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1622_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_6
  simp only [← c99330, ← c99331, ← c99332] at e_1622_6
  have e_1622_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_7
  simp only [← c99333, ← c99334, k2, ← c99335] at e_1622_7
  have hr := e_1622_7
  simp only [e_1622_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1622 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15949)) := by
  have c99336 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.1
  have c99337 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.1
  have c99338 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99339 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99340 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99341 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1622_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_8
  simp only [← c99336, ← c99337, ← c99338] at e_1622_8
  have e_1622_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_9
  simp only [← c99339, ← c99340, k2, ← c99341] at e_1622_9
  have hr := e_1622_9
  simp only [e_1622_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1622 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15950)) := by
  have c99342 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99343 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99344 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99345 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99346 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99347 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1622_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_10
  simp only [← c99342, ← c99343, ← c99344] at e_1622_10
  have e_1622_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_11
  simp only [← c99345, ← c99346, k2, ← c99347] at e_1622_11
  have hr := e_1622_11
  simp only [e_1622_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1622 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15951)) := by
  have c99348 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99349 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99350 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99351 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99352 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99353 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1622_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_12
  simp only [← c99348, ← c99349, ← c99350] at e_1622_12
  have e_1622_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_13
  simp only [← c99351, ← c99352, k2, ← c99353] at e_1622_13
  have hr := e_1622_13
  simp only [e_1622_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1622 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15952)) := by
  have c99354 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99355 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99356 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99357 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99358 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99359 := (publicBatchWrapper89_copies3104 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1622_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_14
  simp only [← c99354, ← c99355, ← c99356] at e_1622_14
  have e_1622_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_15
  simp only [← c99357, ← c99358, k2, ← c99359] at e_1622_15
  have hr := e_1622_15
  simp only [e_1622_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1622 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15953)) := by
  have c99360 := (publicBatchWrapper89_copies3105 a h).1
  have c99361 := (publicBatchWrapper89_copies3105 a h).2.1
  have c99362 := (publicBatchWrapper89_copies3105 a h).2.2.1
  have c99363 := (publicBatchWrapper89_copies3105 a h).2.2.2.1
  have c99364 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.1
  have c99365 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1622_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_16
  simp only [← c99360, ← c99361, ← c99362] at e_1622_16
  have e_1622_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_17
  simp only [← c99363, ← c99364, k2, ← c99365] at e_1622_17
  have hr := e_1622_17
  simp only [e_1622_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1622 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15954)) := by
  have c99366 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.1
  have c99367 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.1
  have c99368 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.1
  have c99369 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.1
  have c99370 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99371 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1622_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_18
  simp only [← c99366, ← c99367, ← c99368] at e_1622_18
  have e_1622_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1622 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1622_19
  simp only [← c99369, ← c99370, k2, ← c99371] at e_1622_19
  have hr := e_1622_19
  simp only [e_1622_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1623 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15955)) := by
  have c99372 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99373 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99374 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99375 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99376 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99377 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1623_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_0
  simp only [← c99372, ← c99373, ← c99374] at e_1623_0
  have e_1623_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_1
  simp only [← c99375, ← c99376, k2, ← c99377] at e_1623_1
  have hr := e_1623_1
  simp only [e_1623_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1623 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15956)) := by
  have c99378 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99379 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99380 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99381 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99382 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99383 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1623_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_2
  simp only [← c99378, ← c99379, ← c99380] at e_1623_2
  have e_1623_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_3
  simp only [← c99381, ← c99382, k2, ← c99383] at e_1623_3
  have hr := e_1623_3
  simp only [e_1623_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1623 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15957)) := by
  have c99384 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99385 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99386 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99387 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99388 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99389 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1623_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_4
  simp only [← c99384, ← c99385, ← c99386] at e_1623_4
  have e_1623_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_5
  simp only [← c99387, ← c99388, k2, ← c99389] at e_1623_5
  have hr := e_1623_5
  simp only [e_1623_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1623 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15958)) := by
  have c99390 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99391 := (publicBatchWrapper89_copies3105 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99392 := (publicBatchWrapper89_copies3106 a h).1
  have c99393 := (publicBatchWrapper89_copies3106 a h).2.1
  have c99394 := (publicBatchWrapper89_copies3106 a h).2.2.1
  have c99395 := (publicBatchWrapper89_copies3106 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1623_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_6
  simp only [← c99390, ← c99391, ← c99392] at e_1623_6
  have e_1623_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_7
  simp only [← c99393, ← c99394, k2, ← c99395] at e_1623_7
  have hr := e_1623_7
  simp only [e_1623_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1623 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15959)) := by
  have c99396 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.1
  have c99397 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.1
  have c99398 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.1
  have c99399 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.1
  have c99400 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.1
  have c99401 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1623_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_8
  simp only [← c99396, ← c99397, ← c99398] at e_1623_8
  have e_1623_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_9
  simp only [← c99399, ← c99400, k2, ← c99401] at e_1623_9
  have hr := e_1623_9
  simp only [e_1623_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1623 47) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15960)) := by
  have c99402 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99403 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99404 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99405 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99406 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99407 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1623_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_10
  simp only [← c99402, ← c99403, ← c99404] at e_1623_10
  have e_1623_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_11
  simp only [← c99405, ← c99406, k2, ← c99407] at e_1623_11
  have hr := e_1623_11
  simp only [e_1623_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1623 55) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15961)) := by
  have c99408 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99409 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99410 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99411 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99412 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99413 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1623_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_12
  simp only [← c99408, ← c99409, ← c99410] at e_1623_12
  have e_1623_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_13
  simp only [← c99411, ← c99412, k2, ← c99413] at e_1623_13
  have hr := e_1623_13
  simp only [e_1623_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1623 63) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15962)) := by
  have c99414 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99415 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99416 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99417 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99418 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99419 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1623_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_14
  simp only [← c99414, ← c99415, ← c99416] at e_1623_14
  have e_1623_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_15
  simp only [← c99417, ← c99418, k2, ← c99419] at e_1623_15
  have hr := e_1623_15
  simp only [e_1623_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1623 71) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15963)) := by
  have c99420 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99421 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99422 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99423 := (publicBatchWrapper89_copies3106 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99424 := (publicBatchWrapper89_copies3107 a h).1
  have c99425 := (publicBatchWrapper89_copies3107 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1623_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_16
  simp only [← c99420, ← c99421, ← c99422] at e_1623_16
  have e_1623_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_17
  simp only [← c99423, ← c99424, k2, ← c99425] at e_1623_17
  have hr := e_1623_17
  simp only [e_1623_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1623 79) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15964)) := by
  have c99426 := (publicBatchWrapper89_copies3107 a h).2.2.1
  have c99427 := (publicBatchWrapper89_copies3107 a h).2.2.2.1
  have c99428 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.1
  have c99429 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.1
  have c99430 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.1
  have c99431 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1623_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_18
  simp only [← c99426, ← c99427, ← c99428] at e_1623_18
  have e_1623_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1623 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1623_19
  simp only [← c99429, ← c99430, k2, ← c99431] at e_1623_19
  have hr := e_1623_19
  simp only [e_1623_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1624 7) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15965)) := by
  have c99432 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.1
  have c99433 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.1
  have c99434 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99435 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99436 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99437 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1624_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_0
  simp only [← c99432, ← c99433, ← c99434] at e_1624_0
  have e_1624_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_1
  simp only [← c99435, ← c99436, k2, ← c99437] at e_1624_1
  have hr := e_1624_1
  simp only [e_1624_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1624 15) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15966)) := by
  have c99438 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99439 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99440 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99441 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99442 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99443 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1624_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_2
  simp only [← c99438, ← c99439, ← c99440] at e_1624_2
  have e_1624_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_3
  simp only [← c99441, ← c99442, k2, ← c99443] at e_1624_3
  have hr := e_1624_3
  simp only [e_1624_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1624 23) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15967)) := by
  have c99444 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99445 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99446 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99447 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99448 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99449 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1624_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_4
  simp only [← c99444, ← c99445, ← c99446] at e_1624_4
  have e_1624_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_5
  simp only [← c99447, ← c99448, k2, ← c99449] at e_1624_5
  have hr := e_1624_5
  simp only [e_1624_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1624 31) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15968)) := by
  have c99450 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99451 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99452 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99453 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99454 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99455 := (publicBatchWrapper89_copies3107 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1624_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_6
  simp only [← c99450, ← c99451, ← c99452] at e_1624_6
  have e_1624_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_7
  simp only [← c99453, ← c99454, k2, ← c99455] at e_1624_7
  have hr := e_1624_7
  simp only [e_1624_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1624 39) = bselect (a (.wire 77 15)) (a (.virt 200)) (a (.virt 15969)) := by
  have c99456 := (publicBatchWrapper89_copies3108 a h).1
  have c99457 := (publicBatchWrapper89_copies3108 a h).2.1
  have c99458 := (publicBatchWrapper89_copies3108 a h).2.2.1
  have c99459 := (publicBatchWrapper89_copies3108 a h).2.2.2.1
  have c99460 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.1
  have c99461 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1624_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_8
  simp only [← c99456, ← c99457, ← c99458] at e_1624_8
  have e_1624_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_9
  simp only [← c99459, ← c99460, k2, ← c99461] at e_1624_9
  have hr := e_1624_9
  simp only [e_1624_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1624 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16010)) := by
  have c99462 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.1
  have c99463 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.1
  have c99464 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.1
  have c99465 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.1
  have c99466 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99467 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1624_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_10
  simp only [← c99462, ← c99463, ← c99464] at e_1624_10
  have e_1624_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_11
  simp only [← c99465, ← c99466, k2, ← c99467] at e_1624_11
  have hr := e_1624_11
  simp only [e_1624_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1624 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16011)) := by
  have c99468 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99469 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99470 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99471 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99472 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99473 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1624_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_12
  simp only [← c99468, ← c99469, ← c99470] at e_1624_12
  have e_1624_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_13
  simp only [← c99471, ← c99472, k2, ← c99473] at e_1624_13
  have hr := e_1624_13
  simp only [e_1624_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1624 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16012)) := by
  have c99474 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99475 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99476 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99477 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99478 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99479 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1624_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_14
  simp only [← c99474, ← c99475, ← c99476] at e_1624_14
  have e_1624_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_15
  simp only [← c99477, ← c99478, k2, ← c99479] at e_1624_15
  have hr := e_1624_15
  simp only [e_1624_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1624 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16013)) := by
  have c99480 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99481 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99482 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99483 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99484 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99485 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1624_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_16
  simp only [← c99480, ← c99481, ← c99482] at e_1624_16
  have e_1624_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_17
  simp only [← c99483, ← c99484, k2, ← c99485] at e_1624_17
  have hr := e_1624_17
  simp only [e_1624_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1624 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16014)) := by
  have c99486 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99487 := (publicBatchWrapper89_copies3108 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99488 := (publicBatchWrapper89_copies3109 a h).1
  have c99489 := (publicBatchWrapper89_copies3109 a h).2.1
  have c99490 := (publicBatchWrapper89_copies3109 a h).2.2.1
  have c99491 := (publicBatchWrapper89_copies3109 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1624_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_18
  simp only [← c99486, ← c99487, ← c99488] at e_1624_18
  have e_1624_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1624 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1624_19
  simp only [← c99489, ← c99490, k2, ← c99491] at e_1624_19
  have hr := e_1624_19
  simp only [e_1624_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1625 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16015)) := by
  have c99492 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.1
  have c99493 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.1
  have c99494 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.1
  have c99495 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.1
  have c99496 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.1
  have c99497 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1625_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_0
  simp only [← c99492, ← c99493, ← c99494] at e_1625_0
  have e_1625_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_1
  simp only [← c99495, ← c99496, k2, ← c99497] at e_1625_1
  have hr := e_1625_1
  simp only [e_1625_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1625 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16016)) := by
  have c99498 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99499 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99500 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99501 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99502 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99503 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1625_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_2
  simp only [← c99498, ← c99499, ← c99500] at e_1625_2
  have e_1625_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_3
  simp only [← c99501, ← c99502, k2, ← c99503] at e_1625_3
  have hr := e_1625_3
  simp only [e_1625_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1625 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16017)) := by
  have c99504 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99505 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99506 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99507 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99508 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99509 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1625_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_4
  simp only [← c99504, ← c99505, ← c99506] at e_1625_4
  have e_1625_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_5
  simp only [← c99507, ← c99508, k2, ← c99509] at e_1625_5
  have hr := e_1625_5
  simp only [e_1625_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1625 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16018)) := by
  have c99510 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99511 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99512 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99513 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99514 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99515 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1625_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_6
  simp only [← c99510, ← c99511, ← c99512] at e_1625_6
  have e_1625_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_7
  simp only [← c99513, ← c99514, k2, ← c99515] at e_1625_7
  have hr := e_1625_7
  simp only [e_1625_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1625 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16019)) := by
  have c99516 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99517 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99518 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99519 := (publicBatchWrapper89_copies3109 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99520 := (publicBatchWrapper89_copies3110 a h).1
  have c99521 := (publicBatchWrapper89_copies3110 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1625_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_8
  simp only [← c99516, ← c99517, ← c99518] at e_1625_8
  have e_1625_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_9
  simp only [← c99519, ← c99520, k2, ← c99521] at e_1625_9
  have hr := e_1625_9
  simp only [e_1625_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1625 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16020)) := by
  have c99522 := (publicBatchWrapper89_copies3110 a h).2.2.1
  have c99523 := (publicBatchWrapper89_copies3110 a h).2.2.2.1
  have c99524 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.1
  have c99525 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.1
  have c99526 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.1
  have c99527 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1625_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_10
  simp only [← c99522, ← c99523, ← c99524] at e_1625_10
  have e_1625_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_11
  simp only [← c99525, ← c99526, k2, ← c99527] at e_1625_11
  have hr := e_1625_11
  simp only [e_1625_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1625 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16021)) := by
  have c99528 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.1
  have c99529 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.1
  have c99530 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99531 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99532 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99533 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1625_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_12
  simp only [← c99528, ← c99529, ← c99530] at e_1625_12
  have e_1625_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_13
  simp only [← c99531, ← c99532, k2, ← c99533] at e_1625_13
  have hr := e_1625_13
  simp only [e_1625_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1625 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16022)) := by
  have c99534 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99535 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99536 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99537 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99538 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99539 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1625_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_14
  simp only [← c99534, ← c99535, ← c99536] at e_1625_14
  have e_1625_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_15
  simp only [← c99537, ← c99538, k2, ← c99539] at e_1625_15
  have hr := e_1625_15
  simp only [e_1625_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1625 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16023)) := by
  have c99540 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99541 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99542 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99543 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99544 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99545 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1625_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_16
  simp only [← c99540, ← c99541, ← c99542] at e_1625_16
  have e_1625_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_17
  simp only [← c99543, ← c99544, k2, ← c99545] at e_1625_17
  have hr := e_1625_17
  simp only [e_1625_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1625 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16024)) := by
  have c99546 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99547 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99548 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99549 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99550 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99551 := (publicBatchWrapper89_copies3110 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1625_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_18
  simp only [← c99546, ← c99547, ← c99548] at e_1625_18
  have e_1625_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1625 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1625_19
  simp only [← c99549, ← c99550, k2, ← c99551] at e_1625_19
  have hr := e_1625_19
  simp only [e_1625_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1626 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16025)) := by
  have c99552 := (publicBatchWrapper89_copies3111 a h).1
  have c99553 := (publicBatchWrapper89_copies3111 a h).2.1
  have c99554 := (publicBatchWrapper89_copies3111 a h).2.2.1
  have c99555 := (publicBatchWrapper89_copies3111 a h).2.2.2.1
  have c99556 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.1
  have c99557 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1626_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_0
  simp only [← c99552, ← c99553, ← c99554] at e_1626_0
  have e_1626_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_1
  simp only [← c99555, ← c99556, k2, ← c99557] at e_1626_1
  have hr := e_1626_1
  simp only [e_1626_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1626 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16026)) := by
  have c99558 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.1
  have c99559 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.1
  have c99560 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.1
  have c99561 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.1
  have c99562 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99563 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1626_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_2
  simp only [← c99558, ← c99559, ← c99560] at e_1626_2
  have e_1626_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_3
  simp only [← c99561, ← c99562, k2, ← c99563] at e_1626_3
  have hr := e_1626_3
  simp only [e_1626_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1626 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16027)) := by
  have c99564 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99565 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99566 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99567 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99568 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99569 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1626_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_4
  simp only [← c99564, ← c99565, ← c99566] at e_1626_4
  have e_1626_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_5
  simp only [← c99567, ← c99568, k2, ← c99569] at e_1626_5
  have hr := e_1626_5
  simp only [e_1626_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1626 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16028)) := by
  have c99570 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99571 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99572 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99573 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99574 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99575 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1626_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_6
  simp only [← c99570, ← c99571, ← c99572] at e_1626_6
  have e_1626_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_7
  simp only [← c99573, ← c99574, k2, ← c99575] at e_1626_7
  have hr := e_1626_7
  simp only [e_1626_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1626 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16029)) := by
  have c99576 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99577 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99578 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99579 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99580 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99581 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1626_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_8
  simp only [← c99576, ← c99577, ← c99578] at e_1626_8
  have e_1626_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_9
  simp only [← c99579, ← c99580, k2, ← c99581] at e_1626_9
  have hr := e_1626_9
  simp only [e_1626_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1626 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16030)) := by
  have c99582 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99583 := (publicBatchWrapper89_copies3111 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99584 := (publicBatchWrapper89_copies3112 a h).1
  have c99585 := (publicBatchWrapper89_copies3112 a h).2.1
  have c99586 := (publicBatchWrapper89_copies3112 a h).2.2.1
  have c99587 := (publicBatchWrapper89_copies3112 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1626_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_10
  simp only [← c99582, ← c99583, ← c99584] at e_1626_10
  have e_1626_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_11
  simp only [← c99585, ← c99586, k2, ← c99587] at e_1626_11
  have hr := e_1626_11
  simp only [e_1626_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1626 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16031)) := by
  have c99588 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.1
  have c99589 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.1
  have c99590 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.1
  have c99591 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.1
  have c99592 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.1
  have c99593 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1626_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_12
  simp only [← c99588, ← c99589, ← c99590] at e_1626_12
  have e_1626_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_13
  simp only [← c99591, ← c99592, k2, ← c99593] at e_1626_13
  have hr := e_1626_13
  simp only [e_1626_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1626 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16032)) := by
  have c99594 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99595 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99596 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99597 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99598 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99599 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1626_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_14
  simp only [← c99594, ← c99595, ← c99596] at e_1626_14
  have e_1626_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_15
  simp only [← c99597, ← c99598, k2, ← c99599] at e_1626_15
  have hr := e_1626_15
  simp only [e_1626_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1626 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16033)) := by
  have c99600 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99601 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99602 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99603 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99604 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99605 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1626_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_16
  simp only [← c99600, ← c99601, ← c99602] at e_1626_16
  have e_1626_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_17
  simp only [← c99603, ← c99604, k2, ← c99605] at e_1626_17
  have hr := e_1626_17
  simp only [e_1626_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1626 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16034)) := by
  have c99606 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99607 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99608 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99609 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99610 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99611 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1626_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_18
  simp only [← c99606, ← c99607, ← c99608] at e_1626_18
  have e_1626_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1626 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1626_19
  simp only [← c99609, ← c99610, k2, ← c99611] at e_1626_19
  have hr := e_1626_19
  simp only [e_1626_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1627 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16035)) := by
  have c99612 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99613 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99614 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99615 := (publicBatchWrapper89_copies3112 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99616 := (publicBatchWrapper89_copies3113 a h).1
  have c99617 := (publicBatchWrapper89_copies3113 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1627_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_0
  simp only [← c99612, ← c99613, ← c99614] at e_1627_0
  have e_1627_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_1
  simp only [← c99615, ← c99616, k2, ← c99617] at e_1627_1
  have hr := e_1627_1
  simp only [e_1627_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1627 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16036)) := by
  have c99618 := (publicBatchWrapper89_copies3113 a h).2.2.1
  have c99619 := (publicBatchWrapper89_copies3113 a h).2.2.2.1
  have c99620 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.1
  have c99621 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.1
  have c99622 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.1
  have c99623 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1627_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_2
  simp only [← c99618, ← c99619, ← c99620] at e_1627_2
  have e_1627_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_3
  simp only [← c99621, ← c99622, k2, ← c99623] at e_1627_3
  have hr := e_1627_3
  simp only [e_1627_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1627 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16037)) := by
  have c99624 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.1
  have c99625 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.1
  have c99626 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99627 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99628 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99629 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1627_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_4
  simp only [← c99624, ← c99625, ← c99626] at e_1627_4
  have e_1627_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_5
  simp only [← c99627, ← c99628, k2, ← c99629] at e_1627_5
  have hr := e_1627_5
  simp only [e_1627_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1627 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16038)) := by
  have c99630 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99631 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99632 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99633 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99634 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99635 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1627_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_6
  simp only [← c99630, ← c99631, ← c99632] at e_1627_6
  have e_1627_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_7
  simp only [← c99633, ← c99634, k2, ← c99635] at e_1627_7
  have hr := e_1627_7
  simp only [e_1627_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1627 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16039)) := by
  have c99636 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99637 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99638 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99639 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99640 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99641 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1627_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_8
  simp only [← c99636, ← c99637, ← c99638] at e_1627_8
  have e_1627_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_9
  simp only [← c99639, ← c99640, k2, ← c99641] at e_1627_9
  have hr := e_1627_9
  simp only [e_1627_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1627 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16040)) := by
  have c99642 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99643 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99644 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99645 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99646 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99647 := (publicBatchWrapper89_copies3113 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1627_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_10
  simp only [← c99642, ← c99643, ← c99644] at e_1627_10
  have e_1627_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_11
  simp only [← c99645, ← c99646, k2, ← c99647] at e_1627_11
  have hr := e_1627_11
  simp only [e_1627_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1627 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16041)) := by
  have c99648 := (publicBatchWrapper89_copies3114 a h).1
  have c99649 := (publicBatchWrapper89_copies3114 a h).2.1
  have c99650 := (publicBatchWrapper89_copies3114 a h).2.2.1
  have c99651 := (publicBatchWrapper89_copies3114 a h).2.2.2.1
  have c99652 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.1
  have c99653 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1627_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_12
  simp only [← c99648, ← c99649, ← c99650] at e_1627_12
  have e_1627_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_13
  simp only [← c99651, ← c99652, k2, ← c99653] at e_1627_13
  have hr := e_1627_13
  simp only [e_1627_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1627 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16042)) := by
  have c99654 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.1
  have c99655 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.1
  have c99656 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.1
  have c99657 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.1
  have c99658 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99659 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1627_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_14
  simp only [← c99654, ← c99655, ← c99656] at e_1627_14
  have e_1627_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_15
  simp only [← c99657, ← c99658, k2, ← c99659] at e_1627_15
  have hr := e_1627_15
  simp only [e_1627_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1627 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16043)) := by
  have c99660 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99661 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99662 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99663 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99664 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99665 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1627_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_16
  simp only [← c99660, ← c99661, ← c99662] at e_1627_16
  have e_1627_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_17
  simp only [← c99663, ← c99664, k2, ← c99665] at e_1627_17
  have hr := e_1627_17
  simp only [e_1627_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1627 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16044)) := by
  have c99666 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99667 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99668 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99669 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99670 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99671 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1627_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_18
  simp only [← c99666, ← c99667, ← c99668] at e_1627_18
  have e_1627_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1627 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1627_19
  simp only [← c99669, ← c99670, k2, ← c99671] at e_1627_19
  have hr := e_1627_19
  simp only [e_1627_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1628 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16045)) := by
  have c99672 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99673 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99674 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99675 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99676 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99677 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1628_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_0
  simp only [← c99672, ← c99673, ← c99674] at e_1628_0
  have e_1628_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_1
  simp only [← c99675, ← c99676, k2, ← c99677] at e_1628_1
  have hr := e_1628_1
  simp only [e_1628_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1628 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16046)) := by
  have c99678 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99679 := (publicBatchWrapper89_copies3114 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99680 := (publicBatchWrapper89_copies3115 a h).1
  have c99681 := (publicBatchWrapper89_copies3115 a h).2.1
  have c99682 := (publicBatchWrapper89_copies3115 a h).2.2.1
  have c99683 := (publicBatchWrapper89_copies3115 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1628_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_2
  simp only [← c99678, ← c99679, ← c99680] at e_1628_2
  have e_1628_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_3
  simp only [← c99681, ← c99682, k2, ← c99683] at e_1628_3
  have hr := e_1628_3
  simp only [e_1628_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1628 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16047)) := by
  have c99684 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.1
  have c99685 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.1
  have c99686 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.1
  have c99687 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.1
  have c99688 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.1
  have c99689 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1628_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_4
  simp only [← c99684, ← c99685, ← c99686] at e_1628_4
  have e_1628_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_5
  simp only [← c99687, ← c99688, k2, ← c99689] at e_1628_5
  have hr := e_1628_5
  simp only [e_1628_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1628 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16048)) := by
  have c99690 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99691 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99692 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99693 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99694 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99695 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1628_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_6
  simp only [← c99690, ← c99691, ← c99692] at e_1628_6
  have e_1628_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_7
  simp only [← c99693, ← c99694, k2, ← c99695] at e_1628_7
  have hr := e_1628_7
  simp only [e_1628_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1628 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16049)) := by
  have c99696 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99697 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99698 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99699 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99700 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99701 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1628_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_8
  simp only [← c99696, ← c99697, ← c99698] at e_1628_8
  have e_1628_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_9
  simp only [← c99699, ← c99700, k2, ← c99701] at e_1628_9
  have hr := e_1628_9
  simp only [e_1628_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1628 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16050)) := by
  have c99702 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99703 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99704 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99705 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99706 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99707 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1628_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_10
  simp only [← c99702, ← c99703, ← c99704] at e_1628_10
  have e_1628_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_11
  simp only [← c99705, ← c99706, k2, ← c99707] at e_1628_11
  have hr := e_1628_11
  simp only [e_1628_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1628 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16051)) := by
  have c99708 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99709 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99710 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99711 := (publicBatchWrapper89_copies3115 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99712 := (publicBatchWrapper89_copies3116 a h).1
  have c99713 := (publicBatchWrapper89_copies3116 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1628_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_12
  simp only [← c99708, ← c99709, ← c99710] at e_1628_12
  have e_1628_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_13
  simp only [← c99711, ← c99712, k2, ← c99713] at e_1628_13
  have hr := e_1628_13
  simp only [e_1628_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1628 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16052)) := by
  have c99714 := (publicBatchWrapper89_copies3116 a h).2.2.1
  have c99715 := (publicBatchWrapper89_copies3116 a h).2.2.2.1
  have c99716 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.1
  have c99717 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.1
  have c99718 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.1
  have c99719 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1628_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_14
  simp only [← c99714, ← c99715, ← c99716] at e_1628_14
  have e_1628_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_15
  simp only [← c99717, ← c99718, k2, ← c99719] at e_1628_15
  have hr := e_1628_15
  simp only [e_1628_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1628 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16053)) := by
  have c99720 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.1
  have c99721 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.1
  have c99722 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99723 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99724 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99725 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1628_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_16
  simp only [← c99720, ← c99721, ← c99722] at e_1628_16
  have e_1628_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_17
  simp only [← c99723, ← c99724, k2, ← c99725] at e_1628_17
  have hr := e_1628_17
  simp only [e_1628_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1628 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16054)) := by
  have c99726 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99727 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99728 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99729 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99730 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99731 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1628_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_18
  simp only [← c99726, ← c99727, ← c99728] at e_1628_18
  have e_1628_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1628 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1628_19
  simp only [← c99729, ← c99730, k2, ← c99731] at e_1628_19
  have hr := e_1628_19
  simp only [e_1628_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1629 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16055)) := by
  have c99732 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99733 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99734 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99735 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99736 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99737 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1629_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_0
  simp only [← c99732, ← c99733, ← c99734] at e_1629_0
  have e_1629_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_1
  simp only [← c99735, ← c99736, k2, ← c99737] at e_1629_1
  have hr := e_1629_1
  simp only [e_1629_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1629 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16056)) := by
  have c99738 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99739 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99740 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99741 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99742 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99743 := (publicBatchWrapper89_copies3116 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1629_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_2
  simp only [← c99738, ← c99739, ← c99740] at e_1629_2
  have e_1629_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_3
  simp only [← c99741, ← c99742, k2, ← c99743] at e_1629_3
  have hr := e_1629_3
  simp only [e_1629_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1629 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16057)) := by
  have c99744 := (publicBatchWrapper89_copies3117 a h).1
  have c99745 := (publicBatchWrapper89_copies3117 a h).2.1
  have c99746 := (publicBatchWrapper89_copies3117 a h).2.2.1
  have c99747 := (publicBatchWrapper89_copies3117 a h).2.2.2.1
  have c99748 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.1
  have c99749 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1629_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_4
  simp only [← c99744, ← c99745, ← c99746] at e_1629_4
  have e_1629_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_5
  simp only [← c99747, ← c99748, k2, ← c99749] at e_1629_5
  have hr := e_1629_5
  simp only [e_1629_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1629 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16058)) := by
  have c99750 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.1
  have c99751 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.1
  have c99752 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.1
  have c99753 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.1
  have c99754 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99755 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1629_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_6
  simp only [← c99750, ← c99751, ← c99752] at e_1629_6
  have e_1629_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_7
  simp only [← c99753, ← c99754, k2, ← c99755] at e_1629_7
  have hr := e_1629_7
  simp only [e_1629_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1629 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16059)) := by
  have c99756 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99757 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99758 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99759 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99760 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99761 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1629_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_8
  simp only [← c99756, ← c99757, ← c99758] at e_1629_8
  have e_1629_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_9
  simp only [← c99759, ← c99760, k2, ← c99761] at e_1629_9
  have hr := e_1629_9
  simp only [e_1629_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1629 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16060)) := by
  have c99762 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99763 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99764 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99765 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99766 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99767 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1629_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_10
  simp only [← c99762, ← c99763, ← c99764] at e_1629_10
  have e_1629_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_11
  simp only [← c99765, ← c99766, k2, ← c99767] at e_1629_11
  have hr := e_1629_11
  simp only [e_1629_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1629 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16061)) := by
  have c99768 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99769 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99770 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99771 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99772 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99773 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1629_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_12
  simp only [← c99768, ← c99769, ← c99770] at e_1629_12
  have e_1629_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_13
  simp only [← c99771, ← c99772, k2, ← c99773] at e_1629_13
  have hr := e_1629_13
  simp only [e_1629_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1629 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16062)) := by
  have c99774 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99775 := (publicBatchWrapper89_copies3117 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99776 := (publicBatchWrapper89_copies3118 a h).1
  have c99777 := (publicBatchWrapper89_copies3118 a h).2.1
  have c99778 := (publicBatchWrapper89_copies3118 a h).2.2.1
  have c99779 := (publicBatchWrapper89_copies3118 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1629_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_14
  simp only [← c99774, ← c99775, ← c99776] at e_1629_14
  have e_1629_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_15
  simp only [← c99777, ← c99778, k2, ← c99779] at e_1629_15
  have hr := e_1629_15
  simp only [e_1629_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1629 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16063)) := by
  have c99780 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.1
  have c99781 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.1
  have c99782 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.1
  have c99783 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.1
  have c99784 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.1
  have c99785 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1629_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_16
  simp only [← c99780, ← c99781, ← c99782] at e_1629_16
  have e_1629_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_17
  simp only [← c99783, ← c99784, k2, ← c99785] at e_1629_17
  have hr := e_1629_17
  simp only [e_1629_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1629 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16064)) := by
  have c99786 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99787 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99788 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99789 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99790 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99791 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1629_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_18
  simp only [← c99786, ← c99787, ← c99788] at e_1629_18
  have e_1629_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1629 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1629_19
  simp only [← c99789, ← c99790, k2, ← c99791] at e_1629_19
  have hr := e_1629_19
  simp only [e_1629_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1630 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16065)) := by
  have c99792 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99793 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99794 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99795 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99796 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99797 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1630_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_0
  simp only [← c99792, ← c99793, ← c99794] at e_1630_0
  have e_1630_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_1
  simp only [← c99795, ← c99796, k2, ← c99797] at e_1630_1
  have hr := e_1630_1
  simp only [e_1630_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1630 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16066)) := by
  have c99798 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99799 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99800 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99801 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99802 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99803 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1630_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_2
  simp only [← c99798, ← c99799, ← c99800] at e_1630_2
  have e_1630_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_3
  simp only [← c99801, ← c99802, k2, ← c99803] at e_1630_3
  have hr := e_1630_3
  simp only [e_1630_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1630 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16067)) := by
  have c99804 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99805 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99806 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99807 := (publicBatchWrapper89_copies3118 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99808 := (publicBatchWrapper89_copies3119 a h).1
  have c99809 := (publicBatchWrapper89_copies3119 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1630_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_4
  simp only [← c99804, ← c99805, ← c99806] at e_1630_4
  have e_1630_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_5
  simp only [← c99807, ← c99808, k2, ← c99809] at e_1630_5
  have hr := e_1630_5
  simp only [e_1630_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1630 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16068)) := by
  have c99810 := (publicBatchWrapper89_copies3119 a h).2.2.1
  have c99811 := (publicBatchWrapper89_copies3119 a h).2.2.2.1
  have c99812 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.1
  have c99813 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.1
  have c99814 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.1
  have c99815 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1630_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_6
  simp only [← c99810, ← c99811, ← c99812] at e_1630_6
  have e_1630_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_7
  simp only [← c99813, ← c99814, k2, ← c99815] at e_1630_7
  have hr := e_1630_7
  simp only [e_1630_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1630 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16069)) := by
  have c99816 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.1
  have c99817 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.1
  have c99818 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99819 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99820 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99821 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1630_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_8
  simp only [← c99816, ← c99817, ← c99818] at e_1630_8
  have e_1630_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_9
  simp only [← c99819, ← c99820, k2, ← c99821] at e_1630_9
  have hr := e_1630_9
  simp only [e_1630_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1630 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16070)) := by
  have c99822 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99823 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99824 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99825 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99826 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99827 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1630_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_10
  simp only [← c99822, ← c99823, ← c99824] at e_1630_10
  have e_1630_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_11
  simp only [← c99825, ← c99826, k2, ← c99827] at e_1630_11
  have hr := e_1630_11
  simp only [e_1630_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1630 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16071)) := by
  have c99828 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99829 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99830 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99831 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99832 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99833 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1630_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_12
  simp only [← c99828, ← c99829, ← c99830] at e_1630_12
  have e_1630_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_13
  simp only [← c99831, ← c99832, k2, ← c99833] at e_1630_13
  have hr := e_1630_13
  simp only [e_1630_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1630 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16072)) := by
  have c99834 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99835 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99836 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99837 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99838 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99839 := (publicBatchWrapper89_copies3119 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1630_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_14
  simp only [← c99834, ← c99835, ← c99836] at e_1630_14
  have e_1630_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_15
  simp only [← c99837, ← c99838, k2, ← c99839] at e_1630_15
  have hr := e_1630_15
  simp only [e_1630_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1630 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16073)) := by
  have c99840 := (publicBatchWrapper89_copies3120 a h).1
  have c99841 := (publicBatchWrapper89_copies3120 a h).2.1
  have c99842 := (publicBatchWrapper89_copies3120 a h).2.2.1
  have c99843 := (publicBatchWrapper89_copies3120 a h).2.2.2.1
  have c99844 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.1
  have c99845 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1630_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_16
  simp only [← c99840, ← c99841, ← c99842] at e_1630_16
  have e_1630_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_17
  simp only [← c99843, ← c99844, k2, ← c99845] at e_1630_17
  have hr := e_1630_17
  simp only [e_1630_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1630 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16074)) := by
  have c99846 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.1
  have c99847 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.1
  have c99848 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.1
  have c99849 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.1
  have c99850 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99851 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1630_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_18
  simp only [← c99846, ← c99847, ← c99848] at e_1630_18
  have e_1630_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1630 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1630_19
  simp only [← c99849, ← c99850, k2, ← c99851] at e_1630_19
  have hr := e_1630_19
  simp only [e_1630_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1631 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16075)) := by
  have c99852 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99853 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99854 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99855 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99856 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99857 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1631_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_0
  simp only [← c99852, ← c99853, ← c99854] at e_1631_0
  have e_1631_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_1
  simp only [← c99855, ← c99856, k2, ← c99857] at e_1631_1
  have hr := e_1631_1
  simp only [e_1631_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1631 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16076)) := by
  have c99858 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99859 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99860 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99861 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99862 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99863 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1631_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_2
  simp only [← c99858, ← c99859, ← c99860] at e_1631_2
  have e_1631_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_3
  simp only [← c99861, ← c99862, k2, ← c99863] at e_1631_3
  have hr := e_1631_3
  simp only [e_1631_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1631 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16077)) := by
  have c99864 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99865 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99866 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99867 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99868 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99869 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1631_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_4
  simp only [← c99864, ← c99865, ← c99866] at e_1631_4
  have e_1631_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_5
  simp only [← c99867, ← c99868, k2, ← c99869] at e_1631_5
  have hr := e_1631_5
  simp only [e_1631_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1631 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16078)) := by
  have c99870 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99871 := (publicBatchWrapper89_copies3120 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99872 := (publicBatchWrapper89_copies3121 a h).1
  have c99873 := (publicBatchWrapper89_copies3121 a h).2.1
  have c99874 := (publicBatchWrapper89_copies3121 a h).2.2.1
  have c99875 := (publicBatchWrapper89_copies3121 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1631_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_6
  simp only [← c99870, ← c99871, ← c99872] at e_1631_6
  have e_1631_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_7
  simp only [← c99873, ← c99874, k2, ← c99875] at e_1631_7
  have hr := e_1631_7
  simp only [e_1631_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1631 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16079)) := by
  have c99876 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.1
  have c99877 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.1
  have c99878 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.1
  have c99879 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.1
  have c99880 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.1
  have c99881 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1631_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_8
  simp only [← c99876, ← c99877, ← c99878] at e_1631_8
  have e_1631_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_9
  simp only [← c99879, ← c99880, k2, ← c99881] at e_1631_9
  have hr := e_1631_9
  simp only [e_1631_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1631 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16080)) := by
  have c99882 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99883 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99884 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99885 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99886 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99887 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1631_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_10
  simp only [← c99882, ← c99883, ← c99884] at e_1631_10
  have e_1631_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_11
  simp only [← c99885, ← c99886, k2, ← c99887] at e_1631_11
  have hr := e_1631_11
  simp only [e_1631_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1631 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16081)) := by
  have c99888 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99889 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99890 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99891 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99892 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99893 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1631_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_12
  simp only [← c99888, ← c99889, ← c99890] at e_1631_12
  have e_1631_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_13
  simp only [← c99891, ← c99892, k2, ← c99893] at e_1631_13
  have hr := e_1631_13
  simp only [e_1631_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1631 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16082)) := by
  have c99894 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99895 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99896 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99897 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99898 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99899 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1631_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_14
  simp only [← c99894, ← c99895, ← c99896] at e_1631_14
  have e_1631_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_15
  simp only [← c99897, ← c99898, k2, ← c99899] at e_1631_15
  have hr := e_1631_15
  simp only [e_1631_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1631 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16083)) := by
  have c99900 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99901 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99902 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99903 := (publicBatchWrapper89_copies3121 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99904 := (publicBatchWrapper89_copies3122 a h).1
  have c99905 := (publicBatchWrapper89_copies3122 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1631_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_16
  simp only [← c99900, ← c99901, ← c99902] at e_1631_16
  have e_1631_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_17
  simp only [← c99903, ← c99904, k2, ← c99905] at e_1631_17
  have hr := e_1631_17
  simp only [e_1631_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1631 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16084)) := by
  have c99906 := (publicBatchWrapper89_copies3122 a h).2.2.1
  have c99907 := (publicBatchWrapper89_copies3122 a h).2.2.2.1
  have c99908 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.1
  have c99909 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.1
  have c99910 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.1
  have c99911 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1631_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_18
  simp only [← c99906, ← c99907, ← c99908] at e_1631_18
  have e_1631_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1631 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1631_19
  simp only [← c99909, ← c99910, k2, ← c99911] at e_1631_19
  have hr := e_1631_19
  simp only [e_1631_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1632 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16085)) := by
  have c99912 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.1
  have c99913 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.1
  have c99914 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99915 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99916 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99917 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1632_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_0
  simp only [← c99912, ← c99913, ← c99914] at e_1632_0
  have e_1632_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_1
  simp only [← c99915, ← c99916, k2, ← c99917] at e_1632_1
  have hr := e_1632_1
  simp only [e_1632_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1632 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16086)) := by
  have c99918 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99919 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99920 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99921 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99922 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99923 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1632_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_2
  simp only [← c99918, ← c99919, ← c99920] at e_1632_2
  have e_1632_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_3
  simp only [← c99921, ← c99922, k2, ← c99923] at e_1632_3
  have hr := e_1632_3
  simp only [e_1632_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1632 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16087)) := by
  have c99924 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99925 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99926 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99927 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99928 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99929 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1632_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_4
  simp only [← c99924, ← c99925, ← c99926] at e_1632_4
  have e_1632_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_5
  simp only [← c99927, ← c99928, k2, ← c99929] at e_1632_5
  have hr := e_1632_5
  simp only [e_1632_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1632 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16088)) := by
  have c99930 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99931 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99932 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99933 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99934 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99935 := (publicBatchWrapper89_copies3122 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1632_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_6
  simp only [← c99930, ← c99931, ← c99932] at e_1632_6
  have e_1632_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_7
  simp only [← c99933, ← c99934, k2, ← c99935] at e_1632_7
  have hr := e_1632_7
  simp only [e_1632_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1632 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16089)) := by
  have c99936 := (publicBatchWrapper89_copies3123 a h).1
  have c99937 := (publicBatchWrapper89_copies3123 a h).2.1
  have c99938 := (publicBatchWrapper89_copies3123 a h).2.2.1
  have c99939 := (publicBatchWrapper89_copies3123 a h).2.2.2.1
  have c99940 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.1
  have c99941 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1632_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_8
  simp only [← c99936, ← c99937, ← c99938] at e_1632_8
  have e_1632_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_9
  simp only [← c99939, ← c99940, k2, ← c99941] at e_1632_9
  have hr := e_1632_9
  simp only [e_1632_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1632 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16090)) := by
  have c99942 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.1
  have c99943 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.1
  have c99944 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.1
  have c99945 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.1
  have c99946 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99947 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1632_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_10
  simp only [← c99942, ← c99943, ← c99944] at e_1632_10
  have e_1632_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_11
  simp only [← c99945, ← c99946, k2, ← c99947] at e_1632_11
  have hr := e_1632_11
  simp only [e_1632_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1632 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16091)) := by
  have c99948 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99949 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99950 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99951 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99952 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99953 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1632_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_12
  simp only [← c99948, ← c99949, ← c99950] at e_1632_12
  have e_1632_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_13
  simp only [← c99951, ← c99952, k2, ← c99953] at e_1632_13
  have hr := e_1632_13
  simp only [e_1632_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1632 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16092)) := by
  have c99954 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99955 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99956 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99957 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99958 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99959 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1632_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_14
  simp only [← c99954, ← c99955, ← c99956] at e_1632_14
  have e_1632_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_15
  simp only [← c99957, ← c99958, k2, ← c99959] at e_1632_15
  have hr := e_1632_15
  simp only [e_1632_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1632 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16093)) := by
  have c99960 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99961 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99962 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99963 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99964 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99965 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1632_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_16
  simp only [← c99960, ← c99961, ← c99962] at e_1632_16
  have e_1632_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_17
  simp only [← c99963, ← c99964, k2, ← c99965] at e_1632_17
  have hr := e_1632_17
  simp only [e_1632_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1632 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16094)) := by
  have c99966 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99967 := (publicBatchWrapper89_copies3123 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c99968 := (publicBatchWrapper89_copies3124 a h).1
  have c99969 := (publicBatchWrapper89_copies3124 a h).2.1
  have c99970 := (publicBatchWrapper89_copies3124 a h).2.2.1
  have c99971 := (publicBatchWrapper89_copies3124 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1632_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_18
  simp only [← c99966, ← c99967, ← c99968] at e_1632_18
  have e_1632_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1632 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1632_19
  simp only [← c99969, ← c99970, k2, ← c99971] at e_1632_19
  have hr := e_1632_19
  simp only [e_1632_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1633 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16095)) := by
  have c99972 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.1
  have c99973 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.1
  have c99974 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.1
  have c99975 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.1
  have c99976 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.1
  have c99977 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1633_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_0
  simp only [← c99972, ← c99973, ← c99974] at e_1633_0
  have e_1633_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_1
  simp only [← c99975, ← c99976, k2, ← c99977] at e_1633_1
  have hr := e_1633_1
  simp only [e_1633_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1633 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16096)) := by
  have c99978 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.1
  have c99979 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c99980 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99981 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99982 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99983 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1633_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_2
  simp only [← c99978, ← c99979, ← c99980] at e_1633_2
  have e_1633_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_3
  simp only [← c99981, ← c99982, k2, ← c99983] at e_1633_3
  have hr := e_1633_3
  simp only [e_1633_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1633 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16097)) := by
  have c99984 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99985 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99986 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99987 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99988 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99989 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1633_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_4
  simp only [← c99984, ← c99985, ← c99986] at e_1633_4
  have e_1633_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_5
  simp only [← c99987, ← c99988, k2, ← c99989] at e_1633_5
  have hr := e_1633_5
  simp only [e_1633_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1633 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16098)) := by
  have c99990 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99991 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99992 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99993 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99994 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99995 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1633_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_6
  simp only [← c99990, ← c99991, ← c99992] at e_1633_6
  have e_1633_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_7
  simp only [← c99993, ← c99994, k2, ← c99995] at e_1633_7
  have hr := e_1633_7
  simp only [e_1633_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1633 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16099)) := by
  have c99996 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99997 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99998 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c99999 := (publicBatchWrapper89_copies3124 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100000 := (publicBatchWrapper89_copies3125 a h).1
  have c100001 := (publicBatchWrapper89_copies3125 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1633_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_8
  simp only [← c99996, ← c99997, ← c99998] at e_1633_8
  have e_1633_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_9
  simp only [← c99999, ← c100000, k2, ← c100001] at e_1633_9
  have hr := e_1633_9
  simp only [e_1633_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1633 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16100)) := by
  have c100002 := (publicBatchWrapper89_copies3125 a h).2.2.1
  have c100003 := (publicBatchWrapper89_copies3125 a h).2.2.2.1
  have c100004 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.1
  have c100005 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.1
  have c100006 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.1
  have c100007 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1633_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_10
  simp only [← c100002, ← c100003, ← c100004] at e_1633_10
  have e_1633_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_11
  simp only [← c100005, ← c100006, k2, ← c100007] at e_1633_11
  have hr := e_1633_11
  simp only [e_1633_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1633 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16101)) := by
  have c100008 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.1
  have c100009 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.1
  have c100010 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100011 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100012 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100013 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1633_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_12
  simp only [← c100008, ← c100009, ← c100010] at e_1633_12
  have e_1633_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_13
  simp only [← c100011, ← c100012, k2, ← c100013] at e_1633_13
  have hr := e_1633_13
  simp only [e_1633_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1633 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16102)) := by
  have c100014 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100015 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100016 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100017 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100018 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100019 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1633_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_14
  simp only [← c100014, ← c100015, ← c100016] at e_1633_14
  have e_1633_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_15
  simp only [← c100017, ← c100018, k2, ← c100019] at e_1633_15
  have hr := e_1633_15
  simp only [e_1633_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1633 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16103)) := by
  have c100020 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100021 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100022 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100023 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100024 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100025 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1633_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_16
  simp only [← c100020, ← c100021, ← c100022] at e_1633_16
  have e_1633_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_17
  simp only [← c100023, ← c100024, k2, ← c100025] at e_1633_17
  have hr := e_1633_17
  simp only [e_1633_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1633 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16104)) := by
  have c100026 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100027 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100028 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100029 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100030 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100031 := (publicBatchWrapper89_copies3125 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1633_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_18
  simp only [← c100026, ← c100027, ← c100028] at e_1633_18
  have e_1633_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1633 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1633_19
  simp only [← c100029, ← c100030, k2, ← c100031] at e_1633_19
  have hr := e_1633_19
  simp only [e_1633_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1634 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16105)) := by
  have c100032 := (publicBatchWrapper89_copies3126 a h).1
  have c100033 := (publicBatchWrapper89_copies3126 a h).2.1
  have c100034 := (publicBatchWrapper89_copies3126 a h).2.2.1
  have c100035 := (publicBatchWrapper89_copies3126 a h).2.2.2.1
  have c100036 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.1
  have c100037 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1634_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_0
  simp only [← c100032, ← c100033, ← c100034] at e_1634_0
  have e_1634_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_1
  simp only [← c100035, ← c100036, k2, ← c100037] at e_1634_1
  have hr := e_1634_1
  simp only [e_1634_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1634 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16106)) := by
  have c100038 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.1
  have c100039 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.1
  have c100040 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.1
  have c100041 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.1
  have c100042 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100043 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1634_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_2
  simp only [← c100038, ← c100039, ← c100040] at e_1634_2
  have e_1634_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_3
  simp only [← c100041, ← c100042, k2, ← c100043] at e_1634_3
  have hr := e_1634_3
  simp only [e_1634_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1634 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16107)) := by
  have c100044 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100045 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100046 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100047 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100048 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100049 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1634_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_4
  simp only [← c100044, ← c100045, ← c100046] at e_1634_4
  have e_1634_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_5
  simp only [← c100047, ← c100048, k2, ← c100049] at e_1634_5
  have hr := e_1634_5
  simp only [e_1634_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1634 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16108)) := by
  have c100050 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100051 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100052 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100053 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100054 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100055 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1634_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_6
  simp only [← c100050, ← c100051, ← c100052] at e_1634_6
  have e_1634_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_7
  simp only [← c100053, ← c100054, k2, ← c100055] at e_1634_7
  have hr := e_1634_7
  simp only [e_1634_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1634 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16109)) := by
  have c100056 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100057 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100058 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100059 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100060 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100061 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1634_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_8
  simp only [← c100056, ← c100057, ← c100058] at e_1634_8
  have e_1634_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_9
  simp only [← c100059, ← c100060, k2, ← c100061] at e_1634_9
  have hr := e_1634_9
  simp only [e_1634_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1634 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16110)) := by
  have c100062 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100063 := (publicBatchWrapper89_copies3126 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100064 := (publicBatchWrapper89_copies3127 a h).1
  have c100065 := (publicBatchWrapper89_copies3127 a h).2.1
  have c100066 := (publicBatchWrapper89_copies3127 a h).2.2.1
  have c100067 := (publicBatchWrapper89_copies3127 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1634_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_10
  simp only [← c100062, ← c100063, ← c100064] at e_1634_10
  have e_1634_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_11
  simp only [← c100065, ← c100066, k2, ← c100067] at e_1634_11
  have hr := e_1634_11
  simp only [e_1634_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1634 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16111)) := by
  have c100068 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.1
  have c100069 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.1
  have c100070 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.1
  have c100071 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.1
  have c100072 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.1
  have c100073 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1634_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_12
  simp only [← c100068, ← c100069, ← c100070] at e_1634_12
  have e_1634_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_13
  simp only [← c100071, ← c100072, k2, ← c100073] at e_1634_13
  have hr := e_1634_13
  simp only [e_1634_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1634 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16112)) := by
  have c100074 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100075 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100076 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100077 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100078 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100079 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1634_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_14
  simp only [← c100074, ← c100075, ← c100076] at e_1634_14
  have e_1634_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_15
  simp only [← c100077, ← c100078, k2, ← c100079] at e_1634_15
  have hr := e_1634_15
  simp only [e_1634_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1634 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16113)) := by
  have c100080 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100081 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100082 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100083 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100084 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100085 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1634_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_16
  simp only [← c100080, ← c100081, ← c100082] at e_1634_16
  have e_1634_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_17
  simp only [← c100083, ← c100084, k2, ← c100085] at e_1634_17
  have hr := e_1634_17
  simp only [e_1634_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1634 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16114)) := by
  have c100086 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100087 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100088 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100089 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100090 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100091 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1634_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_18
  simp only [← c100086, ← c100087, ← c100088] at e_1634_18
  have e_1634_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1634 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1634_19
  simp only [← c100089, ← c100090, k2, ← c100091] at e_1634_19
  have hr := e_1634_19
  simp only [e_1634_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1635 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16115)) := by
  have c100092 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100093 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100094 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100095 := (publicBatchWrapper89_copies3127 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c100096 := (publicBatchWrapper89_copies3128 a h).1
  have c100097 := (publicBatchWrapper89_copies3128 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1635_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_0
  simp only [← c100092, ← c100093, ← c100094] at e_1635_0
  have e_1635_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_1
  simp only [← c100095, ← c100096, k2, ← c100097] at e_1635_1
  have hr := e_1635_1
  simp only [e_1635_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1635 15) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16116)) := by
  have c100098 := (publicBatchWrapper89_copies3128 a h).2.2.1
  have c100099 := (publicBatchWrapper89_copies3128 a h).2.2.2.1
  have c100100 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.1
  have c100101 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.1
  have c100102 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.1
  have c100103 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1635_2 := arithEq_of_rows h (i := 2) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_2
  simp only [← c100098, ← c100099, ← c100100] at e_1635_2
  have e_1635_3 := arithEq_of_rows h (i := 3) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_3
  simp only [← c100101, ← c100102, k2, ← c100103] at e_1635_3
  have hr := e_1635_3
  simp only [e_1635_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1635 23) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16117)) := by
  have c100104 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.1
  have c100105 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.1
  have c100106 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100107 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c100108 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100109 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1635_4 := arithEq_of_rows h (i := 4) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_4
  simp only [← c100104, ← c100105, ← c100106] at e_1635_4
  have e_1635_5 := arithEq_of_rows h (i := 5) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_5
  simp only [← c100107, ← c100108, k2, ← c100109] at e_1635_5
  have hr := e_1635_5
  simp only [e_1635_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1635 31) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16118)) := by
  have c100110 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100111 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100112 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100113 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100114 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100115 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1635_6 := arithEq_of_rows h (i := 6) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_6
  simp only [← c100110, ← c100111, ← c100112] at e_1635_6
  have e_1635_7 := arithEq_of_rows h (i := 7) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_7
  simp only [← c100113, ← c100114, k2, ← c100115] at e_1635_7
  have hr := e_1635_7
  simp only [e_1635_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1635 39) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16119)) := by
  have c100116 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100117 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100118 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100119 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100120 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100121 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1635_8 := arithEq_of_rows h (i := 8) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_8
  simp only [← c100116, ← c100117, ← c100118] at e_1635_8
  have e_1635_9 := arithEq_of_rows h (i := 9) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_9
  simp only [← c100119, ← c100120, k2, ← c100121] at e_1635_9
  have hr := e_1635_9
  simp only [e_1635_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1635 47) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16120)) := by
  have c100122 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100123 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100124 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100125 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100126 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100127 := (publicBatchWrapper89_copies3128 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1635_10 := arithEq_of_rows h (i := 10) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_10
  simp only [← c100122, ← c100123, ← c100124] at e_1635_10
  have e_1635_11 := arithEq_of_rows h (i := 11) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_11
  simp only [← c100125, ← c100126, k2, ← c100127] at e_1635_11
  have hr := e_1635_11
  simp only [e_1635_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1635 55) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16121)) := by
  have c100128 := (publicBatchWrapper89_copies3129 a h).1
  have c100129 := (publicBatchWrapper89_copies3129 a h).2.1
  have c100130 := (publicBatchWrapper89_copies3129 a h).2.2.1
  have c100131 := (publicBatchWrapper89_copies3129 a h).2.2.2.1
  have c100132 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.1
  have c100133 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1635_12 := arithEq_of_rows h (i := 12) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_12
  simp only [← c100128, ← c100129, ← c100130] at e_1635_12
  have e_1635_13 := arithEq_of_rows h (i := 13) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_13
  simp only [← c100131, ← c100132, k2, ← c100133] at e_1635_13
  have hr := e_1635_13
  simp only [e_1635_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1635 63) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16122)) := by
  have c100134 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.1
  have c100135 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.1
  have c100136 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.1
  have c100137 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.1
  have c100138 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.1
  have c100139 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1635_14 := arithEq_of_rows h (i := 14) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_14
  simp only [← c100134, ← c100135, ← c100136] at e_1635_14
  have e_1635_15 := arithEq_of_rows h (i := 15) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_15
  simp only [← c100137, ← c100138, k2, ← c100139] at e_1635_15
  have hr := e_1635_15
  simp only [e_1635_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1635 71) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16123)) := by
  have c100140 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100141 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100142 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100143 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100144 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100145 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1635_16 := arithEq_of_rows h (i := 16) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_16
  simp only [← c100140, ← c100141, ← c100142] at e_1635_16
  have e_1635_17 := arithEq_of_rows h (i := 17) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_17
  simp only [← c100143, ← c100144, k2, ← c100145] at e_1635_17
  have hr := e_1635_17
  simp only [e_1635_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1635 79) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16124)) := by
  have c100146 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100147 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100148 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100149 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100150 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100151 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1635_18 := arithEq_of_rows h (i := 18) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_18
  simp only [← c100146, ← c100147, ← c100148] at e_1635_18
  have e_1635_19 := arithEq_of_rows h (i := 19) publicBatchWrapper89_row1635 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1635_19
  simp only [← c100149, ← c100150, k2, ← c100151] at e_1635_19
  have hr := e_1635_19
  simp only [e_1635_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f15941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1636 7) = bselect (a (.wire 77 59)) (a (.virt 200)) (a (.virt 16125)) := by
  have c100152 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100153 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100154 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100155 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100156 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c100157 := (publicBatchWrapper89_copies3129 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1636_0 := arithEq_of_rows h (i := 0) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_0
  simp only [← c100152, ← c100153, ← c100154] at e_1636_0
  have e_1636_1 := arithEq_of_rows h (i := 1) publicBatchWrapper89_row1636 (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1636_1
  simp only [← c100155, ← c100156, k2, ← c100157] at e_1636_1
  have hr := e_1636_1
  simp only [e_1636_0] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
