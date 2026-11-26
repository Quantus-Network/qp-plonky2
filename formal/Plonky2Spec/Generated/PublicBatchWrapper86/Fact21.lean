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

theorem publicBatchWrapper86_f16228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1663 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16634)) := by
  have c101721 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101722 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101723 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101724 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101725 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101726 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1663_12 := arithEq_of_rows h (row := 1663) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_12
  simp only [← c101721, ← c101722, ← c101723] at e_1663_12
  have e_1663_13 := arithEq_of_rows h (row := 1663) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_13
  simp only [← c101724, ← c101725, k2, ← c101726] at e_1663_13
  have hr := e_1663_13
  simp only [e_1663_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1663 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16635)) := by
  have c101727 := (publicBatchWrapper86_copies3178 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101728 := (publicBatchWrapper86_copies3179 a h).1
  have c101729 := (publicBatchWrapper86_copies3179 a h).2.1
  have c101730 := (publicBatchWrapper86_copies3179 a h).2.2.1
  have c101731 := (publicBatchWrapper86_copies3179 a h).2.2.2.1
  have c101732 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1663_14 := arithEq_of_rows h (row := 1663) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_14
  simp only [← c101727, ← c101728, ← c101729] at e_1663_14
  have e_1663_15 := arithEq_of_rows h (row := 1663) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_15
  simp only [← c101730, ← c101731, k2, ← c101732] at e_1663_15
  have hr := e_1663_15
  simp only [e_1663_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1663 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16636)) := by
  have c101733 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.1
  have c101734 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.1
  have c101735 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.1
  have c101736 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.1
  have c101737 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.1
  have c101738 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1663_16 := arithEq_of_rows h (row := 1663) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_16
  simp only [← c101733, ← c101734, ← c101735] at e_1663_16
  have e_1663_17 := arithEq_of_rows h (row := 1663) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_17
  simp only [← c101736, ← c101737, k2, ← c101738] at e_1663_17
  have hr := e_1663_17
  simp only [e_1663_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1663 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16637)) := by
  have c101739 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101740 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101741 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101742 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101743 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101744 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1663_18 := arithEq_of_rows h (row := 1663) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_18
  simp only [← c101739, ← c101740, ← c101741] at e_1663_18
  have e_1663_19 := arithEq_of_rows h (row := 1663) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1663_19
  simp only [← c101742, ← c101743, k2, ← c101744] at e_1663_19
  have hr := e_1663_19
  simp only [e_1663_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16638)) := by
  have c101745 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101746 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101747 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101748 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101749 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101750 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_0 := arithEq_of_rows h (row := 1664) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_0
  simp only [← c101745, ← c101746, ← c101747] at e_1664_0
  have e_1664_1 := arithEq_of_rows h (row := 1664) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_1
  simp only [← c101748, ← c101749, k2, ← c101750] at e_1664_1
  have hr := e_1664_1
  simp only [e_1664_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16639)) := by
  have c101751 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101752 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101753 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101754 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101755 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101756 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_2 := arithEq_of_rows h (row := 1664) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_2
  simp only [← c101751, ← c101752, ← c101753] at e_1664_2
  have e_1664_3 := arithEq_of_rows h (row := 1664) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_3
  simp only [← c101754, ← c101755, k2, ← c101756] at e_1664_3
  have hr := e_1664_3
  simp only [e_1664_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16640)) := by
  have c101757 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101758 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101759 := (publicBatchWrapper86_copies3179 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101760 := (publicBatchWrapper86_copies3180 a h).1
  have c101761 := (publicBatchWrapper86_copies3180 a h).2.1
  have c101762 := (publicBatchWrapper86_copies3180 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_4 := arithEq_of_rows h (row := 1664) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_4
  simp only [← c101757, ← c101758, ← c101759] at e_1664_4
  have e_1664_5 := arithEq_of_rows h (row := 1664) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_5
  simp only [← c101760, ← c101761, k2, ← c101762] at e_1664_5
  have hr := e_1664_5
  simp only [e_1664_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16641)) := by
  have c101763 := (publicBatchWrapper86_copies3180 a h).2.2.2.1
  have c101764 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.1
  have c101765 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.1
  have c101766 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.1
  have c101767 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.1
  have c101768 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_6 := arithEq_of_rows h (row := 1664) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_6
  simp only [← c101763, ← c101764, ← c101765] at e_1664_6
  have e_1664_7 := arithEq_of_rows h (row := 1664) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_7
  simp only [← c101766, ← c101767, k2, ← c101768] at e_1664_7
  have hr := e_1664_7
  simp only [e_1664_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16642)) := by
  have c101769 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.1
  have c101770 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101771 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101772 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101773 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101774 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_8 := arithEq_of_rows h (row := 1664) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_8
  simp only [← c101769, ← c101770, ← c101771] at e_1664_8
  have e_1664_9 := arithEq_of_rows h (row := 1664) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_9
  simp only [← c101772, ← c101773, k2, ← c101774] at e_1664_9
  have hr := e_1664_9
  simp only [e_1664_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16643)) := by
  have c101775 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101776 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101777 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101778 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101779 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101780 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_10 := arithEq_of_rows h (row := 1664) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_10
  simp only [← c101775, ← c101776, ← c101777] at e_1664_10
  have e_1664_11 := arithEq_of_rows h (row := 1664) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_11
  simp only [← c101778, ← c101779, k2, ← c101780] at e_1664_11
  have hr := e_1664_11
  simp only [e_1664_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16644)) := by
  have c101781 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101782 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101783 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101784 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101785 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101786 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_12 := arithEq_of_rows h (row := 1664) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_12
  simp only [← c101781, ← c101782, ← c101783] at e_1664_12
  have e_1664_13 := arithEq_of_rows h (row := 1664) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_13
  simp only [← c101784, ← c101785, k2, ← c101786] at e_1664_13
  have hr := e_1664_13
  simp only [e_1664_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16645)) := by
  have c101787 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101788 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101789 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101790 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101791 := (publicBatchWrapper86_copies3180 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101792 := (publicBatchWrapper86_copies3181 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_14 := arithEq_of_rows h (row := 1664) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_14
  simp only [← c101787, ← c101788, ← c101789] at e_1664_14
  have e_1664_15 := arithEq_of_rows h (row := 1664) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_15
  simp only [← c101790, ← c101791, k2, ← c101792] at e_1664_15
  have hr := e_1664_15
  simp only [e_1664_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16646)) := by
  have c101793 := (publicBatchWrapper86_copies3181 a h).2.1
  have c101794 := (publicBatchWrapper86_copies3181 a h).2.2.1
  have c101795 := (publicBatchWrapper86_copies3181 a h).2.2.2.1
  have c101796 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.1
  have c101797 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.1
  have c101798 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_16 := arithEq_of_rows h (row := 1664) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_16
  simp only [← c101793, ← c101794, ← c101795] at e_1664_16
  have e_1664_17 := arithEq_of_rows h (row := 1664) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_17
  simp only [← c101796, ← c101797, k2, ← c101798] at e_1664_17
  have hr := e_1664_17
  simp only [e_1664_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1664 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16647)) := by
  have c101799 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.1
  have c101800 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.1
  have c101801 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.1
  have c101802 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101803 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101804 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1664_18 := arithEq_of_rows h (row := 1664) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_18
  simp only [← c101799, ← c101800, ← c101801] at e_1664_18
  have e_1664_19 := arithEq_of_rows h (row := 1664) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1664_19
  simp only [← c101802, ← c101803, k2, ← c101804] at e_1664_19
  have hr := e_1664_19
  simp only [e_1664_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16648)) := by
  have c101805 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101806 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101807 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101808 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101809 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101810 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_0 := arithEq_of_rows h (row := 1665) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_0
  simp only [← c101805, ← c101806, ← c101807] at e_1665_0
  have e_1665_1 := arithEq_of_rows h (row := 1665) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_1
  simp only [← c101808, ← c101809, k2, ← c101810] at e_1665_1
  have hr := e_1665_1
  simp only [e_1665_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16649)) := by
  have c101811 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101812 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101813 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101814 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101815 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101816 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_2 := arithEq_of_rows h (row := 1665) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_2
  simp only [← c101811, ← c101812, ← c101813] at e_1665_2
  have e_1665_3 := arithEq_of_rows h (row := 1665) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_3
  simp only [← c101814, ← c101815, k2, ← c101816] at e_1665_3
  have hr := e_1665_3
  simp only [e_1665_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16650)) := by
  have c101817 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101818 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101819 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101820 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101821 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101822 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_4 := arithEq_of_rows h (row := 1665) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_4
  simp only [← c101817, ← c101818, ← c101819] at e_1665_4
  have e_1665_5 := arithEq_of_rows h (row := 1665) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_5
  simp only [← c101820, ← c101821, k2, ← c101822] at e_1665_5
  have hr := e_1665_5
  simp only [e_1665_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16651)) := by
  have c101823 := (publicBatchWrapper86_copies3181 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101824 := (publicBatchWrapper86_copies3182 a h).1
  have c101825 := (publicBatchWrapper86_copies3182 a h).2.1
  have c101826 := (publicBatchWrapper86_copies3182 a h).2.2.1
  have c101827 := (publicBatchWrapper86_copies3182 a h).2.2.2.1
  have c101828 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_6 := arithEq_of_rows h (row := 1665) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_6
  simp only [← c101823, ← c101824, ← c101825] at e_1665_6
  have e_1665_7 := arithEq_of_rows h (row := 1665) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_7
  simp only [← c101826, ← c101827, k2, ← c101828] at e_1665_7
  have hr := e_1665_7
  simp only [e_1665_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16652)) := by
  have c101829 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.1
  have c101830 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.1
  have c101831 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.1
  have c101832 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.1
  have c101833 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.1
  have c101834 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_8 := arithEq_of_rows h (row := 1665) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_8
  simp only [← c101829, ← c101830, ← c101831] at e_1665_8
  have e_1665_9 := arithEq_of_rows h (row := 1665) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_9
  simp only [← c101832, ← c101833, k2, ← c101834] at e_1665_9
  have hr := e_1665_9
  simp only [e_1665_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16653)) := by
  have c101835 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101836 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101837 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101838 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101839 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101840 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_10 := arithEq_of_rows h (row := 1665) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_10
  simp only [← c101835, ← c101836, ← c101837] at e_1665_10
  have e_1665_11 := arithEq_of_rows h (row := 1665) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_11
  simp only [← c101838, ← c101839, k2, ← c101840] at e_1665_11
  have hr := e_1665_11
  simp only [e_1665_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16654)) := by
  have c101841 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101842 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101843 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101844 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101845 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101846 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_12 := arithEq_of_rows h (row := 1665) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_12
  simp only [← c101841, ← c101842, ← c101843] at e_1665_12
  have e_1665_13 := arithEq_of_rows h (row := 1665) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_13
  simp only [← c101844, ← c101845, k2, ← c101846] at e_1665_13
  have hr := e_1665_13
  simp only [e_1665_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16655)) := by
  have c101847 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101848 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101849 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101850 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101851 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101852 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_14 := arithEq_of_rows h (row := 1665) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_14
  simp only [← c101847, ← c101848, ← c101849] at e_1665_14
  have e_1665_15 := arithEq_of_rows h (row := 1665) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_15
  simp only [← c101850, ← c101851, k2, ← c101852] at e_1665_15
  have hr := e_1665_15
  simp only [e_1665_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16656)) := by
  have c101853 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101854 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101855 := (publicBatchWrapper86_copies3182 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101856 := (publicBatchWrapper86_copies3183 a h).1
  have c101857 := (publicBatchWrapper86_copies3183 a h).2.1
  have c101858 := (publicBatchWrapper86_copies3183 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_16 := arithEq_of_rows h (row := 1665) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_16
  simp only [← c101853, ← c101854, ← c101855] at e_1665_16
  have e_1665_17 := arithEq_of_rows h (row := 1665) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_17
  simp only [← c101856, ← c101857, k2, ← c101858] at e_1665_17
  have hr := e_1665_17
  simp only [e_1665_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1665 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16657)) := by
  have c101859 := (publicBatchWrapper86_copies3183 a h).2.2.2.1
  have c101860 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.1
  have c101861 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.1
  have c101862 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.1
  have c101863 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.1
  have c101864 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1665_18 := arithEq_of_rows h (row := 1665) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_18
  simp only [← c101859, ← c101860, ← c101861] at e_1665_18
  have e_1665_19 := arithEq_of_rows h (row := 1665) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1665_19
  simp only [← c101862, ← c101863, k2, ← c101864] at e_1665_19
  have hr := e_1665_19
  simp only [e_1665_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16658)) := by
  have c101865 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.1
  have c101866 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101867 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101868 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101869 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101870 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_0 := arithEq_of_rows h (row := 1666) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_0
  simp only [← c101865, ← c101866, ← c101867] at e_1666_0
  have e_1666_1 := arithEq_of_rows h (row := 1666) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_1
  simp only [← c101868, ← c101869, k2, ← c101870] at e_1666_1
  have hr := e_1666_1
  simp only [e_1666_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16659)) := by
  have c101871 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101872 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101873 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101874 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101875 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101876 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_2 := arithEq_of_rows h (row := 1666) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_2
  simp only [← c101871, ← c101872, ← c101873] at e_1666_2
  have e_1666_3 := arithEq_of_rows h (row := 1666) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_3
  simp only [← c101874, ← c101875, k2, ← c101876] at e_1666_3
  have hr := e_1666_3
  simp only [e_1666_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16660)) := by
  have c101877 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101878 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101879 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101880 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101881 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101882 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_4 := arithEq_of_rows h (row := 1666) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_4
  simp only [← c101877, ← c101878, ← c101879] at e_1666_4
  have e_1666_5 := arithEq_of_rows h (row := 1666) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_5
  simp only [← c101880, ← c101881, k2, ← c101882] at e_1666_5
  have hr := e_1666_5
  simp only [e_1666_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16661)) := by
  have c101883 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101884 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101885 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101886 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101887 := (publicBatchWrapper86_copies3183 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101888 := (publicBatchWrapper86_copies3184 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_6 := arithEq_of_rows h (row := 1666) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_6
  simp only [← c101883, ← c101884, ← c101885] at e_1666_6
  have e_1666_7 := arithEq_of_rows h (row := 1666) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_7
  simp only [← c101886, ← c101887, k2, ← c101888] at e_1666_7
  have hr := e_1666_7
  simp only [e_1666_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16662)) := by
  have c101889 := (publicBatchWrapper86_copies3184 a h).2.1
  have c101890 := (publicBatchWrapper86_copies3184 a h).2.2.1
  have c101891 := (publicBatchWrapper86_copies3184 a h).2.2.2.1
  have c101892 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.1
  have c101893 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.1
  have c101894 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_8 := arithEq_of_rows h (row := 1666) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_8
  simp only [← c101889, ← c101890, ← c101891] at e_1666_8
  have e_1666_9 := arithEq_of_rows h (row := 1666) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_9
  simp only [← c101892, ← c101893, k2, ← c101894] at e_1666_9
  have hr := e_1666_9
  simp only [e_1666_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16663)) := by
  have c101895 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.1
  have c101896 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.1
  have c101897 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.1
  have c101898 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101899 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101900 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_10 := arithEq_of_rows h (row := 1666) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_10
  simp only [← c101895, ← c101896, ← c101897] at e_1666_10
  have e_1666_11 := arithEq_of_rows h (row := 1666) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_11
  simp only [← c101898, ← c101899, k2, ← c101900] at e_1666_11
  have hr := e_1666_11
  simp only [e_1666_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16664)) := by
  have c101901 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101902 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101903 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101904 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101905 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101906 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_12 := arithEq_of_rows h (row := 1666) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_12
  simp only [← c101901, ← c101902, ← c101903] at e_1666_12
  have e_1666_13 := arithEq_of_rows h (row := 1666) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_13
  simp only [← c101904, ← c101905, k2, ← c101906] at e_1666_13
  have hr := e_1666_13
  simp only [e_1666_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16665)) := by
  have c101907 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101908 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101909 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101910 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101911 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101912 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_14 := arithEq_of_rows h (row := 1666) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_14
  simp only [← c101907, ← c101908, ← c101909] at e_1666_14
  have e_1666_15 := arithEq_of_rows h (row := 1666) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_15
  simp only [← c101910, ← c101911, k2, ← c101912] at e_1666_15
  have hr := e_1666_15
  simp only [e_1666_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16666)) := by
  have c101913 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101914 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101915 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101916 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101917 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101918 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_16 := arithEq_of_rows h (row := 1666) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_16
  simp only [← c101913, ← c101914, ← c101915] at e_1666_16
  have e_1666_17 := arithEq_of_rows h (row := 1666) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_17
  simp only [← c101916, ← c101917, k2, ← c101918] at e_1666_17
  have hr := e_1666_17
  simp only [e_1666_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1666 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16667)) := by
  have c101919 := (publicBatchWrapper86_copies3184 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101920 := (publicBatchWrapper86_copies3185 a h).1
  have c101921 := (publicBatchWrapper86_copies3185 a h).2.1
  have c101922 := (publicBatchWrapper86_copies3185 a h).2.2.1
  have c101923 := (publicBatchWrapper86_copies3185 a h).2.2.2.1
  have c101924 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1666_18 := arithEq_of_rows h (row := 1666) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_18
  simp only [← c101919, ← c101920, ← c101921] at e_1666_18
  have e_1666_19 := arithEq_of_rows h (row := 1666) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1666_19
  simp only [← c101922, ← c101923, k2, ← c101924] at e_1666_19
  have hr := e_1666_19
  simp only [e_1666_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16668)) := by
  have c101925 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.1
  have c101926 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.1
  have c101927 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.1
  have c101928 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.1
  have c101929 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.1
  have c101930 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_0 := arithEq_of_rows h (row := 1667) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_0
  simp only [← c101925, ← c101926, ← c101927] at e_1667_0
  have e_1667_1 := arithEq_of_rows h (row := 1667) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_1
  simp only [← c101928, ← c101929, k2, ← c101930] at e_1667_1
  have hr := e_1667_1
  simp only [e_1667_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16669)) := by
  have c101931 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101932 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101933 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101934 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101935 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101936 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_2 := arithEq_of_rows h (row := 1667) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_2
  simp only [← c101931, ← c101932, ← c101933] at e_1667_2
  have e_1667_3 := arithEq_of_rows h (row := 1667) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_3
  simp only [← c101934, ← c101935, k2, ← c101936] at e_1667_3
  have hr := e_1667_3
  simp only [e_1667_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16670)) := by
  have c101937 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101938 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101939 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101940 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101941 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101942 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_4 := arithEq_of_rows h (row := 1667) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_4
  simp only [← c101937, ← c101938, ← c101939] at e_1667_4
  have e_1667_5 := arithEq_of_rows h (row := 1667) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_5
  simp only [← c101940, ← c101941, k2, ← c101942] at e_1667_5
  have hr := e_1667_5
  simp only [e_1667_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16671)) := by
  have c101943 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101944 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101945 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101946 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101947 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101948 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_6 := arithEq_of_rows h (row := 1667) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_6
  simp only [← c101943, ← c101944, ← c101945] at e_1667_6
  have e_1667_7 := arithEq_of_rows h (row := 1667) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_7
  simp only [← c101946, ← c101947, k2, ← c101948] at e_1667_7
  have hr := e_1667_7
  simp only [e_1667_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16672)) := by
  have c101949 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101950 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101951 := (publicBatchWrapper86_copies3185 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101952 := (publicBatchWrapper86_copies3186 a h).1
  have c101953 := (publicBatchWrapper86_copies3186 a h).2.1
  have c101954 := (publicBatchWrapper86_copies3186 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_8 := arithEq_of_rows h (row := 1667) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_8
  simp only [← c101949, ← c101950, ← c101951] at e_1667_8
  have e_1667_9 := arithEq_of_rows h (row := 1667) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_9
  simp only [← c101952, ← c101953, k2, ← c101954] at e_1667_9
  have hr := e_1667_9
  simp only [e_1667_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16673)) := by
  have c101955 := (publicBatchWrapper86_copies3186 a h).2.2.2.1
  have c101956 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.1
  have c101957 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.1
  have c101958 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.1
  have c101959 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.1
  have c101960 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_10 := arithEq_of_rows h (row := 1667) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_10
  simp only [← c101955, ← c101956, ← c101957] at e_1667_10
  have e_1667_11 := arithEq_of_rows h (row := 1667) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_11
  simp only [← c101958, ← c101959, k2, ← c101960] at e_1667_11
  have hr := e_1667_11
  simp only [e_1667_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16674)) := by
  have c101961 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.1
  have c101962 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101963 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101964 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101965 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101966 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_12 := arithEq_of_rows h (row := 1667) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_12
  simp only [← c101961, ← c101962, ← c101963] at e_1667_12
  have e_1667_13 := arithEq_of_rows h (row := 1667) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_13
  simp only [← c101964, ← c101965, k2, ← c101966] at e_1667_13
  have hr := e_1667_13
  simp only [e_1667_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16675)) := by
  have c101967 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101968 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101969 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101970 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101971 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101972 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_14 := arithEq_of_rows h (row := 1667) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_14
  simp only [← c101967, ← c101968, ← c101969] at e_1667_14
  have e_1667_15 := arithEq_of_rows h (row := 1667) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_15
  simp only [← c101970, ← c101971, k2, ← c101972] at e_1667_15
  have hr := e_1667_15
  simp only [e_1667_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16676)) := by
  have c101973 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101974 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101975 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101976 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101977 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101978 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_16 := arithEq_of_rows h (row := 1667) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_16
  simp only [← c101973, ← c101974, ← c101975] at e_1667_16
  have e_1667_17 := arithEq_of_rows h (row := 1667) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_17
  simp only [← c101976, ← c101977, k2, ← c101978] at e_1667_17
  have hr := e_1667_17
  simp only [e_1667_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1667 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16677)) := by
  have c101979 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101980 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101981 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101982 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101983 := (publicBatchWrapper86_copies3186 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c101984 := (publicBatchWrapper86_copies3187 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1667_18 := arithEq_of_rows h (row := 1667) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_18
  simp only [← c101979, ← c101980, ← c101981] at e_1667_18
  have e_1667_19 := arithEq_of_rows h (row := 1667) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1667_19
  simp only [← c101982, ← c101983, k2, ← c101984] at e_1667_19
  have hr := e_1667_19
  simp only [e_1667_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16678)) := by
  have c101985 := (publicBatchWrapper86_copies3187 a h).2.1
  have c101986 := (publicBatchWrapper86_copies3187 a h).2.2.1
  have c101987 := (publicBatchWrapper86_copies3187 a h).2.2.2.1
  have c101988 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.1
  have c101989 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.1
  have c101990 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_0 := arithEq_of_rows h (row := 1668) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_0
  simp only [← c101985, ← c101986, ← c101987] at e_1668_0
  have e_1668_1 := arithEq_of_rows h (row := 1668) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_1
  simp only [← c101988, ← c101989, k2, ← c101990] at e_1668_1
  have hr := e_1668_1
  simp only [e_1668_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16679)) := by
  have c101991 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.1
  have c101992 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.1
  have c101993 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.1
  have c101994 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.1
  have c101995 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c101996 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_2 := arithEq_of_rows h (row := 1668) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_2
  simp only [← c101991, ← c101992, ← c101993] at e_1668_2
  have e_1668_3 := arithEq_of_rows h (row := 1668) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_3
  simp only [← c101994, ← c101995, k2, ← c101996] at e_1668_3
  have hr := e_1668_3
  simp only [e_1668_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16680)) := by
  have c101997 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101998 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c101999 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102000 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102001 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102002 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_4 := arithEq_of_rows h (row := 1668) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_4
  simp only [← c101997, ← c101998, ← c101999] at e_1668_4
  have e_1668_5 := arithEq_of_rows h (row := 1668) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_5
  simp only [← c102000, ← c102001, k2, ← c102002] at e_1668_5
  have hr := e_1668_5
  simp only [e_1668_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16681)) := by
  have c102003 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102004 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102005 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102006 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102007 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102008 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_6 := arithEq_of_rows h (row := 1668) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_6
  simp only [← c102003, ← c102004, ← c102005] at e_1668_6
  have e_1668_7 := arithEq_of_rows h (row := 1668) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_7
  simp only [← c102006, ← c102007, k2, ← c102008] at e_1668_7
  have hr := e_1668_7
  simp only [e_1668_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16682)) := by
  have c102009 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102010 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102011 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102012 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102013 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102014 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_8 := arithEq_of_rows h (row := 1668) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_8
  simp only [← c102009, ← c102010, ← c102011] at e_1668_8
  have e_1668_9 := arithEq_of_rows h (row := 1668) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_9
  simp only [← c102012, ← c102013, k2, ← c102014] at e_1668_9
  have hr := e_1668_9
  simp only [e_1668_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16683)) := by
  have c102015 := (publicBatchWrapper86_copies3187 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102016 := (publicBatchWrapper86_copies3188 a h).1
  have c102017 := (publicBatchWrapper86_copies3188 a h).2.1
  have c102018 := (publicBatchWrapper86_copies3188 a h).2.2.1
  have c102019 := (publicBatchWrapper86_copies3188 a h).2.2.2.1
  have c102020 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_10 := arithEq_of_rows h (row := 1668) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_10
  simp only [← c102015, ← c102016, ← c102017] at e_1668_10
  have e_1668_11 := arithEq_of_rows h (row := 1668) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_11
  simp only [← c102018, ← c102019, k2, ← c102020] at e_1668_11
  have hr := e_1668_11
  simp only [e_1668_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16684)) := by
  have c102021 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.1
  have c102022 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.1
  have c102023 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.1
  have c102024 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.1
  have c102025 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.1
  have c102026 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_12 := arithEq_of_rows h (row := 1668) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_12
  simp only [← c102021, ← c102022, ← c102023] at e_1668_12
  have e_1668_13 := arithEq_of_rows h (row := 1668) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_13
  simp only [← c102024, ← c102025, k2, ← c102026] at e_1668_13
  have hr := e_1668_13
  simp only [e_1668_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16685)) := by
  have c102027 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102028 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102029 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102030 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102031 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102032 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_14 := arithEq_of_rows h (row := 1668) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_14
  simp only [← c102027, ← c102028, ← c102029] at e_1668_14
  have e_1668_15 := arithEq_of_rows h (row := 1668) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_15
  simp only [← c102030, ← c102031, k2, ← c102032] at e_1668_15
  have hr := e_1668_15
  simp only [e_1668_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16686)) := by
  have c102033 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102034 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102035 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102036 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102037 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102038 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_16 := arithEq_of_rows h (row := 1668) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_16
  simp only [← c102033, ← c102034, ← c102035] at e_1668_16
  have e_1668_17 := arithEq_of_rows h (row := 1668) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_17
  simp only [← c102036, ← c102037, k2, ← c102038] at e_1668_17
  have hr := e_1668_17
  simp only [e_1668_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1668 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16687)) := by
  have c102039 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102040 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102041 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102042 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102043 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102044 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1668_18 := arithEq_of_rows h (row := 1668) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_18
  simp only [← c102039, ← c102040, ← c102041] at e_1668_18
  have e_1668_19 := arithEq_of_rows h (row := 1668) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1668_19
  simp only [← c102042, ← c102043, k2, ← c102044] at e_1668_19
  have hr := e_1668_19
  simp only [e_1668_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16688)) := by
  have c102045 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102046 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102047 := (publicBatchWrapper86_copies3188 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102048 := (publicBatchWrapper86_copies3189 a h).1
  have c102049 := (publicBatchWrapper86_copies3189 a h).2.1
  have c102050 := (publicBatchWrapper86_copies3189 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_0 := arithEq_of_rows h (row := 1669) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_0
  simp only [← c102045, ← c102046, ← c102047] at e_1669_0
  have e_1669_1 := arithEq_of_rows h (row := 1669) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_1
  simp only [← c102048, ← c102049, k2, ← c102050] at e_1669_1
  have hr := e_1669_1
  simp only [e_1669_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16689)) := by
  have c102051 := (publicBatchWrapper86_copies3189 a h).2.2.2.1
  have c102052 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.1
  have c102053 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.1
  have c102054 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.1
  have c102055 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.1
  have c102056 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_2 := arithEq_of_rows h (row := 1669) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_2
  simp only [← c102051, ← c102052, ← c102053] at e_1669_2
  have e_1669_3 := arithEq_of_rows h (row := 1669) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_3
  simp only [← c102054, ← c102055, k2, ← c102056] at e_1669_3
  have hr := e_1669_3
  simp only [e_1669_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16690)) := by
  have c102057 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.1
  have c102058 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102059 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102060 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102061 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102062 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_4 := arithEq_of_rows h (row := 1669) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_4
  simp only [← c102057, ← c102058, ← c102059] at e_1669_4
  have e_1669_5 := arithEq_of_rows h (row := 1669) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_5
  simp only [← c102060, ← c102061, k2, ← c102062] at e_1669_5
  have hr := e_1669_5
  simp only [e_1669_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16691)) := by
  have c102063 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102064 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102065 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102066 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102067 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102068 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_6 := arithEq_of_rows h (row := 1669) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_6
  simp only [← c102063, ← c102064, ← c102065] at e_1669_6
  have e_1669_7 := arithEq_of_rows h (row := 1669) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_7
  simp only [← c102066, ← c102067, k2, ← c102068] at e_1669_7
  have hr := e_1669_7
  simp only [e_1669_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16692)) := by
  have c102069 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102070 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102071 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102072 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102073 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102074 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_8 := arithEq_of_rows h (row := 1669) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_8
  simp only [← c102069, ← c102070, ← c102071] at e_1669_8
  have e_1669_9 := arithEq_of_rows h (row := 1669) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_9
  simp only [← c102072, ← c102073, k2, ← c102074] at e_1669_9
  have hr := e_1669_9
  simp only [e_1669_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16693)) := by
  have c102075 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102076 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102077 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102078 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102079 := (publicBatchWrapper86_copies3189 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102080 := (publicBatchWrapper86_copies3190 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_10 := arithEq_of_rows h (row := 1669) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_10
  simp only [← c102075, ← c102076, ← c102077] at e_1669_10
  have e_1669_11 := arithEq_of_rows h (row := 1669) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_11
  simp only [← c102078, ← c102079, k2, ← c102080] at e_1669_11
  have hr := e_1669_11
  simp only [e_1669_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16694)) := by
  have c102081 := (publicBatchWrapper86_copies3190 a h).2.1
  have c102082 := (publicBatchWrapper86_copies3190 a h).2.2.1
  have c102083 := (publicBatchWrapper86_copies3190 a h).2.2.2.1
  have c102084 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.1
  have c102085 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.1
  have c102086 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_12 := arithEq_of_rows h (row := 1669) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_12
  simp only [← c102081, ← c102082, ← c102083] at e_1669_12
  have e_1669_13 := arithEq_of_rows h (row := 1669) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_13
  simp only [← c102084, ← c102085, k2, ← c102086] at e_1669_13
  have hr := e_1669_13
  simp only [e_1669_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16695)) := by
  have c102087 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.1
  have c102088 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.1
  have c102089 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.1
  have c102090 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102091 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102092 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_14 := arithEq_of_rows h (row := 1669) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_14
  simp only [← c102087, ← c102088, ← c102089] at e_1669_14
  have e_1669_15 := arithEq_of_rows h (row := 1669) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_15
  simp only [← c102090, ← c102091, k2, ← c102092] at e_1669_15
  have hr := e_1669_15
  simp only [e_1669_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16696)) := by
  have c102093 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102094 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102095 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102096 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102097 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102098 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_16 := arithEq_of_rows h (row := 1669) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_16
  simp only [← c102093, ← c102094, ← c102095] at e_1669_16
  have e_1669_17 := arithEq_of_rows h (row := 1669) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_17
  simp only [← c102096, ← c102097, k2, ← c102098] at e_1669_17
  have hr := e_1669_17
  simp only [e_1669_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1669 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16697)) := by
  have c102099 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102100 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102101 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102102 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102103 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102104 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1669_18 := arithEq_of_rows h (row := 1669) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_18
  simp only [← c102099, ← c102100, ← c102101] at e_1669_18
  have e_1669_19 := arithEq_of_rows h (row := 1669) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1669_19
  simp only [← c102102, ← c102103, k2, ← c102104] at e_1669_19
  have hr := e_1669_19
  simp only [e_1669_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16698)) := by
  have c102105 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102106 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102107 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102108 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102109 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102110 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_0 := arithEq_of_rows h (row := 1670) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_0
  simp only [← c102105, ← c102106, ← c102107] at e_1670_0
  have e_1670_1 := arithEq_of_rows h (row := 1670) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_1
  simp only [← c102108, ← c102109, k2, ← c102110] at e_1670_1
  have hr := e_1670_1
  simp only [e_1670_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16699)) := by
  have c102111 := (publicBatchWrapper86_copies3190 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102112 := (publicBatchWrapper86_copies3191 a h).1
  have c102113 := (publicBatchWrapper86_copies3191 a h).2.1
  have c102114 := (publicBatchWrapper86_copies3191 a h).2.2.1
  have c102115 := (publicBatchWrapper86_copies3191 a h).2.2.2.1
  have c102116 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_2 := arithEq_of_rows h (row := 1670) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_2
  simp only [← c102111, ← c102112, ← c102113] at e_1670_2
  have e_1670_3 := arithEq_of_rows h (row := 1670) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_3
  simp only [← c102114, ← c102115, k2, ← c102116] at e_1670_3
  have hr := e_1670_3
  simp only [e_1670_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16700)) := by
  have c102117 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.1
  have c102118 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.1
  have c102119 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.1
  have c102120 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.1
  have c102121 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.1
  have c102122 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_4 := arithEq_of_rows h (row := 1670) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_4
  simp only [← c102117, ← c102118, ← c102119] at e_1670_4
  have e_1670_5 := arithEq_of_rows h (row := 1670) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_5
  simp only [← c102120, ← c102121, k2, ← c102122] at e_1670_5
  have hr := e_1670_5
  simp only [e_1670_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16701)) := by
  have c102123 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102124 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102125 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102126 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102127 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102128 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_6 := arithEq_of_rows h (row := 1670) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_6
  simp only [← c102123, ← c102124, ← c102125] at e_1670_6
  have e_1670_7 := arithEq_of_rows h (row := 1670) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_7
  simp only [← c102126, ← c102127, k2, ← c102128] at e_1670_7
  have hr := e_1670_7
  simp only [e_1670_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16702)) := by
  have c102129 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102130 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102131 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102132 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102133 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102134 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_8 := arithEq_of_rows h (row := 1670) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_8
  simp only [← c102129, ← c102130, ← c102131] at e_1670_8
  have e_1670_9 := arithEq_of_rows h (row := 1670) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_9
  simp only [← c102132, ← c102133, k2, ← c102134] at e_1670_9
  have hr := e_1670_9
  simp only [e_1670_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16703)) := by
  have c102135 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102136 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102137 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102138 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102139 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102140 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_10 := arithEq_of_rows h (row := 1670) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_10
  simp only [← c102135, ← c102136, ← c102137] at e_1670_10
  have e_1670_11 := arithEq_of_rows h (row := 1670) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_11
  simp only [← c102138, ← c102139, k2, ← c102140] at e_1670_11
  have hr := e_1670_11
  simp only [e_1670_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16704)) := by
  have c102141 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102142 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102143 := (publicBatchWrapper86_copies3191 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102144 := (publicBatchWrapper86_copies3192 a h).1
  have c102145 := (publicBatchWrapper86_copies3192 a h).2.1
  have c102146 := (publicBatchWrapper86_copies3192 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_12 := arithEq_of_rows h (row := 1670) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_12
  simp only [← c102141, ← c102142, ← c102143] at e_1670_12
  have e_1670_13 := arithEq_of_rows h (row := 1670) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_13
  simp only [← c102144, ← c102145, k2, ← c102146] at e_1670_13
  have hr := e_1670_13
  simp only [e_1670_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16705)) := by
  have c102147 := (publicBatchWrapper86_copies3192 a h).2.2.2.1
  have c102148 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.1
  have c102149 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.1
  have c102150 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.1
  have c102151 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.1
  have c102152 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_14 := arithEq_of_rows h (row := 1670) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_14
  simp only [← c102147, ← c102148, ← c102149] at e_1670_14
  have e_1670_15 := arithEq_of_rows h (row := 1670) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_15
  simp only [← c102150, ← c102151, k2, ← c102152] at e_1670_15
  have hr := e_1670_15
  simp only [e_1670_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16706)) := by
  have c102153 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.1
  have c102154 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102155 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102156 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102157 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102158 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_16 := arithEq_of_rows h (row := 1670) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_16
  simp only [← c102153, ← c102154, ← c102155] at e_1670_16
  have e_1670_17 := arithEq_of_rows h (row := 1670) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_17
  simp only [← c102156, ← c102157, k2, ← c102158] at e_1670_17
  have hr := e_1670_17
  simp only [e_1670_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1670 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16707)) := by
  have c102159 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102160 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102161 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102162 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102163 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102164 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1670_18 := arithEq_of_rows h (row := 1670) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_18
  simp only [← c102159, ← c102160, ← c102161] at e_1670_18
  have e_1670_19 := arithEq_of_rows h (row := 1670) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1670_19
  simp only [← c102162, ← c102163, k2, ← c102164] at e_1670_19
  have hr := e_1670_19
  simp only [e_1670_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16708)) := by
  have c102165 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102166 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102167 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102168 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102169 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102170 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_0 := arithEq_of_rows h (row := 1671) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_0
  simp only [← c102165, ← c102166, ← c102167] at e_1671_0
  have e_1671_1 := arithEq_of_rows h (row := 1671) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_1
  simp only [← c102168, ← c102169, k2, ← c102170] at e_1671_1
  have hr := e_1671_1
  simp only [e_1671_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16709)) := by
  have c102171 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102172 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102173 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102174 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102175 := (publicBatchWrapper86_copies3192 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102176 := (publicBatchWrapper86_copies3193 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_2 := arithEq_of_rows h (row := 1671) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_2
  simp only [← c102171, ← c102172, ← c102173] at e_1671_2
  have e_1671_3 := arithEq_of_rows h (row := 1671) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_3
  simp only [← c102174, ← c102175, k2, ← c102176] at e_1671_3
  have hr := e_1671_3
  simp only [e_1671_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16710)) := by
  have c102177 := (publicBatchWrapper86_copies3193 a h).2.1
  have c102178 := (publicBatchWrapper86_copies3193 a h).2.2.1
  have c102179 := (publicBatchWrapper86_copies3193 a h).2.2.2.1
  have c102180 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.1
  have c102181 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.1
  have c102182 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_4 := arithEq_of_rows h (row := 1671) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_4
  simp only [← c102177, ← c102178, ← c102179] at e_1671_4
  have e_1671_5 := arithEq_of_rows h (row := 1671) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_5
  simp only [← c102180, ← c102181, k2, ← c102182] at e_1671_5
  have hr := e_1671_5
  simp only [e_1671_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16711)) := by
  have c102183 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.1
  have c102184 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.1
  have c102185 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.1
  have c102186 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102187 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102188 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_6 := arithEq_of_rows h (row := 1671) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_6
  simp only [← c102183, ← c102184, ← c102185] at e_1671_6
  have e_1671_7 := arithEq_of_rows h (row := 1671) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_7
  simp only [← c102186, ← c102187, k2, ← c102188] at e_1671_7
  have hr := e_1671_7
  simp only [e_1671_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16712)) := by
  have c102189 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102190 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102191 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102192 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102193 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102194 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_8 := arithEq_of_rows h (row := 1671) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_8
  simp only [← c102189, ← c102190, ← c102191] at e_1671_8
  have e_1671_9 := arithEq_of_rows h (row := 1671) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_9
  simp only [← c102192, ← c102193, k2, ← c102194] at e_1671_9
  have hr := e_1671_9
  simp only [e_1671_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16713)) := by
  have c102195 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102196 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102197 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102198 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102199 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102200 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_10 := arithEq_of_rows h (row := 1671) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_10
  simp only [← c102195, ← c102196, ← c102197] at e_1671_10
  have e_1671_11 := arithEq_of_rows h (row := 1671) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_11
  simp only [← c102198, ← c102199, k2, ← c102200] at e_1671_11
  have hr := e_1671_11
  simp only [e_1671_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16714)) := by
  have c102201 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102202 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102203 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102204 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102205 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102206 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_12 := arithEq_of_rows h (row := 1671) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_12
  simp only [← c102201, ← c102202, ← c102203] at e_1671_12
  have e_1671_13 := arithEq_of_rows h (row := 1671) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_13
  simp only [← c102204, ← c102205, k2, ← c102206] at e_1671_13
  have hr := e_1671_13
  simp only [e_1671_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16715)) := by
  have c102207 := (publicBatchWrapper86_copies3193 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102208 := (publicBatchWrapper86_copies3194 a h).1
  have c102209 := (publicBatchWrapper86_copies3194 a h).2.1
  have c102210 := (publicBatchWrapper86_copies3194 a h).2.2.1
  have c102211 := (publicBatchWrapper86_copies3194 a h).2.2.2.1
  have c102212 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_14 := arithEq_of_rows h (row := 1671) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_14
  simp only [← c102207, ← c102208, ← c102209] at e_1671_14
  have e_1671_15 := arithEq_of_rows h (row := 1671) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_15
  simp only [← c102210, ← c102211, k2, ← c102212] at e_1671_15
  have hr := e_1671_15
  simp only [e_1671_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16716)) := by
  have c102213 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.1
  have c102214 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.1
  have c102215 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.1
  have c102216 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.1
  have c102217 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.1
  have c102218 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_16 := arithEq_of_rows h (row := 1671) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_16
  simp only [← c102213, ← c102214, ← c102215] at e_1671_16
  have e_1671_17 := arithEq_of_rows h (row := 1671) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_17
  simp only [← c102216, ← c102217, k2, ← c102218] at e_1671_17
  have hr := e_1671_17
  simp only [e_1671_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1671 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16717)) := by
  have c102219 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102220 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102221 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102222 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102223 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102224 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1671_18 := arithEq_of_rows h (row := 1671) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_18
  simp only [← c102219, ← c102220, ← c102221] at e_1671_18
  have e_1671_19 := arithEq_of_rows h (row := 1671) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1671_19
  simp only [← c102222, ← c102223, k2, ← c102224] at e_1671_19
  have hr := e_1671_19
  simp only [e_1671_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16718)) := by
  have c102225 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102226 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102227 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102228 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102229 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102230 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_0 := arithEq_of_rows h (row := 1672) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_0
  simp only [← c102225, ← c102226, ← c102227] at e_1672_0
  have e_1672_1 := arithEq_of_rows h (row := 1672) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_1
  simp only [← c102228, ← c102229, k2, ← c102230] at e_1672_1
  have hr := e_1672_1
  simp only [e_1672_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16719)) := by
  have c102231 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102232 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102233 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102234 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102235 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102236 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_2 := arithEq_of_rows h (row := 1672) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_2
  simp only [← c102231, ← c102232, ← c102233] at e_1672_2
  have e_1672_3 := arithEq_of_rows h (row := 1672) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_3
  simp only [← c102234, ← c102235, k2, ← c102236] at e_1672_3
  have hr := e_1672_3
  simp only [e_1672_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16720)) := by
  have c102237 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102238 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102239 := (publicBatchWrapper86_copies3194 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102240 := (publicBatchWrapper86_copies3195 a h).1
  have c102241 := (publicBatchWrapper86_copies3195 a h).2.1
  have c102242 := (publicBatchWrapper86_copies3195 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_4 := arithEq_of_rows h (row := 1672) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_4
  simp only [← c102237, ← c102238, ← c102239] at e_1672_4
  have e_1672_5 := arithEq_of_rows h (row := 1672) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_5
  simp only [← c102240, ← c102241, k2, ← c102242] at e_1672_5
  have hr := e_1672_5
  simp only [e_1672_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16721)) := by
  have c102243 := (publicBatchWrapper86_copies3195 a h).2.2.2.1
  have c102244 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.1
  have c102245 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.1
  have c102246 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.1
  have c102247 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.1
  have c102248 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_6 := arithEq_of_rows h (row := 1672) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_6
  simp only [← c102243, ← c102244, ← c102245] at e_1672_6
  have e_1672_7 := arithEq_of_rows h (row := 1672) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_7
  simp only [← c102246, ← c102247, k2, ← c102248] at e_1672_7
  have hr := e_1672_7
  simp only [e_1672_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16722)) := by
  have c102249 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.1
  have c102250 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102251 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102252 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102253 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102254 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_8 := arithEq_of_rows h (row := 1672) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_8
  simp only [← c102249, ← c102250, ← c102251] at e_1672_8
  have e_1672_9 := arithEq_of_rows h (row := 1672) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_9
  simp only [← c102252, ← c102253, k2, ← c102254] at e_1672_9
  have hr := e_1672_9
  simp only [e_1672_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16723)) := by
  have c102255 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102256 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102257 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102258 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102259 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102260 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_10 := arithEq_of_rows h (row := 1672) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_10
  simp only [← c102255, ← c102256, ← c102257] at e_1672_10
  have e_1672_11 := arithEq_of_rows h (row := 1672) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_11
  simp only [← c102258, ← c102259, k2, ← c102260] at e_1672_11
  have hr := e_1672_11
  simp only [e_1672_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16724)) := by
  have c102261 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102262 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102263 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102264 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102265 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102266 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_12 := arithEq_of_rows h (row := 1672) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_12
  simp only [← c102261, ← c102262, ← c102263] at e_1672_12
  have e_1672_13 := arithEq_of_rows h (row := 1672) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_13
  simp only [← c102264, ← c102265, k2, ← c102266] at e_1672_13
  have hr := e_1672_13
  simp only [e_1672_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16725)) := by
  have c102267 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102268 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102269 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102270 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102271 := (publicBatchWrapper86_copies3195 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102272 := (publicBatchWrapper86_copies3196 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_14 := arithEq_of_rows h (row := 1672) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_14
  simp only [← c102267, ← c102268, ← c102269] at e_1672_14
  have e_1672_15 := arithEq_of_rows h (row := 1672) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_15
  simp only [← c102270, ← c102271, k2, ← c102272] at e_1672_15
  have hr := e_1672_15
  simp only [e_1672_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16726)) := by
  have c102273 := (publicBatchWrapper86_copies3196 a h).2.1
  have c102274 := (publicBatchWrapper86_copies3196 a h).2.2.1
  have c102275 := (publicBatchWrapper86_copies3196 a h).2.2.2.1
  have c102276 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.1
  have c102277 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.1
  have c102278 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_16 := arithEq_of_rows h (row := 1672) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_16
  simp only [← c102273, ← c102274, ← c102275] at e_1672_16
  have e_1672_17 := arithEq_of_rows h (row := 1672) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_17
  simp only [← c102276, ← c102277, k2, ← c102278] at e_1672_17
  have hr := e_1672_17
  simp only [e_1672_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1672 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16727)) := by
  have c102279 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.1
  have c102280 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.1
  have c102281 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.1
  have c102282 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102283 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102284 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1672_18 := arithEq_of_rows h (row := 1672) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_18
  simp only [← c102279, ← c102280, ← c102281] at e_1672_18
  have e_1672_19 := arithEq_of_rows h (row := 1672) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1672_19
  simp only [← c102282, ← c102283, k2, ← c102284] at e_1672_19
  have hr := e_1672_19
  simp only [e_1672_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16728)) := by
  have c102285 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102286 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102287 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102288 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102289 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102290 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_0 := arithEq_of_rows h (row := 1673) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_0
  simp only [← c102285, ← c102286, ← c102287] at e_1673_0
  have e_1673_1 := arithEq_of_rows h (row := 1673) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_1
  simp only [← c102288, ← c102289, k2, ← c102290] at e_1673_1
  have hr := e_1673_1
  simp only [e_1673_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16729)) := by
  have c102291 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102292 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102293 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102294 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102295 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102296 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_2 := arithEq_of_rows h (row := 1673) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_2
  simp only [← c102291, ← c102292, ← c102293] at e_1673_2
  have e_1673_3 := arithEq_of_rows h (row := 1673) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_3
  simp only [← c102294, ← c102295, k2, ← c102296] at e_1673_3
  have hr := e_1673_3
  simp only [e_1673_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16730)) := by
  have c102297 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102298 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102299 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102300 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102301 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102302 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_4 := arithEq_of_rows h (row := 1673) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_4
  simp only [← c102297, ← c102298, ← c102299] at e_1673_4
  have e_1673_5 := arithEq_of_rows h (row := 1673) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_5
  simp only [← c102300, ← c102301, k2, ← c102302] at e_1673_5
  have hr := e_1673_5
  simp only [e_1673_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16731)) := by
  have c102303 := (publicBatchWrapper86_copies3196 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102304 := (publicBatchWrapper86_copies3197 a h).1
  have c102305 := (publicBatchWrapper86_copies3197 a h).2.1
  have c102306 := (publicBatchWrapper86_copies3197 a h).2.2.1
  have c102307 := (publicBatchWrapper86_copies3197 a h).2.2.2.1
  have c102308 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_6 := arithEq_of_rows h (row := 1673) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_6
  simp only [← c102303, ← c102304, ← c102305] at e_1673_6
  have e_1673_7 := arithEq_of_rows h (row := 1673) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_7
  simp only [← c102306, ← c102307, k2, ← c102308] at e_1673_7
  have hr := e_1673_7
  simp only [e_1673_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16732)) := by
  have c102309 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.1
  have c102310 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.1
  have c102311 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.1
  have c102312 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.1
  have c102313 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.1
  have c102314 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_8 := arithEq_of_rows h (row := 1673) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_8
  simp only [← c102309, ← c102310, ← c102311] at e_1673_8
  have e_1673_9 := arithEq_of_rows h (row := 1673) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_9
  simp only [← c102312, ← c102313, k2, ← c102314] at e_1673_9
  have hr := e_1673_9
  simp only [e_1673_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16733)) := by
  have c102315 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102316 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102317 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102318 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102319 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102320 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_10 := arithEq_of_rows h (row := 1673) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_10
  simp only [← c102315, ← c102316, ← c102317] at e_1673_10
  have e_1673_11 := arithEq_of_rows h (row := 1673) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_11
  simp only [← c102318, ← c102319, k2, ← c102320] at e_1673_11
  have hr := e_1673_11
  simp only [e_1673_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16734)) := by
  have c102321 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102322 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102323 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102324 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102325 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102326 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_12 := arithEq_of_rows h (row := 1673) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_12
  simp only [← c102321, ← c102322, ← c102323] at e_1673_12
  have e_1673_13 := arithEq_of_rows h (row := 1673) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_13
  simp only [← c102324, ← c102325, k2, ← c102326] at e_1673_13
  have hr := e_1673_13
  simp only [e_1673_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16735)) := by
  have c102327 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102328 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102329 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102330 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102331 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102332 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_14 := arithEq_of_rows h (row := 1673) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_14
  simp only [← c102327, ← c102328, ← c102329] at e_1673_14
  have e_1673_15 := arithEq_of_rows h (row := 1673) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_15
  simp only [← c102330, ← c102331, k2, ← c102332] at e_1673_15
  have hr := e_1673_15
  simp only [e_1673_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16736)) := by
  have c102333 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102334 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102335 := (publicBatchWrapper86_copies3197 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102336 := (publicBatchWrapper86_copies3198 a h).1
  have c102337 := (publicBatchWrapper86_copies3198 a h).2.1
  have c102338 := (publicBatchWrapper86_copies3198 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_16 := arithEq_of_rows h (row := 1673) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_16
  simp only [← c102333, ← c102334, ← c102335] at e_1673_16
  have e_1673_17 := arithEq_of_rows h (row := 1673) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_17
  simp only [← c102336, ← c102337, k2, ← c102338] at e_1673_17
  have hr := e_1673_17
  simp only [e_1673_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1673 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16737)) := by
  have c102339 := (publicBatchWrapper86_copies3198 a h).2.2.2.1
  have c102340 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.1
  have c102341 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.1
  have c102342 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.1
  have c102343 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.1
  have c102344 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1673_18 := arithEq_of_rows h (row := 1673) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_18
  simp only [← c102339, ← c102340, ← c102341] at e_1673_18
  have e_1673_19 := arithEq_of_rows h (row := 1673) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1673_19
  simp only [← c102342, ← c102343, k2, ← c102344] at e_1673_19
  have hr := e_1673_19
  simp only [e_1673_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16738)) := by
  have c102345 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.1
  have c102346 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102347 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102348 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102349 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102350 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_0 := arithEq_of_rows h (row := 1674) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_0
  simp only [← c102345, ← c102346, ← c102347] at e_1674_0
  have e_1674_1 := arithEq_of_rows h (row := 1674) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_1
  simp only [← c102348, ← c102349, k2, ← c102350] at e_1674_1
  have hr := e_1674_1
  simp only [e_1674_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16739)) := by
  have c102351 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102352 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102353 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102354 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102355 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102356 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_2 := arithEq_of_rows h (row := 1674) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_2
  simp only [← c102351, ← c102352, ← c102353] at e_1674_2
  have e_1674_3 := arithEq_of_rows h (row := 1674) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_3
  simp only [← c102354, ← c102355, k2, ← c102356] at e_1674_3
  have hr := e_1674_3
  simp only [e_1674_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16740)) := by
  have c102357 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102358 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102359 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102360 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102361 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102362 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_4 := arithEq_of_rows h (row := 1674) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_4
  simp only [← c102357, ← c102358, ← c102359] at e_1674_4
  have e_1674_5 := arithEq_of_rows h (row := 1674) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_5
  simp only [← c102360, ← c102361, k2, ← c102362] at e_1674_5
  have hr := e_1674_5
  simp only [e_1674_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16741)) := by
  have c102363 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102364 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102365 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102366 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102367 := (publicBatchWrapper86_copies3198 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102368 := (publicBatchWrapper86_copies3199 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_6 := arithEq_of_rows h (row := 1674) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_6
  simp only [← c102363, ← c102364, ← c102365] at e_1674_6
  have e_1674_7 := arithEq_of_rows h (row := 1674) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_7
  simp only [← c102366, ← c102367, k2, ← c102368] at e_1674_7
  have hr := e_1674_7
  simp only [e_1674_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16742)) := by
  have c102369 := (publicBatchWrapper86_copies3199 a h).2.1
  have c102370 := (publicBatchWrapper86_copies3199 a h).2.2.1
  have c102371 := (publicBatchWrapper86_copies3199 a h).2.2.2.1
  have c102372 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.1
  have c102373 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.1
  have c102374 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_8 := arithEq_of_rows h (row := 1674) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_8
  simp only [← c102369, ← c102370, ← c102371] at e_1674_8
  have e_1674_9 := arithEq_of_rows h (row := 1674) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_9
  simp only [← c102372, ← c102373, k2, ← c102374] at e_1674_9
  have hr := e_1674_9
  simp only [e_1674_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16743)) := by
  have c102375 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.1
  have c102376 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.1
  have c102377 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.1
  have c102378 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102379 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102380 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_10 := arithEq_of_rows h (row := 1674) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_10
  simp only [← c102375, ← c102376, ← c102377] at e_1674_10
  have e_1674_11 := arithEq_of_rows h (row := 1674) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_11
  simp only [← c102378, ← c102379, k2, ← c102380] at e_1674_11
  have hr := e_1674_11
  simp only [e_1674_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16744)) := by
  have c102381 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102382 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102383 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102384 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102385 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102386 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_12 := arithEq_of_rows h (row := 1674) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_12
  simp only [← c102381, ← c102382, ← c102383] at e_1674_12
  have e_1674_13 := arithEq_of_rows h (row := 1674) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_13
  simp only [← c102384, ← c102385, k2, ← c102386] at e_1674_13
  have hr := e_1674_13
  simp only [e_1674_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16745)) := by
  have c102387 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102388 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102389 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102390 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102391 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102392 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_14 := arithEq_of_rows h (row := 1674) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_14
  simp only [← c102387, ← c102388, ← c102389] at e_1674_14
  have e_1674_15 := arithEq_of_rows h (row := 1674) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_15
  simp only [← c102390, ← c102391, k2, ← c102392] at e_1674_15
  have hr := e_1674_15
  simp only [e_1674_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16746)) := by
  have c102393 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102394 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102395 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102396 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102397 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102398 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_16 := arithEq_of_rows h (row := 1674) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_16
  simp only [← c102393, ← c102394, ← c102395] at e_1674_16
  have e_1674_17 := arithEq_of_rows h (row := 1674) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_17
  simp only [← c102396, ← c102397, k2, ← c102398] at e_1674_17
  have hr := e_1674_17
  simp only [e_1674_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1674 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16747)) := by
  have c102399 := (publicBatchWrapper86_copies3199 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102400 := (publicBatchWrapper86_copies3200 a h).1
  have c102401 := (publicBatchWrapper86_copies3200 a h).2.1
  have c102402 := (publicBatchWrapper86_copies3200 a h).2.2.1
  have c102403 := (publicBatchWrapper86_copies3200 a h).2.2.2.1
  have c102404 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1674_18 := arithEq_of_rows h (row := 1674) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_18
  simp only [← c102399, ← c102400, ← c102401] at e_1674_18
  have e_1674_19 := arithEq_of_rows h (row := 1674) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1674_19
  simp only [← c102402, ← c102403, k2, ← c102404] at e_1674_19
  have hr := e_1674_19
  simp only [e_1674_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16748)) := by
  have c102405 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.1
  have c102406 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.1
  have c102407 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.1
  have c102408 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.1
  have c102409 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.1
  have c102410 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_0 := arithEq_of_rows h (row := 1675) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_0
  simp only [← c102405, ← c102406, ← c102407] at e_1675_0
  have e_1675_1 := arithEq_of_rows h (row := 1675) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_1
  simp only [← c102408, ← c102409, k2, ← c102410] at e_1675_1
  have hr := e_1675_1
  simp only [e_1675_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16749)) := by
  have c102411 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102412 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102413 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102414 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102415 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102416 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_2 := arithEq_of_rows h (row := 1675) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_2
  simp only [← c102411, ← c102412, ← c102413] at e_1675_2
  have e_1675_3 := arithEq_of_rows h (row := 1675) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_3
  simp only [← c102414, ← c102415, k2, ← c102416] at e_1675_3
  have hr := e_1675_3
  simp only [e_1675_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16750)) := by
  have c102417 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102418 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102419 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102420 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102421 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102422 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_4 := arithEq_of_rows h (row := 1675) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_4
  simp only [← c102417, ← c102418, ← c102419] at e_1675_4
  have e_1675_5 := arithEq_of_rows h (row := 1675) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_5
  simp only [← c102420, ← c102421, k2, ← c102422] at e_1675_5
  have hr := e_1675_5
  simp only [e_1675_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16751)) := by
  have c102423 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102424 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102425 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102426 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102427 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102428 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_6 := arithEq_of_rows h (row := 1675) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_6
  simp only [← c102423, ← c102424, ← c102425] at e_1675_6
  have e_1675_7 := arithEq_of_rows h (row := 1675) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_7
  simp only [← c102426, ← c102427, k2, ← c102428] at e_1675_7
  have hr := e_1675_7
  simp only [e_1675_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16752)) := by
  have c102429 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102430 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102431 := (publicBatchWrapper86_copies3200 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102432 := (publicBatchWrapper86_copies3201 a h).1
  have c102433 := (publicBatchWrapper86_copies3201 a h).2.1
  have c102434 := (publicBatchWrapper86_copies3201 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_8 := arithEq_of_rows h (row := 1675) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_8
  simp only [← c102429, ← c102430, ← c102431] at e_1675_8
  have e_1675_9 := arithEq_of_rows h (row := 1675) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_9
  simp only [← c102432, ← c102433, k2, ← c102434] at e_1675_9
  have hr := e_1675_9
  simp only [e_1675_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16753)) := by
  have c102435 := (publicBatchWrapper86_copies3201 a h).2.2.2.1
  have c102436 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.1
  have c102437 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.1
  have c102438 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.1
  have c102439 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.1
  have c102440 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_10 := arithEq_of_rows h (row := 1675) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_10
  simp only [← c102435, ← c102436, ← c102437] at e_1675_10
  have e_1675_11 := arithEq_of_rows h (row := 1675) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_11
  simp only [← c102438, ← c102439, k2, ← c102440] at e_1675_11
  have hr := e_1675_11
  simp only [e_1675_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16754)) := by
  have c102441 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.1
  have c102442 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102443 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102444 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102445 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102446 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_12 := arithEq_of_rows h (row := 1675) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_12
  simp only [← c102441, ← c102442, ← c102443] at e_1675_12
  have e_1675_13 := arithEq_of_rows h (row := 1675) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_13
  simp only [← c102444, ← c102445, k2, ← c102446] at e_1675_13
  have hr := e_1675_13
  simp only [e_1675_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16755)) := by
  have c102447 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102448 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102449 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102450 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102451 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102452 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_14 := arithEq_of_rows h (row := 1675) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_14
  simp only [← c102447, ← c102448, ← c102449] at e_1675_14
  have e_1675_15 := arithEq_of_rows h (row := 1675) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_15
  simp only [← c102450, ← c102451, k2, ← c102452] at e_1675_15
  have hr := e_1675_15
  simp only [e_1675_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16756)) := by
  have c102453 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102454 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102455 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102456 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102457 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102458 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_16 := arithEq_of_rows h (row := 1675) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_16
  simp only [← c102453, ← c102454, ← c102455] at e_1675_16
  have e_1675_17 := arithEq_of_rows h (row := 1675) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_17
  simp only [← c102456, ← c102457, k2, ← c102458] at e_1675_17
  have hr := e_1675_17
  simp only [e_1675_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1675 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16757)) := by
  have c102459 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102460 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102461 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102462 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102463 := (publicBatchWrapper86_copies3201 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102464 := (publicBatchWrapper86_copies3202 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1675_18 := arithEq_of_rows h (row := 1675) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_18
  simp only [← c102459, ← c102460, ← c102461] at e_1675_18
  have e_1675_19 := arithEq_of_rows h (row := 1675) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1675_19
  simp only [← c102462, ← c102463, k2, ← c102464] at e_1675_19
  have hr := e_1675_19
  simp only [e_1675_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16758)) := by
  have c102465 := (publicBatchWrapper86_copies3202 a h).2.1
  have c102466 := (publicBatchWrapper86_copies3202 a h).2.2.1
  have c102467 := (publicBatchWrapper86_copies3202 a h).2.2.2.1
  have c102468 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.1
  have c102469 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.1
  have c102470 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_0 := arithEq_of_rows h (row := 1676) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_0
  simp only [← c102465, ← c102466, ← c102467] at e_1676_0
  have e_1676_1 := arithEq_of_rows h (row := 1676) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_1
  simp only [← c102468, ← c102469, k2, ← c102470] at e_1676_1
  have hr := e_1676_1
  simp only [e_1676_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16759)) := by
  have c102471 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.1
  have c102472 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.1
  have c102473 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.1
  have c102474 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102475 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102476 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_2 := arithEq_of_rows h (row := 1676) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_2
  simp only [← c102471, ← c102472, ← c102473] at e_1676_2
  have e_1676_3 := arithEq_of_rows h (row := 1676) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_3
  simp only [← c102474, ← c102475, k2, ← c102476] at e_1676_3
  have hr := e_1676_3
  simp only [e_1676_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 23) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16760)) := by
  have c102477 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102478 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102479 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102480 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102481 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102482 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_4 := arithEq_of_rows h (row := 1676) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_4
  simp only [← c102477, ← c102478, ← c102479] at e_1676_4
  have e_1676_5 := arithEq_of_rows h (row := 1676) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_5
  simp only [← c102480, ← c102481, k2, ← c102482] at e_1676_5
  have hr := e_1676_5
  simp only [e_1676_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 31) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16761)) := by
  have c102483 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102484 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102485 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102486 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102487 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102488 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_6 := arithEq_of_rows h (row := 1676) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_6
  simp only [← c102483, ← c102484, ← c102485] at e_1676_6
  have e_1676_7 := arithEq_of_rows h (row := 1676) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_7
  simp only [← c102486, ← c102487, k2, ← c102488] at e_1676_7
  have hr := e_1676_7
  simp only [e_1676_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 39) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16762)) := by
  have c102489 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102490 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102491 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102492 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102493 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102494 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_8 := arithEq_of_rows h (row := 1676) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_8
  simp only [← c102489, ← c102490, ← c102491] at e_1676_8
  have e_1676_9 := arithEq_of_rows h (row := 1676) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_9
  simp only [← c102492, ← c102493, k2, ← c102494] at e_1676_9
  have hr := e_1676_9
  simp only [e_1676_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 47) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16763)) := by
  have c102495 := (publicBatchWrapper86_copies3202 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102496 := (publicBatchWrapper86_copies3203 a h).1
  have c102497 := (publicBatchWrapper86_copies3203 a h).2.1
  have c102498 := (publicBatchWrapper86_copies3203 a h).2.2.1
  have c102499 := (publicBatchWrapper86_copies3203 a h).2.2.2.1
  have c102500 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_10 := arithEq_of_rows h (row := 1676) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_10
  simp only [← c102495, ← c102496, ← c102497] at e_1676_10
  have e_1676_11 := arithEq_of_rows h (row := 1676) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_11
  simp only [← c102498, ← c102499, k2, ← c102500] at e_1676_11
  have hr := e_1676_11
  simp only [e_1676_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 55) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16764)) := by
  have c102501 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.1
  have c102502 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.1
  have c102503 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.1
  have c102504 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.1
  have c102505 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.1
  have c102506 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_12 := arithEq_of_rows h (row := 1676) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_12
  simp only [← c102501, ← c102502, ← c102503] at e_1676_12
  have e_1676_13 := arithEq_of_rows h (row := 1676) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_13
  simp only [← c102504, ← c102505, k2, ← c102506] at e_1676_13
  have hr := e_1676_13
  simp only [e_1676_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 63) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16765)) := by
  have c102507 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102508 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102509 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102510 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102511 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102512 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_14 := arithEq_of_rows h (row := 1676) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_14
  simp only [← c102507, ← c102508, ← c102509] at e_1676_14
  have e_1676_15 := arithEq_of_rows h (row := 1676) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_15
  simp only [← c102510, ← c102511, k2, ← c102512] at e_1676_15
  have hr := e_1676_15
  simp only [e_1676_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 71) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16766)) := by
  have c102513 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102514 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102515 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102516 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102517 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102518 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_16 := arithEq_of_rows h (row := 1676) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_16
  simp only [← c102513, ← c102514, ← c102515] at e_1676_16
  have e_1676_17 := arithEq_of_rows h (row := 1676) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_17
  simp only [← c102516, ← c102517, k2, ← c102518] at e_1676_17
  have hr := e_1676_17
  simp only [e_1676_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1676 79) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16767)) := by
  have c102519 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102520 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102521 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102522 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102523 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102524 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1676_18 := arithEq_of_rows h (row := 1676) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_18
  simp only [← c102519, ← c102520, ← c102521] at e_1676_18
  have e_1676_19 := arithEq_of_rows h (row := 1676) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1676_19
  simp only [← c102522, ← c102523, k2, ← c102524] at e_1676_19
  have hr := e_1676_19
  simp only [e_1676_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 7) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16768)) := by
  have c102525 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102526 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102527 := (publicBatchWrapper86_copies3203 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102528 := (publicBatchWrapper86_copies3204 a h).1
  have c102529 := (publicBatchWrapper86_copies3204 a h).2.1
  have c102530 := (publicBatchWrapper86_copies3204 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_0 := arithEq_of_rows h (row := 1677) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_0
  simp only [← c102525, ← c102526, ← c102527] at e_1677_0
  have e_1677_1 := arithEq_of_rows h (row := 1677) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_1
  simp only [← c102528, ← c102529, k2, ← c102530] at e_1677_1
  have hr := e_1677_1
  simp only [e_1677_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 15) = bselect (a (.wire 80 79)) (a (.virt 200)) (a (.virt 16769)) := by
  have c102531 := (publicBatchWrapper86_copies3204 a h).2.2.2.1
  have c102532 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.1
  have c102533 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.1
  have c102534 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.1
  have c102535 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.1
  have c102536 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_2 := arithEq_of_rows h (row := 1677) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_2
  simp only [← c102531, ← c102532, ← c102533] at e_1677_2
  have e_1677_3 := arithEq_of_rows h (row := 1677) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_3
  simp only [← c102534, ← c102535, k2, ← c102536] at e_1677_3
  have hr := e_1677_3
  simp only [e_1677_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16810)) := by
  have c102537 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.1
  have c102538 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102539 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102540 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102541 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102542 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_4 := arithEq_of_rows h (row := 1677) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_4
  simp only [← c102537, ← c102538, ← c102539] at e_1677_4
  have e_1677_5 := arithEq_of_rows h (row := 1677) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_5
  simp only [← c102540, ← c102541, k2, ← c102542] at e_1677_5
  have hr := e_1677_5
  simp only [e_1677_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16811)) := by
  have c102543 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102544 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102545 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102546 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102547 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102548 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_6 := arithEq_of_rows h (row := 1677) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_6
  simp only [← c102543, ← c102544, ← c102545] at e_1677_6
  have e_1677_7 := arithEq_of_rows h (row := 1677) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_7
  simp only [← c102546, ← c102547, k2, ← c102548] at e_1677_7
  have hr := e_1677_7
  simp only [e_1677_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16812)) := by
  have c102549 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102550 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102551 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102552 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102553 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102554 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_8 := arithEq_of_rows h (row := 1677) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_8
  simp only [← c102549, ← c102550, ← c102551] at e_1677_8
  have e_1677_9 := arithEq_of_rows h (row := 1677) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_9
  simp only [← c102552, ← c102553, k2, ← c102554] at e_1677_9
  have hr := e_1677_9
  simp only [e_1677_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16813)) := by
  have c102555 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102556 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102557 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102558 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102559 := (publicBatchWrapper86_copies3204 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102560 := (publicBatchWrapper86_copies3205 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_10 := arithEq_of_rows h (row := 1677) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_10
  simp only [← c102555, ← c102556, ← c102557] at e_1677_10
  have e_1677_11 := arithEq_of_rows h (row := 1677) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_11
  simp only [← c102558, ← c102559, k2, ← c102560] at e_1677_11
  have hr := e_1677_11
  simp only [e_1677_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16814)) := by
  have c102561 := (publicBatchWrapper86_copies3205 a h).2.1
  have c102562 := (publicBatchWrapper86_copies3205 a h).2.2.1
  have c102563 := (publicBatchWrapper86_copies3205 a h).2.2.2.1
  have c102564 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.1
  have c102565 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.1
  have c102566 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_12 := arithEq_of_rows h (row := 1677) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_12
  simp only [← c102561, ← c102562, ← c102563] at e_1677_12
  have e_1677_13 := arithEq_of_rows h (row := 1677) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_13
  simp only [← c102564, ← c102565, k2, ← c102566] at e_1677_13
  have hr := e_1677_13
  simp only [e_1677_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16815)) := by
  have c102567 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.1
  have c102568 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.1
  have c102569 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.1
  have c102570 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102571 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102572 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_14 := arithEq_of_rows h (row := 1677) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_14
  simp only [← c102567, ← c102568, ← c102569] at e_1677_14
  have e_1677_15 := arithEq_of_rows h (row := 1677) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_15
  simp only [← c102570, ← c102571, k2, ← c102572] at e_1677_15
  have hr := e_1677_15
  simp only [e_1677_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16816)) := by
  have c102573 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102574 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102575 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102576 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102577 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102578 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_16 := arithEq_of_rows h (row := 1677) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_16
  simp only [← c102573, ← c102574, ← c102575] at e_1677_16
  have e_1677_17 := arithEq_of_rows h (row := 1677) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_17
  simp only [← c102576, ← c102577, k2, ← c102578] at e_1677_17
  have hr := e_1677_17
  simp only [e_1677_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1677 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16817)) := by
  have c102579 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102580 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102581 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102582 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102583 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102584 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1677_18 := arithEq_of_rows h (row := 1677) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_18
  simp only [← c102579, ← c102580, ← c102581] at e_1677_18
  have e_1677_19 := arithEq_of_rows h (row := 1677) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1677_19
  simp only [← c102582, ← c102583, k2, ← c102584] at e_1677_19
  have hr := e_1677_19
  simp only [e_1677_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16818)) := by
  have c102585 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102586 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102587 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102588 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102589 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102590 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_0 := arithEq_of_rows h (row := 1678) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_0
  simp only [← c102585, ← c102586, ← c102587] at e_1678_0
  have e_1678_1 := arithEq_of_rows h (row := 1678) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_1
  simp only [← c102588, ← c102589, k2, ← c102590] at e_1678_1
  have hr := e_1678_1
  simp only [e_1678_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16819)) := by
  have c102591 := (publicBatchWrapper86_copies3205 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102592 := (publicBatchWrapper86_copies3206 a h).1
  have c102593 := (publicBatchWrapper86_copies3206 a h).2.1
  have c102594 := (publicBatchWrapper86_copies3206 a h).2.2.1
  have c102595 := (publicBatchWrapper86_copies3206 a h).2.2.2.1
  have c102596 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_2 := arithEq_of_rows h (row := 1678) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_2
  simp only [← c102591, ← c102592, ← c102593] at e_1678_2
  have e_1678_3 := arithEq_of_rows h (row := 1678) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_3
  simp only [← c102594, ← c102595, k2, ← c102596] at e_1678_3
  have hr := e_1678_3
  simp only [e_1678_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16820)) := by
  have c102597 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.1
  have c102598 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.1
  have c102599 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.1
  have c102600 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.1
  have c102601 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.1
  have c102602 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_4 := arithEq_of_rows h (row := 1678) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_4
  simp only [← c102597, ← c102598, ← c102599] at e_1678_4
  have e_1678_5 := arithEq_of_rows h (row := 1678) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_5
  simp only [← c102600, ← c102601, k2, ← c102602] at e_1678_5
  have hr := e_1678_5
  simp only [e_1678_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16821)) := by
  have c102603 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102604 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102605 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102606 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102607 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102608 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_6 := arithEq_of_rows h (row := 1678) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_6
  simp only [← c102603, ← c102604, ← c102605] at e_1678_6
  have e_1678_7 := arithEq_of_rows h (row := 1678) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_7
  simp only [← c102606, ← c102607, k2, ← c102608] at e_1678_7
  have hr := e_1678_7
  simp only [e_1678_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16822)) := by
  have c102609 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102610 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102611 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102612 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102613 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102614 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_8 := arithEq_of_rows h (row := 1678) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_8
  simp only [← c102609, ← c102610, ← c102611] at e_1678_8
  have e_1678_9 := arithEq_of_rows h (row := 1678) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_9
  simp only [← c102612, ← c102613, k2, ← c102614] at e_1678_9
  have hr := e_1678_9
  simp only [e_1678_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16823)) := by
  have c102615 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102616 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102617 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102618 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102619 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102620 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_10 := arithEq_of_rows h (row := 1678) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_10
  simp only [← c102615, ← c102616, ← c102617] at e_1678_10
  have e_1678_11 := arithEq_of_rows h (row := 1678) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_11
  simp only [← c102618, ← c102619, k2, ← c102620] at e_1678_11
  have hr := e_1678_11
  simp only [e_1678_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16824)) := by
  have c102621 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102622 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102623 := (publicBatchWrapper86_copies3206 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102624 := (publicBatchWrapper86_copies3207 a h).1
  have c102625 := (publicBatchWrapper86_copies3207 a h).2.1
  have c102626 := (publicBatchWrapper86_copies3207 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_12 := arithEq_of_rows h (row := 1678) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_12
  simp only [← c102621, ← c102622, ← c102623] at e_1678_12
  have e_1678_13 := arithEq_of_rows h (row := 1678) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_13
  simp only [← c102624, ← c102625, k2, ← c102626] at e_1678_13
  have hr := e_1678_13
  simp only [e_1678_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16825)) := by
  have c102627 := (publicBatchWrapper86_copies3207 a h).2.2.2.1
  have c102628 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.1
  have c102629 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.1
  have c102630 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.1
  have c102631 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.1
  have c102632 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_14 := arithEq_of_rows h (row := 1678) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_14
  simp only [← c102627, ← c102628, ← c102629] at e_1678_14
  have e_1678_15 := arithEq_of_rows h (row := 1678) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_15
  simp only [← c102630, ← c102631, k2, ← c102632] at e_1678_15
  have hr := e_1678_15
  simp only [e_1678_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16826)) := by
  have c102633 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.1
  have c102634 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102635 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102636 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102637 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102638 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_16 := arithEq_of_rows h (row := 1678) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_16
  simp only [← c102633, ← c102634, ← c102635] at e_1678_16
  have e_1678_17 := arithEq_of_rows h (row := 1678) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_17
  simp only [← c102636, ← c102637, k2, ← c102638] at e_1678_17
  have hr := e_1678_17
  simp only [e_1678_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1678 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16827)) := by
  have c102639 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102640 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102641 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102642 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102643 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102644 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1678_18 := arithEq_of_rows h (row := 1678) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_18
  simp only [← c102639, ← c102640, ← c102641] at e_1678_18
  have e_1678_19 := arithEq_of_rows h (row := 1678) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1678_19
  simp only [← c102642, ← c102643, k2, ← c102644] at e_1678_19
  have hr := e_1678_19
  simp only [e_1678_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16828)) := by
  have c102645 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102646 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102647 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102648 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102649 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102650 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_0 := arithEq_of_rows h (row := 1679) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_0
  simp only [← c102645, ← c102646, ← c102647] at e_1679_0
  have e_1679_1 := arithEq_of_rows h (row := 1679) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_1
  simp only [← c102648, ← c102649, k2, ← c102650] at e_1679_1
  have hr := e_1679_1
  simp only [e_1679_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16829)) := by
  have c102651 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102652 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102653 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102654 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102655 := (publicBatchWrapper86_copies3207 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102656 := (publicBatchWrapper86_copies3208 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_2 := arithEq_of_rows h (row := 1679) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_2
  simp only [← c102651, ← c102652, ← c102653] at e_1679_2
  have e_1679_3 := arithEq_of_rows h (row := 1679) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_3
  simp only [← c102654, ← c102655, k2, ← c102656] at e_1679_3
  have hr := e_1679_3
  simp only [e_1679_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16830)) := by
  have c102657 := (publicBatchWrapper86_copies3208 a h).2.1
  have c102658 := (publicBatchWrapper86_copies3208 a h).2.2.1
  have c102659 := (publicBatchWrapper86_copies3208 a h).2.2.2.1
  have c102660 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.1
  have c102661 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.1
  have c102662 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_4 := arithEq_of_rows h (row := 1679) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_4
  simp only [← c102657, ← c102658, ← c102659] at e_1679_4
  have e_1679_5 := arithEq_of_rows h (row := 1679) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_5
  simp only [← c102660, ← c102661, k2, ← c102662] at e_1679_5
  have hr := e_1679_5
  simp only [e_1679_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16831)) := by
  have c102663 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.1
  have c102664 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.1
  have c102665 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.1
  have c102666 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102667 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102668 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_6 := arithEq_of_rows h (row := 1679) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_6
  simp only [← c102663, ← c102664, ← c102665] at e_1679_6
  have e_1679_7 := arithEq_of_rows h (row := 1679) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_7
  simp only [← c102666, ← c102667, k2, ← c102668] at e_1679_7
  have hr := e_1679_7
  simp only [e_1679_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16832)) := by
  have c102669 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102670 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102671 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102672 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102673 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102674 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_8 := arithEq_of_rows h (row := 1679) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_8
  simp only [← c102669, ← c102670, ← c102671] at e_1679_8
  have e_1679_9 := arithEq_of_rows h (row := 1679) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_9
  simp only [← c102672, ← c102673, k2, ← c102674] at e_1679_9
  have hr := e_1679_9
  simp only [e_1679_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16833)) := by
  have c102675 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102676 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102677 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102678 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102679 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102680 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_10 := arithEq_of_rows h (row := 1679) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_10
  simp only [← c102675, ← c102676, ← c102677] at e_1679_10
  have e_1679_11 := arithEq_of_rows h (row := 1679) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_11
  simp only [← c102678, ← c102679, k2, ← c102680] at e_1679_11
  have hr := e_1679_11
  simp only [e_1679_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16834)) := by
  have c102681 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102682 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102683 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102684 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102685 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102686 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_12 := arithEq_of_rows h (row := 1679) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_12
  simp only [← c102681, ← c102682, ← c102683] at e_1679_12
  have e_1679_13 := arithEq_of_rows h (row := 1679) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_13
  simp only [← c102684, ← c102685, k2, ← c102686] at e_1679_13
  have hr := e_1679_13
  simp only [e_1679_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16835)) := by
  have c102687 := (publicBatchWrapper86_copies3208 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102688 := (publicBatchWrapper86_copies3209 a h).1
  have c102689 := (publicBatchWrapper86_copies3209 a h).2.1
  have c102690 := (publicBatchWrapper86_copies3209 a h).2.2.1
  have c102691 := (publicBatchWrapper86_copies3209 a h).2.2.2.1
  have c102692 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_14 := arithEq_of_rows h (row := 1679) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_14
  simp only [← c102687, ← c102688, ← c102689] at e_1679_14
  have e_1679_15 := arithEq_of_rows h (row := 1679) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_15
  simp only [← c102690, ← c102691, k2, ← c102692] at e_1679_15
  have hr := e_1679_15
  simp only [e_1679_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16836)) := by
  have c102693 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.1
  have c102694 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.1
  have c102695 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.1
  have c102696 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.1
  have c102697 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.1
  have c102698 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_16 := arithEq_of_rows h (row := 1679) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_16
  simp only [← c102693, ← c102694, ← c102695] at e_1679_16
  have e_1679_17 := arithEq_of_rows h (row := 1679) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_17
  simp only [← c102696, ← c102697, k2, ← c102698] at e_1679_17
  have hr := e_1679_17
  simp only [e_1679_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1679 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16837)) := by
  have c102699 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102700 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102701 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102702 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102703 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102704 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1679_18 := arithEq_of_rows h (row := 1679) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_18
  simp only [← c102699, ← c102700, ← c102701] at e_1679_18
  have e_1679_19 := arithEq_of_rows h (row := 1679) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1679_19
  simp only [← c102702, ← c102703, k2, ← c102704] at e_1679_19
  have hr := e_1679_19
  simp only [e_1679_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16838)) := by
  have c102705 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102706 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102707 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102708 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102709 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102710 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_0 := arithEq_of_rows h (row := 1680) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_0
  simp only [← c102705, ← c102706, ← c102707] at e_1680_0
  have e_1680_1 := arithEq_of_rows h (row := 1680) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_1
  simp only [← c102708, ← c102709, k2, ← c102710] at e_1680_1
  have hr := e_1680_1
  simp only [e_1680_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16839)) := by
  have c102711 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102712 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102713 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102714 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102715 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102716 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_2 := arithEq_of_rows h (row := 1680) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_2
  simp only [← c102711, ← c102712, ← c102713] at e_1680_2
  have e_1680_3 := arithEq_of_rows h (row := 1680) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_3
  simp only [← c102714, ← c102715, k2, ← c102716] at e_1680_3
  have hr := e_1680_3
  simp only [e_1680_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16840)) := by
  have c102717 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102718 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102719 := (publicBatchWrapper86_copies3209 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102720 := (publicBatchWrapper86_copies3210 a h).1
  have c102721 := (publicBatchWrapper86_copies3210 a h).2.1
  have c102722 := (publicBatchWrapper86_copies3210 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_4 := arithEq_of_rows h (row := 1680) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_4
  simp only [← c102717, ← c102718, ← c102719] at e_1680_4
  have e_1680_5 := arithEq_of_rows h (row := 1680) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_5
  simp only [← c102720, ← c102721, k2, ← c102722] at e_1680_5
  have hr := e_1680_5
  simp only [e_1680_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16841)) := by
  have c102723 := (publicBatchWrapper86_copies3210 a h).2.2.2.1
  have c102724 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.1
  have c102725 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.1
  have c102726 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.1
  have c102727 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.1
  have c102728 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_6 := arithEq_of_rows h (row := 1680) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_6
  simp only [← c102723, ← c102724, ← c102725] at e_1680_6
  have e_1680_7 := arithEq_of_rows h (row := 1680) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_7
  simp only [← c102726, ← c102727, k2, ← c102728] at e_1680_7
  have hr := e_1680_7
  simp only [e_1680_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16842)) := by
  have c102729 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.1
  have c102730 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102731 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102732 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102733 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102734 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_8 := arithEq_of_rows h (row := 1680) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_8
  simp only [← c102729, ← c102730, ← c102731] at e_1680_8
  have e_1680_9 := arithEq_of_rows h (row := 1680) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_9
  simp only [← c102732, ← c102733, k2, ← c102734] at e_1680_9
  have hr := e_1680_9
  simp only [e_1680_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16843)) := by
  have c102735 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102736 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102737 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102738 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102739 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102740 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_10 := arithEq_of_rows h (row := 1680) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_10
  simp only [← c102735, ← c102736, ← c102737] at e_1680_10
  have e_1680_11 := arithEq_of_rows h (row := 1680) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_11
  simp only [← c102738, ← c102739, k2, ← c102740] at e_1680_11
  have hr := e_1680_11
  simp only [e_1680_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16844)) := by
  have c102741 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102742 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102743 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102744 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102745 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102746 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_12 := arithEq_of_rows h (row := 1680) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_12
  simp only [← c102741, ← c102742, ← c102743] at e_1680_12
  have e_1680_13 := arithEq_of_rows h (row := 1680) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_13
  simp only [← c102744, ← c102745, k2, ← c102746] at e_1680_13
  have hr := e_1680_13
  simp only [e_1680_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16845)) := by
  have c102747 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102748 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102749 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102750 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102751 := (publicBatchWrapper86_copies3210 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102752 := (publicBatchWrapper86_copies3211 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_14 := arithEq_of_rows h (row := 1680) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_14
  simp only [← c102747, ← c102748, ← c102749] at e_1680_14
  have e_1680_15 := arithEq_of_rows h (row := 1680) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_15
  simp only [← c102750, ← c102751, k2, ← c102752] at e_1680_15
  have hr := e_1680_15
  simp only [e_1680_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16846)) := by
  have c102753 := (publicBatchWrapper86_copies3211 a h).2.1
  have c102754 := (publicBatchWrapper86_copies3211 a h).2.2.1
  have c102755 := (publicBatchWrapper86_copies3211 a h).2.2.2.1
  have c102756 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.1
  have c102757 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.1
  have c102758 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_16 := arithEq_of_rows h (row := 1680) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_16
  simp only [← c102753, ← c102754, ← c102755] at e_1680_16
  have e_1680_17 := arithEq_of_rows h (row := 1680) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_17
  simp only [← c102756, ← c102757, k2, ← c102758] at e_1680_17
  have hr := e_1680_17
  simp only [e_1680_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1680 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16847)) := by
  have c102759 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.1
  have c102760 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.1
  have c102761 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.1
  have c102762 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102763 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102764 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1680_18 := arithEq_of_rows h (row := 1680) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_18
  simp only [← c102759, ← c102760, ← c102761] at e_1680_18
  have e_1680_19 := arithEq_of_rows h (row := 1680) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1680_19
  simp only [← c102762, ← c102763, k2, ← c102764] at e_1680_19
  have hr := e_1680_19
  simp only [e_1680_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16848)) := by
  have c102765 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102766 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102767 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102768 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102769 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102770 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_0 := arithEq_of_rows h (row := 1681) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_0
  simp only [← c102765, ← c102766, ← c102767] at e_1681_0
  have e_1681_1 := arithEq_of_rows h (row := 1681) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_1
  simp only [← c102768, ← c102769, k2, ← c102770] at e_1681_1
  have hr := e_1681_1
  simp only [e_1681_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16849)) := by
  have c102771 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102772 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102773 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102774 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102775 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102776 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_2 := arithEq_of_rows h (row := 1681) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_2
  simp only [← c102771, ← c102772, ← c102773] at e_1681_2
  have e_1681_3 := arithEq_of_rows h (row := 1681) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_3
  simp only [← c102774, ← c102775, k2, ← c102776] at e_1681_3
  have hr := e_1681_3
  simp only [e_1681_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16850)) := by
  have c102777 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102778 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102779 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102780 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102781 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102782 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_4 := arithEq_of_rows h (row := 1681) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_4
  simp only [← c102777, ← c102778, ← c102779] at e_1681_4
  have e_1681_5 := arithEq_of_rows h (row := 1681) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_5
  simp only [← c102780, ← c102781, k2, ← c102782] at e_1681_5
  have hr := e_1681_5
  simp only [e_1681_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16851)) := by
  have c102783 := (publicBatchWrapper86_copies3211 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102784 := (publicBatchWrapper86_copies3212 a h).1
  have c102785 := (publicBatchWrapper86_copies3212 a h).2.1
  have c102786 := (publicBatchWrapper86_copies3212 a h).2.2.1
  have c102787 := (publicBatchWrapper86_copies3212 a h).2.2.2.1
  have c102788 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_6 := arithEq_of_rows h (row := 1681) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_6
  simp only [← c102783, ← c102784, ← c102785] at e_1681_6
  have e_1681_7 := arithEq_of_rows h (row := 1681) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_7
  simp only [← c102786, ← c102787, k2, ← c102788] at e_1681_7
  have hr := e_1681_7
  simp only [e_1681_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16852)) := by
  have c102789 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.1
  have c102790 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.1
  have c102791 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.1
  have c102792 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.1
  have c102793 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.1
  have c102794 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_8 := arithEq_of_rows h (row := 1681) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_8
  simp only [← c102789, ← c102790, ← c102791] at e_1681_8
  have e_1681_9 := arithEq_of_rows h (row := 1681) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_9
  simp only [← c102792, ← c102793, k2, ← c102794] at e_1681_9
  have hr := e_1681_9
  simp only [e_1681_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16853)) := by
  have c102795 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102796 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102797 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102798 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102799 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102800 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_10 := arithEq_of_rows h (row := 1681) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_10
  simp only [← c102795, ← c102796, ← c102797] at e_1681_10
  have e_1681_11 := arithEq_of_rows h (row := 1681) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_11
  simp only [← c102798, ← c102799, k2, ← c102800] at e_1681_11
  have hr := e_1681_11
  simp only [e_1681_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16854)) := by
  have c102801 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102802 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102803 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102804 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102805 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102806 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_12 := arithEq_of_rows h (row := 1681) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_12
  simp only [← c102801, ← c102802, ← c102803] at e_1681_12
  have e_1681_13 := arithEq_of_rows h (row := 1681) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_13
  simp only [← c102804, ← c102805, k2, ← c102806] at e_1681_13
  have hr := e_1681_13
  simp only [e_1681_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16855)) := by
  have c102807 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102808 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102809 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102810 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102811 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102812 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_14 := arithEq_of_rows h (row := 1681) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_14
  simp only [← c102807, ← c102808, ← c102809] at e_1681_14
  have e_1681_15 := arithEq_of_rows h (row := 1681) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_15
  simp only [← c102810, ← c102811, k2, ← c102812] at e_1681_15
  have hr := e_1681_15
  simp only [e_1681_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16856)) := by
  have c102813 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102814 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102815 := (publicBatchWrapper86_copies3212 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102816 := (publicBatchWrapper86_copies3213 a h).1
  have c102817 := (publicBatchWrapper86_copies3213 a h).2.1
  have c102818 := (publicBatchWrapper86_copies3213 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_16 := arithEq_of_rows h (row := 1681) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_16
  simp only [← c102813, ← c102814, ← c102815] at e_1681_16
  have e_1681_17 := arithEq_of_rows h (row := 1681) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_17
  simp only [← c102816, ← c102817, k2, ← c102818] at e_1681_17
  have hr := e_1681_17
  simp only [e_1681_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1681 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16857)) := by
  have c102819 := (publicBatchWrapper86_copies3213 a h).2.2.2.1
  have c102820 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.1
  have c102821 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.1
  have c102822 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.1
  have c102823 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.1
  have c102824 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1681_18 := arithEq_of_rows h (row := 1681) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_18
  simp only [← c102819, ← c102820, ← c102821] at e_1681_18
  have e_1681_19 := arithEq_of_rows h (row := 1681) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1681_19
  simp only [← c102822, ← c102823, k2, ← c102824] at e_1681_19
  have hr := e_1681_19
  simp only [e_1681_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16858)) := by
  have c102825 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.1
  have c102826 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102827 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102828 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102829 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102830 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_0 := arithEq_of_rows h (row := 1682) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_0
  simp only [← c102825, ← c102826, ← c102827] at e_1682_0
  have e_1682_1 := arithEq_of_rows h (row := 1682) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_1
  simp only [← c102828, ← c102829, k2, ← c102830] at e_1682_1
  have hr := e_1682_1
  simp only [e_1682_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16859)) := by
  have c102831 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102832 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102833 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102834 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102835 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102836 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_2 := arithEq_of_rows h (row := 1682) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_2
  simp only [← c102831, ← c102832, ← c102833] at e_1682_2
  have e_1682_3 := arithEq_of_rows h (row := 1682) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_3
  simp only [← c102834, ← c102835, k2, ← c102836] at e_1682_3
  have hr := e_1682_3
  simp only [e_1682_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16860)) := by
  have c102837 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102838 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102839 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102840 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102841 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102842 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_4 := arithEq_of_rows h (row := 1682) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_4
  simp only [← c102837, ← c102838, ← c102839] at e_1682_4
  have e_1682_5 := arithEq_of_rows h (row := 1682) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_5
  simp only [← c102840, ← c102841, k2, ← c102842] at e_1682_5
  have hr := e_1682_5
  simp only [e_1682_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16861)) := by
  have c102843 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102844 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102845 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102846 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102847 := (publicBatchWrapper86_copies3213 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102848 := (publicBatchWrapper86_copies3214 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_6 := arithEq_of_rows h (row := 1682) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_6
  simp only [← c102843, ← c102844, ← c102845] at e_1682_6
  have e_1682_7 := arithEq_of_rows h (row := 1682) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_7
  simp only [← c102846, ← c102847, k2, ← c102848] at e_1682_7
  have hr := e_1682_7
  simp only [e_1682_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16862)) := by
  have c102849 := (publicBatchWrapper86_copies3214 a h).2.1
  have c102850 := (publicBatchWrapper86_copies3214 a h).2.2.1
  have c102851 := (publicBatchWrapper86_copies3214 a h).2.2.2.1
  have c102852 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.1
  have c102853 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.1
  have c102854 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_8 := arithEq_of_rows h (row := 1682) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_8
  simp only [← c102849, ← c102850, ← c102851] at e_1682_8
  have e_1682_9 := arithEq_of_rows h (row := 1682) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_9
  simp only [← c102852, ← c102853, k2, ← c102854] at e_1682_9
  have hr := e_1682_9
  simp only [e_1682_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16863)) := by
  have c102855 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.1
  have c102856 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.1
  have c102857 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.1
  have c102858 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102859 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102860 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_10 := arithEq_of_rows h (row := 1682) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_10
  simp only [← c102855, ← c102856, ← c102857] at e_1682_10
  have e_1682_11 := arithEq_of_rows h (row := 1682) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_11
  simp only [← c102858, ← c102859, k2, ← c102860] at e_1682_11
  have hr := e_1682_11
  simp only [e_1682_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16864)) := by
  have c102861 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102862 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102863 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102864 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102865 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102866 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_12 := arithEq_of_rows h (row := 1682) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_12
  simp only [← c102861, ← c102862, ← c102863] at e_1682_12
  have e_1682_13 := arithEq_of_rows h (row := 1682) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_13
  simp only [← c102864, ← c102865, k2, ← c102866] at e_1682_13
  have hr := e_1682_13
  simp only [e_1682_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16865)) := by
  have c102867 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102868 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102869 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102870 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102871 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102872 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_14 := arithEq_of_rows h (row := 1682) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_14
  simp only [← c102867, ← c102868, ← c102869] at e_1682_14
  have e_1682_15 := arithEq_of_rows h (row := 1682) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_15
  simp only [← c102870, ← c102871, k2, ← c102872] at e_1682_15
  have hr := e_1682_15
  simp only [e_1682_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16866)) := by
  have c102873 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102874 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102875 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102876 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102877 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102878 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_16 := arithEq_of_rows h (row := 1682) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_16
  simp only [← c102873, ← c102874, ← c102875] at e_1682_16
  have e_1682_17 := arithEq_of_rows h (row := 1682) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_17
  simp only [← c102876, ← c102877, k2, ← c102878] at e_1682_17
  have hr := e_1682_17
  simp only [e_1682_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1682 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16867)) := by
  have c102879 := (publicBatchWrapper86_copies3214 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102880 := (publicBatchWrapper86_copies3215 a h).1
  have c102881 := (publicBatchWrapper86_copies3215 a h).2.1
  have c102882 := (publicBatchWrapper86_copies3215 a h).2.2.1
  have c102883 := (publicBatchWrapper86_copies3215 a h).2.2.2.1
  have c102884 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1682_18 := arithEq_of_rows h (row := 1682) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_18
  simp only [← c102879, ← c102880, ← c102881] at e_1682_18
  have e_1682_19 := arithEq_of_rows h (row := 1682) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1682_19
  simp only [← c102882, ← c102883, k2, ← c102884] at e_1682_19
  have hr := e_1682_19
  simp only [e_1682_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16868)) := by
  have c102885 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.1
  have c102886 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.1
  have c102887 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.1
  have c102888 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.1
  have c102889 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.1
  have c102890 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_0 := arithEq_of_rows h (row := 1683) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_0
  simp only [← c102885, ← c102886, ← c102887] at e_1683_0
  have e_1683_1 := arithEq_of_rows h (row := 1683) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_1
  simp only [← c102888, ← c102889, k2, ← c102890] at e_1683_1
  have hr := e_1683_1
  simp only [e_1683_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16869)) := by
  have c102891 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102892 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102893 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102894 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102895 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102896 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_2 := arithEq_of_rows h (row := 1683) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_2
  simp only [← c102891, ← c102892, ← c102893] at e_1683_2
  have e_1683_3 := arithEq_of_rows h (row := 1683) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_3
  simp only [← c102894, ← c102895, k2, ← c102896] at e_1683_3
  have hr := e_1683_3
  simp only [e_1683_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16870)) := by
  have c102897 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102898 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102899 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102900 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102901 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102902 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_4 := arithEq_of_rows h (row := 1683) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_4
  simp only [← c102897, ← c102898, ← c102899] at e_1683_4
  have e_1683_5 := arithEq_of_rows h (row := 1683) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_5
  simp only [← c102900, ← c102901, k2, ← c102902] at e_1683_5
  have hr := e_1683_5
  simp only [e_1683_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16871)) := by
  have c102903 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102904 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102905 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102906 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102907 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102908 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_6 := arithEq_of_rows h (row := 1683) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_6
  simp only [← c102903, ← c102904, ← c102905] at e_1683_6
  have e_1683_7 := arithEq_of_rows h (row := 1683) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_7
  simp only [← c102906, ← c102907, k2, ← c102908] at e_1683_7
  have hr := e_1683_7
  simp only [e_1683_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16872)) := by
  have c102909 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102910 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102911 := (publicBatchWrapper86_copies3215 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102912 := (publicBatchWrapper86_copies3216 a h).1
  have c102913 := (publicBatchWrapper86_copies3216 a h).2.1
  have c102914 := (publicBatchWrapper86_copies3216 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_8 := arithEq_of_rows h (row := 1683) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_8
  simp only [← c102909, ← c102910, ← c102911] at e_1683_8
  have e_1683_9 := arithEq_of_rows h (row := 1683) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_9
  simp only [← c102912, ← c102913, k2, ← c102914] at e_1683_9
  have hr := e_1683_9
  simp only [e_1683_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16873)) := by
  have c102915 := (publicBatchWrapper86_copies3216 a h).2.2.2.1
  have c102916 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.1
  have c102917 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.1
  have c102918 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.1
  have c102919 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.1
  have c102920 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_10 := arithEq_of_rows h (row := 1683) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_10
  simp only [← c102915, ← c102916, ← c102917] at e_1683_10
  have e_1683_11 := arithEq_of_rows h (row := 1683) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_11
  simp only [← c102918, ← c102919, k2, ← c102920] at e_1683_11
  have hr := e_1683_11
  simp only [e_1683_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16874)) := by
  have c102921 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.1
  have c102922 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102923 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102924 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102925 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102926 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_12 := arithEq_of_rows h (row := 1683) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_12
  simp only [← c102921, ← c102922, ← c102923] at e_1683_12
  have e_1683_13 := arithEq_of_rows h (row := 1683) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_13
  simp only [← c102924, ← c102925, k2, ← c102926] at e_1683_13
  have hr := e_1683_13
  simp only [e_1683_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16875)) := by
  have c102927 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102928 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102929 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102930 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102931 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102932 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_14 := arithEq_of_rows h (row := 1683) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_14
  simp only [← c102927, ← c102928, ← c102929] at e_1683_14
  have e_1683_15 := arithEq_of_rows h (row := 1683) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_15
  simp only [← c102930, ← c102931, k2, ← c102932] at e_1683_15
  have hr := e_1683_15
  simp only [e_1683_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16876)) := by
  have c102933 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102934 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102935 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102936 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102937 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102938 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_16 := arithEq_of_rows h (row := 1683) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_16
  simp only [← c102933, ← c102934, ← c102935] at e_1683_16
  have e_1683_17 := arithEq_of_rows h (row := 1683) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_17
  simp only [← c102936, ← c102937, k2, ← c102938] at e_1683_17
  have hr := e_1683_17
  simp only [e_1683_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1683 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16877)) := by
  have c102939 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102940 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102941 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102942 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102943 := (publicBatchWrapper86_copies3216 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102944 := (publicBatchWrapper86_copies3217 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1683_18 := arithEq_of_rows h (row := 1683) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_18
  simp only [← c102939, ← c102940, ← c102941] at e_1683_18
  have e_1683_19 := arithEq_of_rows h (row := 1683) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1683_19
  simp only [← c102942, ← c102943, k2, ← c102944] at e_1683_19
  have hr := e_1683_19
  simp only [e_1683_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16878)) := by
  have c102945 := (publicBatchWrapper86_copies3217 a h).2.1
  have c102946 := (publicBatchWrapper86_copies3217 a h).2.2.1
  have c102947 := (publicBatchWrapper86_copies3217 a h).2.2.2.1
  have c102948 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.1
  have c102949 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.1
  have c102950 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_0 := arithEq_of_rows h (row := 1684) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_0
  simp only [← c102945, ← c102946, ← c102947] at e_1684_0
  have e_1684_1 := arithEq_of_rows h (row := 1684) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_1
  simp only [← c102948, ← c102949, k2, ← c102950] at e_1684_1
  have hr := e_1684_1
  simp only [e_1684_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16879)) := by
  have c102951 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.1
  have c102952 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.1
  have c102953 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.1
  have c102954 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.1
  have c102955 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102956 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_2 := arithEq_of_rows h (row := 1684) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_2
  simp only [← c102951, ← c102952, ← c102953] at e_1684_2
  have e_1684_3 := arithEq_of_rows h (row := 1684) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_3
  simp only [← c102954, ← c102955, k2, ← c102956] at e_1684_3
  have hr := e_1684_3
  simp only [e_1684_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16880)) := by
  have c102957 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102958 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102959 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102960 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102961 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102962 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_4 := arithEq_of_rows h (row := 1684) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_4
  simp only [← c102957, ← c102958, ← c102959] at e_1684_4
  have e_1684_5 := arithEq_of_rows h (row := 1684) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_5
  simp only [← c102960, ← c102961, k2, ← c102962] at e_1684_5
  have hr := e_1684_5
  simp only [e_1684_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16881)) := by
  have c102963 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102964 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102965 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102966 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102967 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102968 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_6 := arithEq_of_rows h (row := 1684) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_6
  simp only [← c102963, ← c102964, ← c102965] at e_1684_6
  have e_1684_7 := arithEq_of_rows h (row := 1684) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_7
  simp only [← c102966, ← c102967, k2, ← c102968] at e_1684_7
  have hr := e_1684_7
  simp only [e_1684_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16882)) := by
  have c102969 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102970 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102971 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102972 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102973 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102974 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_8 := arithEq_of_rows h (row := 1684) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_8
  simp only [← c102969, ← c102970, ← c102971] at e_1684_8
  have e_1684_9 := arithEq_of_rows h (row := 1684) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_9
  simp only [← c102972, ← c102973, k2, ← c102974] at e_1684_9
  have hr := e_1684_9
  simp only [e_1684_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16883)) := by
  have c102975 := (publicBatchWrapper86_copies3217 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c102976 := (publicBatchWrapper86_copies3218 a h).1
  have c102977 := (publicBatchWrapper86_copies3218 a h).2.1
  have c102978 := (publicBatchWrapper86_copies3218 a h).2.2.1
  have c102979 := (publicBatchWrapper86_copies3218 a h).2.2.2.1
  have c102980 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_10 := arithEq_of_rows h (row := 1684) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_10
  simp only [← c102975, ← c102976, ← c102977] at e_1684_10
  have e_1684_11 := arithEq_of_rows h (row := 1684) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_11
  simp only [← c102978, ← c102979, k2, ← c102980] at e_1684_11
  have hr := e_1684_11
  simp only [e_1684_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16884)) := by
  have c102981 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.1
  have c102982 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.1
  have c102983 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.1
  have c102984 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.1
  have c102985 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.1
  have c102986 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_12 := arithEq_of_rows h (row := 1684) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_12
  simp only [← c102981, ← c102982, ← c102983] at e_1684_12
  have e_1684_13 := arithEq_of_rows h (row := 1684) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_13
  simp only [← c102984, ← c102985, k2, ← c102986] at e_1684_13
  have hr := e_1684_13
  simp only [e_1684_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16885)) := by
  have c102987 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c102988 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102989 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102990 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102991 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102992 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_14 := arithEq_of_rows h (row := 1684) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_14
  simp only [← c102987, ← c102988, ← c102989] at e_1684_14
  have e_1684_15 := arithEq_of_rows h (row := 1684) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_15
  simp only [← c102990, ← c102991, k2, ← c102992] at e_1684_15
  have hr := e_1684_15
  simp only [e_1684_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16886)) := by
  have c102993 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102994 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102995 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102996 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102997 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c102998 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_16 := arithEq_of_rows h (row := 1684) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_16
  simp only [← c102993, ← c102994, ← c102995] at e_1684_16
  have e_1684_17 := arithEq_of_rows h (row := 1684) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_17
  simp only [← c102996, ← c102997, k2, ← c102998] at e_1684_17
  have hr := e_1684_17
  simp only [e_1684_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1684 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16887)) := by
  have c102999 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103000 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103001 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103002 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103003 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103004 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1684_18 := arithEq_of_rows h (row := 1684) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_18
  simp only [← c102999, ← c103000, ← c103001] at e_1684_18
  have e_1684_19 := arithEq_of_rows h (row := 1684) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1684_19
  simp only [← c103002, ← c103003, k2, ← c103004] at e_1684_19
  have hr := e_1684_19
  simp only [e_1684_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16888)) := by
  have c103005 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103006 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103007 := (publicBatchWrapper86_copies3218 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103008 := (publicBatchWrapper86_copies3219 a h).1
  have c103009 := (publicBatchWrapper86_copies3219 a h).2.1
  have c103010 := (publicBatchWrapper86_copies3219 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_0 := arithEq_of_rows h (row := 1685) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_0
  simp only [← c103005, ← c103006, ← c103007] at e_1685_0
  have e_1685_1 := arithEq_of_rows h (row := 1685) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_1
  simp only [← c103008, ← c103009, k2, ← c103010] at e_1685_1
  have hr := e_1685_1
  simp only [e_1685_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16889)) := by
  have c103011 := (publicBatchWrapper86_copies3219 a h).2.2.2.1
  have c103012 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.1
  have c103013 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.1
  have c103014 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.1
  have c103015 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.1
  have c103016 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_2 := arithEq_of_rows h (row := 1685) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_2
  simp only [← c103011, ← c103012, ← c103013] at e_1685_2
  have e_1685_3 := arithEq_of_rows h (row := 1685) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_3
  simp only [← c103014, ← c103015, k2, ← c103016] at e_1685_3
  have hr := e_1685_3
  simp only [e_1685_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16890)) := by
  have c103017 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.1
  have c103018 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103019 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103020 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103021 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103022 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_4 := arithEq_of_rows h (row := 1685) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_4
  simp only [← c103017, ← c103018, ← c103019] at e_1685_4
  have e_1685_5 := arithEq_of_rows h (row := 1685) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_5
  simp only [← c103020, ← c103021, k2, ← c103022] at e_1685_5
  have hr := e_1685_5
  simp only [e_1685_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16891)) := by
  have c103023 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103024 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103025 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103026 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103027 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103028 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_6 := arithEq_of_rows h (row := 1685) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_6
  simp only [← c103023, ← c103024, ← c103025] at e_1685_6
  have e_1685_7 := arithEq_of_rows h (row := 1685) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_7
  simp only [← c103026, ← c103027, k2, ← c103028] at e_1685_7
  have hr := e_1685_7
  simp only [e_1685_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16892)) := by
  have c103029 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103030 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103031 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103032 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103033 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103034 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_8 := arithEq_of_rows h (row := 1685) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_8
  simp only [← c103029, ← c103030, ← c103031] at e_1685_8
  have e_1685_9 := arithEq_of_rows h (row := 1685) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_9
  simp only [← c103032, ← c103033, k2, ← c103034] at e_1685_9
  have hr := e_1685_9
  simp only [e_1685_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16893)) := by
  have c103035 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103036 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103037 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103038 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103039 := (publicBatchWrapper86_copies3219 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103040 := (publicBatchWrapper86_copies3220 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_10 := arithEq_of_rows h (row := 1685) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_10
  simp only [← c103035, ← c103036, ← c103037] at e_1685_10
  have e_1685_11 := arithEq_of_rows h (row := 1685) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_11
  simp only [← c103038, ← c103039, k2, ← c103040] at e_1685_11
  have hr := e_1685_11
  simp only [e_1685_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16894)) := by
  have c103041 := (publicBatchWrapper86_copies3220 a h).2.1
  have c103042 := (publicBatchWrapper86_copies3220 a h).2.2.1
  have c103043 := (publicBatchWrapper86_copies3220 a h).2.2.2.1
  have c103044 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.1
  have c103045 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.1
  have c103046 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_12 := arithEq_of_rows h (row := 1685) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_12
  simp only [← c103041, ← c103042, ← c103043] at e_1685_12
  have e_1685_13 := arithEq_of_rows h (row := 1685) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_13
  simp only [← c103044, ← c103045, k2, ← c103046] at e_1685_13
  have hr := e_1685_13
  simp only [e_1685_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16895)) := by
  have c103047 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.1
  have c103048 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.1
  have c103049 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.1
  have c103050 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103051 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103052 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_14 := arithEq_of_rows h (row := 1685) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_14
  simp only [← c103047, ← c103048, ← c103049] at e_1685_14
  have e_1685_15 := arithEq_of_rows h (row := 1685) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_15
  simp only [← c103050, ← c103051, k2, ← c103052] at e_1685_15
  have hr := e_1685_15
  simp only [e_1685_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16896)) := by
  have c103053 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103054 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103055 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103056 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103057 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103058 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_16 := arithEq_of_rows h (row := 1685) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_16
  simp only [← c103053, ← c103054, ← c103055] at e_1685_16
  have e_1685_17 := arithEq_of_rows h (row := 1685) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_17
  simp only [← c103056, ← c103057, k2, ← c103058] at e_1685_17
  have hr := e_1685_17
  simp only [e_1685_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1685 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16897)) := by
  have c103059 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103060 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103061 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103062 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103063 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103064 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1685_18 := arithEq_of_rows h (row := 1685) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_18
  simp only [← c103059, ← c103060, ← c103061] at e_1685_18
  have e_1685_19 := arithEq_of_rows h (row := 1685) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1685_19
  simp only [← c103062, ← c103063, k2, ← c103064] at e_1685_19
  have hr := e_1685_19
  simp only [e_1685_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16898)) := by
  have c103065 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103066 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103067 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103068 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103069 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103070 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_0 := arithEq_of_rows h (row := 1686) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_0
  simp only [← c103065, ← c103066, ← c103067] at e_1686_0
  have e_1686_1 := arithEq_of_rows h (row := 1686) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_1
  simp only [← c103068, ← c103069, k2, ← c103070] at e_1686_1
  have hr := e_1686_1
  simp only [e_1686_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16899)) := by
  have c103071 := (publicBatchWrapper86_copies3220 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103072 := (publicBatchWrapper86_copies3221 a h).1
  have c103073 := (publicBatchWrapper86_copies3221 a h).2.1
  have c103074 := (publicBatchWrapper86_copies3221 a h).2.2.1
  have c103075 := (publicBatchWrapper86_copies3221 a h).2.2.2.1
  have c103076 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_2 := arithEq_of_rows h (row := 1686) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_2
  simp only [← c103071, ← c103072, ← c103073] at e_1686_2
  have e_1686_3 := arithEq_of_rows h (row := 1686) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_3
  simp only [← c103074, ← c103075, k2, ← c103076] at e_1686_3
  have hr := e_1686_3
  simp only [e_1686_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16900)) := by
  have c103077 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.1
  have c103078 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.1
  have c103079 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.1
  have c103080 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.1
  have c103081 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.1
  have c103082 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_4 := arithEq_of_rows h (row := 1686) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_4
  simp only [← c103077, ← c103078, ← c103079] at e_1686_4
  have e_1686_5 := arithEq_of_rows h (row := 1686) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_5
  simp only [← c103080, ← c103081, k2, ← c103082] at e_1686_5
  have hr := e_1686_5
  simp only [e_1686_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16901)) := by
  have c103083 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103084 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103085 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103086 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103087 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103088 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_6 := arithEq_of_rows h (row := 1686) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_6
  simp only [← c103083, ← c103084, ← c103085] at e_1686_6
  have e_1686_7 := arithEq_of_rows h (row := 1686) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_7
  simp only [← c103086, ← c103087, k2, ← c103088] at e_1686_7
  have hr := e_1686_7
  simp only [e_1686_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16902)) := by
  have c103089 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103090 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103091 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103092 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103093 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103094 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_8 := arithEq_of_rows h (row := 1686) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_8
  simp only [← c103089, ← c103090, ← c103091] at e_1686_8
  have e_1686_9 := arithEq_of_rows h (row := 1686) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_9
  simp only [← c103092, ← c103093, k2, ← c103094] at e_1686_9
  have hr := e_1686_9
  simp only [e_1686_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16903)) := by
  have c103095 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103096 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103097 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103098 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103099 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103100 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_10 := arithEq_of_rows h (row := 1686) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_10
  simp only [← c103095, ← c103096, ← c103097] at e_1686_10
  have e_1686_11 := arithEq_of_rows h (row := 1686) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_11
  simp only [← c103098, ← c103099, k2, ← c103100] at e_1686_11
  have hr := e_1686_11
  simp only [e_1686_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16904)) := by
  have c103101 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103102 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103103 := (publicBatchWrapper86_copies3221 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103104 := (publicBatchWrapper86_copies3222 a h).1
  have c103105 := (publicBatchWrapper86_copies3222 a h).2.1
  have c103106 := (publicBatchWrapper86_copies3222 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_12 := arithEq_of_rows h (row := 1686) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_12
  simp only [← c103101, ← c103102, ← c103103] at e_1686_12
  have e_1686_13 := arithEq_of_rows h (row := 1686) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_13
  simp only [← c103104, ← c103105, k2, ← c103106] at e_1686_13
  have hr := e_1686_13
  simp only [e_1686_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16905)) := by
  have c103107 := (publicBatchWrapper86_copies3222 a h).2.2.2.1
  have c103108 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.1
  have c103109 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.1
  have c103110 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.1
  have c103111 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.1
  have c103112 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_14 := arithEq_of_rows h (row := 1686) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_14
  simp only [← c103107, ← c103108, ← c103109] at e_1686_14
  have e_1686_15 := arithEq_of_rows h (row := 1686) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_15
  simp only [← c103110, ← c103111, k2, ← c103112] at e_1686_15
  have hr := e_1686_15
  simp only [e_1686_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16906)) := by
  have c103113 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.1
  have c103114 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103115 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103116 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103117 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103118 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_16 := arithEq_of_rows h (row := 1686) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_16
  simp only [← c103113, ← c103114, ← c103115] at e_1686_16
  have e_1686_17 := arithEq_of_rows h (row := 1686) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_17
  simp only [← c103116, ← c103117, k2, ← c103118] at e_1686_17
  have hr := e_1686_17
  simp only [e_1686_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1686 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16907)) := by
  have c103119 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103120 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103121 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103122 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103123 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103124 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1686_18 := arithEq_of_rows h (row := 1686) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_18
  simp only [← c103119, ← c103120, ← c103121] at e_1686_18
  have e_1686_19 := arithEq_of_rows h (row := 1686) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1686_19
  simp only [← c103122, ← c103123, k2, ← c103124] at e_1686_19
  have hr := e_1686_19
  simp only [e_1686_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16908)) := by
  have c103125 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103126 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103127 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103128 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103129 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103130 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_0 := arithEq_of_rows h (row := 1687) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_0
  simp only [← c103125, ← c103126, ← c103127] at e_1687_0
  have e_1687_1 := arithEq_of_rows h (row := 1687) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_1
  simp only [← c103128, ← c103129, k2, ← c103130] at e_1687_1
  have hr := e_1687_1
  simp only [e_1687_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16909)) := by
  have c103131 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103132 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103133 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103134 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103135 := (publicBatchWrapper86_copies3222 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103136 := (publicBatchWrapper86_copies3223 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_2 := arithEq_of_rows h (row := 1687) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_2
  simp only [← c103131, ← c103132, ← c103133] at e_1687_2
  have e_1687_3 := arithEq_of_rows h (row := 1687) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_3
  simp only [← c103134, ← c103135, k2, ← c103136] at e_1687_3
  have hr := e_1687_3
  simp only [e_1687_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16910)) := by
  have c103137 := (publicBatchWrapper86_copies3223 a h).2.1
  have c103138 := (publicBatchWrapper86_copies3223 a h).2.2.1
  have c103139 := (publicBatchWrapper86_copies3223 a h).2.2.2.1
  have c103140 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.1
  have c103141 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.1
  have c103142 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_4 := arithEq_of_rows h (row := 1687) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_4
  simp only [← c103137, ← c103138, ← c103139] at e_1687_4
  have e_1687_5 := arithEq_of_rows h (row := 1687) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_5
  simp only [← c103140, ← c103141, k2, ← c103142] at e_1687_5
  have hr := e_1687_5
  simp only [e_1687_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16911)) := by
  have c103143 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.1
  have c103144 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.1
  have c103145 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.1
  have c103146 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103147 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103148 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_6 := arithEq_of_rows h (row := 1687) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_6
  simp only [← c103143, ← c103144, ← c103145] at e_1687_6
  have e_1687_7 := arithEq_of_rows h (row := 1687) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_7
  simp only [← c103146, ← c103147, k2, ← c103148] at e_1687_7
  have hr := e_1687_7
  simp only [e_1687_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16912)) := by
  have c103149 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103150 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103151 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103152 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103153 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103154 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_8 := arithEq_of_rows h (row := 1687) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_8
  simp only [← c103149, ← c103150, ← c103151] at e_1687_8
  have e_1687_9 := arithEq_of_rows h (row := 1687) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_9
  simp only [← c103152, ← c103153, k2, ← c103154] at e_1687_9
  have hr := e_1687_9
  simp only [e_1687_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16913)) := by
  have c103155 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103156 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103157 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103158 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103159 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103160 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_10 := arithEq_of_rows h (row := 1687) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_10
  simp only [← c103155, ← c103156, ← c103157] at e_1687_10
  have e_1687_11 := arithEq_of_rows h (row := 1687) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_11
  simp only [← c103158, ← c103159, k2, ← c103160] at e_1687_11
  have hr := e_1687_11
  simp only [e_1687_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16914)) := by
  have c103161 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103162 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103163 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103164 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103165 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103166 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_12 := arithEq_of_rows h (row := 1687) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_12
  simp only [← c103161, ← c103162, ← c103163] at e_1687_12
  have e_1687_13 := arithEq_of_rows h (row := 1687) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_13
  simp only [← c103164, ← c103165, k2, ← c103166] at e_1687_13
  have hr := e_1687_13
  simp only [e_1687_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16915)) := by
  have c103167 := (publicBatchWrapper86_copies3223 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103168 := (publicBatchWrapper86_copies3224 a h).1
  have c103169 := (publicBatchWrapper86_copies3224 a h).2.1
  have c103170 := (publicBatchWrapper86_copies3224 a h).2.2.1
  have c103171 := (publicBatchWrapper86_copies3224 a h).2.2.2.1
  have c103172 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_14 := arithEq_of_rows h (row := 1687) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_14
  simp only [← c103167, ← c103168, ← c103169] at e_1687_14
  have e_1687_15 := arithEq_of_rows h (row := 1687) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_15
  simp only [← c103170, ← c103171, k2, ← c103172] at e_1687_15
  have hr := e_1687_15
  simp only [e_1687_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16916)) := by
  have c103173 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.1
  have c103174 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.1
  have c103175 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.1
  have c103176 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.1
  have c103177 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.1
  have c103178 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_16 := arithEq_of_rows h (row := 1687) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_16
  simp only [← c103173, ← c103174, ← c103175] at e_1687_16
  have e_1687_17 := arithEq_of_rows h (row := 1687) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_17
  simp only [← c103176, ← c103177, k2, ← c103178] at e_1687_17
  have hr := e_1687_17
  simp only [e_1687_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1687 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16917)) := by
  have c103179 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103180 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103181 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103182 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103183 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103184 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1687_18 := arithEq_of_rows h (row := 1687) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_18
  simp only [← c103179, ← c103180, ← c103181] at e_1687_18
  have e_1687_19 := arithEq_of_rows h (row := 1687) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1687_19
  simp only [← c103182, ← c103183, k2, ← c103184] at e_1687_19
  have hr := e_1687_19
  simp only [e_1687_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16918)) := by
  have c103185 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103186 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103187 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103188 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103189 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103190 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_0 := arithEq_of_rows h (row := 1688) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_0
  simp only [← c103185, ← c103186, ← c103187] at e_1688_0
  have e_1688_1 := arithEq_of_rows h (row := 1688) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_1
  simp only [← c103188, ← c103189, k2, ← c103190] at e_1688_1
  have hr := e_1688_1
  simp only [e_1688_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16919)) := by
  have c103191 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103192 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103193 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103194 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103195 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103196 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_2 := arithEq_of_rows h (row := 1688) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_2
  simp only [← c103191, ← c103192, ← c103193] at e_1688_2
  have e_1688_3 := arithEq_of_rows h (row := 1688) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_3
  simp only [← c103194, ← c103195, k2, ← c103196] at e_1688_3
  have hr := e_1688_3
  simp only [e_1688_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16920)) := by
  have c103197 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103198 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103199 := (publicBatchWrapper86_copies3224 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103200 := (publicBatchWrapper86_copies3225 a h).1
  have c103201 := (publicBatchWrapper86_copies3225 a h).2.1
  have c103202 := (publicBatchWrapper86_copies3225 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_4 := arithEq_of_rows h (row := 1688) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_4
  simp only [← c103197, ← c103198, ← c103199] at e_1688_4
  have e_1688_5 := arithEq_of_rows h (row := 1688) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_5
  simp only [← c103200, ← c103201, k2, ← c103202] at e_1688_5
  have hr := e_1688_5
  simp only [e_1688_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16921)) := by
  have c103203 := (publicBatchWrapper86_copies3225 a h).2.2.2.1
  have c103204 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.1
  have c103205 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.1
  have c103206 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.1
  have c103207 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.1
  have c103208 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_6 := arithEq_of_rows h (row := 1688) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_6
  simp only [← c103203, ← c103204, ← c103205] at e_1688_6
  have e_1688_7 := arithEq_of_rows h (row := 1688) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_7
  simp only [← c103206, ← c103207, k2, ← c103208] at e_1688_7
  have hr := e_1688_7
  simp only [e_1688_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16922)) := by
  have c103209 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.1
  have c103210 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103211 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103212 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103213 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103214 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_8 := arithEq_of_rows h (row := 1688) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_8
  simp only [← c103209, ← c103210, ← c103211] at e_1688_8
  have e_1688_9 := arithEq_of_rows h (row := 1688) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_9
  simp only [← c103212, ← c103213, k2, ← c103214] at e_1688_9
  have hr := e_1688_9
  simp only [e_1688_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16923)) := by
  have c103215 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103216 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103217 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103218 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103219 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103220 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_10 := arithEq_of_rows h (row := 1688) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_10
  simp only [← c103215, ← c103216, ← c103217] at e_1688_10
  have e_1688_11 := arithEq_of_rows h (row := 1688) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_11
  simp only [← c103218, ← c103219, k2, ← c103220] at e_1688_11
  have hr := e_1688_11
  simp only [e_1688_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16924)) := by
  have c103221 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103222 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103223 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103224 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103225 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103226 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_12 := arithEq_of_rows h (row := 1688) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_12
  simp only [← c103221, ← c103222, ← c103223] at e_1688_12
  have e_1688_13 := arithEq_of_rows h (row := 1688) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_13
  simp only [← c103224, ← c103225, k2, ← c103226] at e_1688_13
  have hr := e_1688_13
  simp only [e_1688_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16925)) := by
  have c103227 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103228 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103229 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103230 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103231 := (publicBatchWrapper86_copies3225 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103232 := (publicBatchWrapper86_copies3226 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_14 := arithEq_of_rows h (row := 1688) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_14
  simp only [← c103227, ← c103228, ← c103229] at e_1688_14
  have e_1688_15 := arithEq_of_rows h (row := 1688) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_15
  simp only [← c103230, ← c103231, k2, ← c103232] at e_1688_15
  have hr := e_1688_15
  simp only [e_1688_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16926)) := by
  have c103233 := (publicBatchWrapper86_copies3226 a h).2.1
  have c103234 := (publicBatchWrapper86_copies3226 a h).2.2.1
  have c103235 := (publicBatchWrapper86_copies3226 a h).2.2.2.1
  have c103236 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.1
  have c103237 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.1
  have c103238 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_16 := arithEq_of_rows h (row := 1688) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_16
  simp only [← c103233, ← c103234, ← c103235] at e_1688_16
  have e_1688_17 := arithEq_of_rows h (row := 1688) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_17
  simp only [← c103236, ← c103237, k2, ← c103238] at e_1688_17
  have hr := e_1688_17
  simp only [e_1688_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1688 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16927)) := by
  have c103239 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.1
  have c103240 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.1
  have c103241 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.1
  have c103242 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103243 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103244 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1688_18 := arithEq_of_rows h (row := 1688) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_18
  simp only [← c103239, ← c103240, ← c103241] at e_1688_18
  have e_1688_19 := arithEq_of_rows h (row := 1688) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1688_19
  simp only [← c103242, ← c103243, k2, ← c103244] at e_1688_19
  have hr := e_1688_19
  simp only [e_1688_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16928)) := by
  have c103245 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103246 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103247 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103248 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103249 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103250 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_0 := arithEq_of_rows h (row := 1689) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_0
  simp only [← c103245, ← c103246, ← c103247] at e_1689_0
  have e_1689_1 := arithEq_of_rows h (row := 1689) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_1
  simp only [← c103248, ← c103249, k2, ← c103250] at e_1689_1
  have hr := e_1689_1
  simp only [e_1689_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16929)) := by
  have c103251 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103252 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103253 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103254 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103255 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103256 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_2 := arithEq_of_rows h (row := 1689) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_2
  simp only [← c103251, ← c103252, ← c103253] at e_1689_2
  have e_1689_3 := arithEq_of_rows h (row := 1689) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_3
  simp only [← c103254, ← c103255, k2, ← c103256] at e_1689_3
  have hr := e_1689_3
  simp only [e_1689_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16930)) := by
  have c103257 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103258 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103259 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103260 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103261 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103262 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_4 := arithEq_of_rows h (row := 1689) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_4
  simp only [← c103257, ← c103258, ← c103259] at e_1689_4
  have e_1689_5 := arithEq_of_rows h (row := 1689) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_5
  simp only [← c103260, ← c103261, k2, ← c103262] at e_1689_5
  have hr := e_1689_5
  simp only [e_1689_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16931)) := by
  have c103263 := (publicBatchWrapper86_copies3226 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103264 := (publicBatchWrapper86_copies3227 a h).1
  have c103265 := (publicBatchWrapper86_copies3227 a h).2.1
  have c103266 := (publicBatchWrapper86_copies3227 a h).2.2.1
  have c103267 := (publicBatchWrapper86_copies3227 a h).2.2.2.1
  have c103268 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_6 := arithEq_of_rows h (row := 1689) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_6
  simp only [← c103263, ← c103264, ← c103265] at e_1689_6
  have e_1689_7 := arithEq_of_rows h (row := 1689) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_7
  simp only [← c103266, ← c103267, k2, ← c103268] at e_1689_7
  have hr := e_1689_7
  simp only [e_1689_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16932)) := by
  have c103269 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.1
  have c103270 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.1
  have c103271 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.1
  have c103272 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.1
  have c103273 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.1
  have c103274 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_8 := arithEq_of_rows h (row := 1689) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_8
  simp only [← c103269, ← c103270, ← c103271] at e_1689_8
  have e_1689_9 := arithEq_of_rows h (row := 1689) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_9
  simp only [← c103272, ← c103273, k2, ← c103274] at e_1689_9
  have hr := e_1689_9
  simp only [e_1689_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16933)) := by
  have c103275 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103276 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103277 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103278 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103279 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103280 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_10 := arithEq_of_rows h (row := 1689) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_10
  simp only [← c103275, ← c103276, ← c103277] at e_1689_10
  have e_1689_11 := arithEq_of_rows h (row := 1689) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_11
  simp only [← c103278, ← c103279, k2, ← c103280] at e_1689_11
  have hr := e_1689_11
  simp only [e_1689_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16934)) := by
  have c103281 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103282 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103283 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103284 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103285 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103286 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_12 := arithEq_of_rows h (row := 1689) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_12
  simp only [← c103281, ← c103282, ← c103283] at e_1689_12
  have e_1689_13 := arithEq_of_rows h (row := 1689) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_13
  simp only [← c103284, ← c103285, k2, ← c103286] at e_1689_13
  have hr := e_1689_13
  simp only [e_1689_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16935)) := by
  have c103287 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103288 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103289 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103290 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103291 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103292 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_14 := arithEq_of_rows h (row := 1689) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_14
  simp only [← c103287, ← c103288, ← c103289] at e_1689_14
  have e_1689_15 := arithEq_of_rows h (row := 1689) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_15
  simp only [← c103290, ← c103291, k2, ← c103292] at e_1689_15
  have hr := e_1689_15
  simp only [e_1689_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16936)) := by
  have c103293 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103294 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103295 := (publicBatchWrapper86_copies3227 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103296 := (publicBatchWrapper86_copies3228 a h).1
  have c103297 := (publicBatchWrapper86_copies3228 a h).2.1
  have c103298 := (publicBatchWrapper86_copies3228 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_16 := arithEq_of_rows h (row := 1689) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_16
  simp only [← c103293, ← c103294, ← c103295] at e_1689_16
  have e_1689_17 := arithEq_of_rows h (row := 1689) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_17
  simp only [← c103296, ← c103297, k2, ← c103298] at e_1689_17
  have hr := e_1689_17
  simp only [e_1689_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1689 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16937)) := by
  have c103299 := (publicBatchWrapper86_copies3228 a h).2.2.2.1
  have c103300 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.1
  have c103301 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.1
  have c103302 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.1
  have c103303 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.1
  have c103304 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1689_18 := arithEq_of_rows h (row := 1689) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_18
  simp only [← c103299, ← c103300, ← c103301] at e_1689_18
  have e_1689_19 := arithEq_of_rows h (row := 1689) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1689_19
  simp only [← c103302, ← c103303, k2, ← c103304] at e_1689_19
  have hr := e_1689_19
  simp only [e_1689_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16938)) := by
  have c103305 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.1
  have c103306 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103307 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103308 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103309 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103310 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_0 := arithEq_of_rows h (row := 1690) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_0
  simp only [← c103305, ← c103306, ← c103307] at e_1690_0
  have e_1690_1 := arithEq_of_rows h (row := 1690) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_1
  simp only [← c103308, ← c103309, k2, ← c103310] at e_1690_1
  have hr := e_1690_1
  simp only [e_1690_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16939)) := by
  have c103311 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103312 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103313 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103314 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103315 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103316 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_2 := arithEq_of_rows h (row := 1690) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_2
  simp only [← c103311, ← c103312, ← c103313] at e_1690_2
  have e_1690_3 := arithEq_of_rows h (row := 1690) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_3
  simp only [← c103314, ← c103315, k2, ← c103316] at e_1690_3
  have hr := e_1690_3
  simp only [e_1690_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16940)) := by
  have c103317 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103318 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103319 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103320 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103321 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103322 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_4 := arithEq_of_rows h (row := 1690) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_4
  simp only [← c103317, ← c103318, ← c103319] at e_1690_4
  have e_1690_5 := arithEq_of_rows h (row := 1690) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_5
  simp only [← c103320, ← c103321, k2, ← c103322] at e_1690_5
  have hr := e_1690_5
  simp only [e_1690_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16941)) := by
  have c103323 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103324 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103325 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103326 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103327 := (publicBatchWrapper86_copies3228 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103328 := (publicBatchWrapper86_copies3229 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_6 := arithEq_of_rows h (row := 1690) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_6
  simp only [← c103323, ← c103324, ← c103325] at e_1690_6
  have e_1690_7 := arithEq_of_rows h (row := 1690) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_7
  simp only [← c103326, ← c103327, k2, ← c103328] at e_1690_7
  have hr := e_1690_7
  simp only [e_1690_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16942)) := by
  have c103329 := (publicBatchWrapper86_copies3229 a h).2.1
  have c103330 := (publicBatchWrapper86_copies3229 a h).2.2.1
  have c103331 := (publicBatchWrapper86_copies3229 a h).2.2.2.1
  have c103332 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.1
  have c103333 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.1
  have c103334 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_8 := arithEq_of_rows h (row := 1690) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_8
  simp only [← c103329, ← c103330, ← c103331] at e_1690_8
  have e_1690_9 := arithEq_of_rows h (row := 1690) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_9
  simp only [← c103332, ← c103333, k2, ← c103334] at e_1690_9
  have hr := e_1690_9
  simp only [e_1690_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16943)) := by
  have c103335 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.1
  have c103336 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.1
  have c103337 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.1
  have c103338 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103339 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103340 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_10 := arithEq_of_rows h (row := 1690) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_10
  simp only [← c103335, ← c103336, ← c103337] at e_1690_10
  have e_1690_11 := arithEq_of_rows h (row := 1690) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_11
  simp only [← c103338, ← c103339, k2, ← c103340] at e_1690_11
  have hr := e_1690_11
  simp only [e_1690_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16944)) := by
  have c103341 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103342 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103343 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103344 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103345 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103346 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_12 := arithEq_of_rows h (row := 1690) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_12
  simp only [← c103341, ← c103342, ← c103343] at e_1690_12
  have e_1690_13 := arithEq_of_rows h (row := 1690) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_13
  simp only [← c103344, ← c103345, k2, ← c103346] at e_1690_13
  have hr := e_1690_13
  simp only [e_1690_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16945)) := by
  have c103347 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103348 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103349 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103350 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103351 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103352 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_14 := arithEq_of_rows h (row := 1690) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_14
  simp only [← c103347, ← c103348, ← c103349] at e_1690_14
  have e_1690_15 := arithEq_of_rows h (row := 1690) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_15
  simp only [← c103350, ← c103351, k2, ← c103352] at e_1690_15
  have hr := e_1690_15
  simp only [e_1690_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16946)) := by
  have c103353 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103354 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103355 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103356 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103357 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103358 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_16 := arithEq_of_rows h (row := 1690) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_16
  simp only [← c103353, ← c103354, ← c103355] at e_1690_16
  have e_1690_17 := arithEq_of_rows h (row := 1690) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_17
  simp only [← c103356, ← c103357, k2, ← c103358] at e_1690_17
  have hr := e_1690_17
  simp only [e_1690_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1690 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16947)) := by
  have c103359 := (publicBatchWrapper86_copies3229 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103360 := (publicBatchWrapper86_copies3230 a h).1
  have c103361 := (publicBatchWrapper86_copies3230 a h).2.1
  have c103362 := (publicBatchWrapper86_copies3230 a h).2.2.1
  have c103363 := (publicBatchWrapper86_copies3230 a h).2.2.2.1
  have c103364 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1690_18 := arithEq_of_rows h (row := 1690) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_18
  simp only [← c103359, ← c103360, ← c103361] at e_1690_18
  have e_1690_19 := arithEq_of_rows h (row := 1690) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1690_19
  simp only [← c103362, ← c103363, k2, ← c103364] at e_1690_19
  have hr := e_1690_19
  simp only [e_1690_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16948)) := by
  have c103365 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.1
  have c103366 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.1
  have c103367 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.1
  have c103368 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.1
  have c103369 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.1
  have c103370 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_0 := arithEq_of_rows h (row := 1691) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_0
  simp only [← c103365, ← c103366, ← c103367] at e_1691_0
  have e_1691_1 := arithEq_of_rows h (row := 1691) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_1
  simp only [← c103368, ← c103369, k2, ← c103370] at e_1691_1
  have hr := e_1691_1
  simp only [e_1691_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16949)) := by
  have c103371 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103372 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103373 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103374 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103375 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103376 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_2 := arithEq_of_rows h (row := 1691) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_2
  simp only [← c103371, ← c103372, ← c103373] at e_1691_2
  have e_1691_3 := arithEq_of_rows h (row := 1691) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_3
  simp only [← c103374, ← c103375, k2, ← c103376] at e_1691_3
  have hr := e_1691_3
  simp only [e_1691_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16950)) := by
  have c103377 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103378 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103379 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103380 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103381 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103382 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_4 := arithEq_of_rows h (row := 1691) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_4
  simp only [← c103377, ← c103378, ← c103379] at e_1691_4
  have e_1691_5 := arithEq_of_rows h (row := 1691) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_5
  simp only [← c103380, ← c103381, k2, ← c103382] at e_1691_5
  have hr := e_1691_5
  simp only [e_1691_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16951)) := by
  have c103383 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103384 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103385 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103386 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103387 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103388 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_6 := arithEq_of_rows h (row := 1691) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_6
  simp only [← c103383, ← c103384, ← c103385] at e_1691_6
  have e_1691_7 := arithEq_of_rows h (row := 1691) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_7
  simp only [← c103386, ← c103387, k2, ← c103388] at e_1691_7
  have hr := e_1691_7
  simp only [e_1691_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16952)) := by
  have c103389 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103390 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103391 := (publicBatchWrapper86_copies3230 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103392 := (publicBatchWrapper86_copies3231 a h).1
  have c103393 := (publicBatchWrapper86_copies3231 a h).2.1
  have c103394 := (publicBatchWrapper86_copies3231 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_8 := arithEq_of_rows h (row := 1691) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_8
  simp only [← c103389, ← c103390, ← c103391] at e_1691_8
  have e_1691_9 := arithEq_of_rows h (row := 1691) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_9
  simp only [← c103392, ← c103393, k2, ← c103394] at e_1691_9
  have hr := e_1691_9
  simp only [e_1691_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16953)) := by
  have c103395 := (publicBatchWrapper86_copies3231 a h).2.2.2.1
  have c103396 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.1
  have c103397 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.1
  have c103398 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.1
  have c103399 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.1
  have c103400 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_10 := arithEq_of_rows h (row := 1691) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_10
  simp only [← c103395, ← c103396, ← c103397] at e_1691_10
  have e_1691_11 := arithEq_of_rows h (row := 1691) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_11
  simp only [← c103398, ← c103399, k2, ← c103400] at e_1691_11
  have hr := e_1691_11
  simp only [e_1691_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16954)) := by
  have c103401 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.1
  have c103402 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103403 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103404 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103405 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103406 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_12 := arithEq_of_rows h (row := 1691) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_12
  simp only [← c103401, ← c103402, ← c103403] at e_1691_12
  have e_1691_13 := arithEq_of_rows h (row := 1691) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_13
  simp only [← c103404, ← c103405, k2, ← c103406] at e_1691_13
  have hr := e_1691_13
  simp only [e_1691_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16955)) := by
  have c103407 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103408 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103409 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103410 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103411 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103412 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_14 := arithEq_of_rows h (row := 1691) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_14
  simp only [← c103407, ← c103408, ← c103409] at e_1691_14
  have e_1691_15 := arithEq_of_rows h (row := 1691) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_15
  simp only [← c103410, ← c103411, k2, ← c103412] at e_1691_15
  have hr := e_1691_15
  simp only [e_1691_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16956)) := by
  have c103413 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103414 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103415 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103416 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103417 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103418 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_16 := arithEq_of_rows h (row := 1691) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_16
  simp only [← c103413, ← c103414, ← c103415] at e_1691_16
  have e_1691_17 := arithEq_of_rows h (row := 1691) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_17
  simp only [← c103416, ← c103417, k2, ← c103418] at e_1691_17
  have hr := e_1691_17
  simp only [e_1691_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1691 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16957)) := by
  have c103419 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103420 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103421 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103422 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103423 := (publicBatchWrapper86_copies3231 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103424 := (publicBatchWrapper86_copies3232 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1691_18 := arithEq_of_rows h (row := 1691) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_18
  simp only [← c103419, ← c103420, ← c103421] at e_1691_18
  have e_1691_19 := arithEq_of_rows h (row := 1691) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1691_19
  simp only [← c103422, ← c103423, k2, ← c103424] at e_1691_19
  have hr := e_1691_19
  simp only [e_1691_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16958)) := by
  have c103425 := (publicBatchWrapper86_copies3232 a h).2.1
  have c103426 := (publicBatchWrapper86_copies3232 a h).2.2.1
  have c103427 := (publicBatchWrapper86_copies3232 a h).2.2.2.1
  have c103428 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.1
  have c103429 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.1
  have c103430 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_0 := arithEq_of_rows h (row := 1692) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_0
  simp only [← c103425, ← c103426, ← c103427] at e_1692_0
  have e_1692_1 := arithEq_of_rows h (row := 1692) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_1
  simp only [← c103428, ← c103429, k2, ← c103430] at e_1692_1
  have hr := e_1692_1
  simp only [e_1692_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16959)) := by
  have c103431 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.1
  have c103432 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.1
  have c103433 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.1
  have c103434 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103435 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103436 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_2 := arithEq_of_rows h (row := 1692) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_2
  simp only [← c103431, ← c103432, ← c103433] at e_1692_2
  have e_1692_3 := arithEq_of_rows h (row := 1692) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_3
  simp only [← c103434, ← c103435, k2, ← c103436] at e_1692_3
  have hr := e_1692_3
  simp only [e_1692_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 23) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16960)) := by
  have c103437 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103438 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103439 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103440 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103441 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103442 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_4 := arithEq_of_rows h (row := 1692) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_4
  simp only [← c103437, ← c103438, ← c103439] at e_1692_4
  have e_1692_5 := arithEq_of_rows h (row := 1692) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_5
  simp only [← c103440, ← c103441, k2, ← c103442] at e_1692_5
  have hr := e_1692_5
  simp only [e_1692_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 31) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16961)) := by
  have c103443 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103444 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103445 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103446 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103447 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103448 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_6 := arithEq_of_rows h (row := 1692) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_6
  simp only [← c103443, ← c103444, ← c103445] at e_1692_6
  have e_1692_7 := arithEq_of_rows h (row := 1692) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_7
  simp only [← c103446, ← c103447, k2, ← c103448] at e_1692_7
  have hr := e_1692_7
  simp only [e_1692_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 39) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16962)) := by
  have c103449 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103450 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103451 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103452 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103453 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103454 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_8 := arithEq_of_rows h (row := 1692) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_8
  simp only [← c103449, ← c103450, ← c103451] at e_1692_8
  have e_1692_9 := arithEq_of_rows h (row := 1692) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_9
  simp only [← c103452, ← c103453, k2, ← c103454] at e_1692_9
  have hr := e_1692_9
  simp only [e_1692_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 47) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16963)) := by
  have c103455 := (publicBatchWrapper86_copies3232 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103456 := (publicBatchWrapper86_copies3233 a h).1
  have c103457 := (publicBatchWrapper86_copies3233 a h).2.1
  have c103458 := (publicBatchWrapper86_copies3233 a h).2.2.1
  have c103459 := (publicBatchWrapper86_copies3233 a h).2.2.2.1
  have c103460 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_10 := arithEq_of_rows h (row := 1692) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_10
  simp only [← c103455, ← c103456, ← c103457] at e_1692_10
  have e_1692_11 := arithEq_of_rows h (row := 1692) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_11
  simp only [← c103458, ← c103459, k2, ← c103460] at e_1692_11
  have hr := e_1692_11
  simp only [e_1692_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 55) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16964)) := by
  have c103461 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.1
  have c103462 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.1
  have c103463 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.1
  have c103464 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.1
  have c103465 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.1
  have c103466 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_12 := arithEq_of_rows h (row := 1692) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_12
  simp only [← c103461, ← c103462, ← c103463] at e_1692_12
  have e_1692_13 := arithEq_of_rows h (row := 1692) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_13
  simp only [← c103464, ← c103465, k2, ← c103466] at e_1692_13
  have hr := e_1692_13
  simp only [e_1692_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 63) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16965)) := by
  have c103467 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103468 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103469 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103470 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103471 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103472 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_14 := arithEq_of_rows h (row := 1692) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_14
  simp only [← c103467, ← c103468, ← c103469] at e_1692_14
  have e_1692_15 := arithEq_of_rows h (row := 1692) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_15
  simp only [← c103470, ← c103471, k2, ← c103472] at e_1692_15
  have hr := e_1692_15
  simp only [e_1692_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 71) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16966)) := by
  have c103473 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103474 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103475 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103476 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103477 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103478 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_16 := arithEq_of_rows h (row := 1692) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_16
  simp only [← c103473, ← c103474, ← c103475] at e_1692_16
  have e_1692_17 := arithEq_of_rows h (row := 1692) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_17
  simp only [← c103476, ← c103477, k2, ← c103478] at e_1692_17
  have hr := e_1692_17
  simp only [e_1692_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1692 79) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16967)) := by
  have c103479 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103480 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103481 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103482 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103483 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103484 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1692_18 := arithEq_of_rows h (row := 1692) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_18
  simp only [← c103479, ← c103480, ← c103481] at e_1692_18
  have e_1692_19 := arithEq_of_rows h (row := 1692) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1692_19
  simp only [← c103482, ← c103483, k2, ← c103484] at e_1692_19
  have hr := e_1692_19
  simp only [e_1692_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 7) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16968)) := by
  have c103485 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103486 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103487 := (publicBatchWrapper86_copies3233 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103488 := (publicBatchWrapper86_copies3234 a h).1
  have c103489 := (publicBatchWrapper86_copies3234 a h).2.1
  have c103490 := (publicBatchWrapper86_copies3234 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_0 := arithEq_of_rows h (row := 1693) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_0
  simp only [← c103485, ← c103486, ← c103487] at e_1693_0
  have e_1693_1 := arithEq_of_rows h (row := 1693) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_1
  simp only [← c103488, ← c103489, k2, ← c103490] at e_1693_1
  have hr := e_1693_1
  simp only [e_1693_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 15) = bselect (a (.wire 82 43)) (a (.virt 200)) (a (.virt 16969)) := by
  have c103491 := (publicBatchWrapper86_copies3234 a h).2.2.2.1
  have c103492 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.1
  have c103493 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.1
  have c103494 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.1
  have c103495 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.1
  have c103496 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_2 := arithEq_of_rows h (row := 1693) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_2
  simp only [← c103491, ← c103492, ← c103493] at e_1693_2
  have e_1693_3 := arithEq_of_rows h (row := 1693) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_3
  simp only [← c103494, ← c103495, k2, ← c103496] at e_1693_3
  have hr := e_1693_3
  simp only [e_1693_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17010)) := by
  have c103497 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.1
  have c103498 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103499 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103500 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103501 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103502 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_4 := arithEq_of_rows h (row := 1693) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_4
  simp only [← c103497, ← c103498, ← c103499] at e_1693_4
  have e_1693_5 := arithEq_of_rows h (row := 1693) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_5
  simp only [← c103500, ← c103501, k2, ← c103502] at e_1693_5
  have hr := e_1693_5
  simp only [e_1693_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17011)) := by
  have c103503 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103504 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103505 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103506 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103507 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103508 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_6 := arithEq_of_rows h (row := 1693) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_6
  simp only [← c103503, ← c103504, ← c103505] at e_1693_6
  have e_1693_7 := arithEq_of_rows h (row := 1693) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_7
  simp only [← c103506, ← c103507, k2, ← c103508] at e_1693_7
  have hr := e_1693_7
  simp only [e_1693_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17012)) := by
  have c103509 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103510 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103511 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103512 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103513 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103514 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_8 := arithEq_of_rows h (row := 1693) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_8
  simp only [← c103509, ← c103510, ← c103511] at e_1693_8
  have e_1693_9 := arithEq_of_rows h (row := 1693) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_9
  simp only [← c103512, ← c103513, k2, ← c103514] at e_1693_9
  have hr := e_1693_9
  simp only [e_1693_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17013)) := by
  have c103515 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103516 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103517 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103518 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103519 := (publicBatchWrapper86_copies3234 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103520 := (publicBatchWrapper86_copies3235 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_10 := arithEq_of_rows h (row := 1693) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_10
  simp only [← c103515, ← c103516, ← c103517] at e_1693_10
  have e_1693_11 := arithEq_of_rows h (row := 1693) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_11
  simp only [← c103518, ← c103519, k2, ← c103520] at e_1693_11
  have hr := e_1693_11
  simp only [e_1693_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17014)) := by
  have c103521 := (publicBatchWrapper86_copies3235 a h).2.1
  have c103522 := (publicBatchWrapper86_copies3235 a h).2.2.1
  have c103523 := (publicBatchWrapper86_copies3235 a h).2.2.2.1
  have c103524 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.1
  have c103525 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.1
  have c103526 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_12 := arithEq_of_rows h (row := 1693) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_12
  simp only [← c103521, ← c103522, ← c103523] at e_1693_12
  have e_1693_13 := arithEq_of_rows h (row := 1693) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_13
  simp only [← c103524, ← c103525, k2, ← c103526] at e_1693_13
  have hr := e_1693_13
  simp only [e_1693_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17015)) := by
  have c103527 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.1
  have c103528 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.1
  have c103529 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.1
  have c103530 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103531 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103532 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_14 := arithEq_of_rows h (row := 1693) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_14
  simp only [← c103527, ← c103528, ← c103529] at e_1693_14
  have e_1693_15 := arithEq_of_rows h (row := 1693) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_15
  simp only [← c103530, ← c103531, k2, ← c103532] at e_1693_15
  have hr := e_1693_15
  simp only [e_1693_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17016)) := by
  have c103533 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103534 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103535 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103536 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103537 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103538 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_16 := arithEq_of_rows h (row := 1693) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_16
  simp only [← c103533, ← c103534, ← c103535] at e_1693_16
  have e_1693_17 := arithEq_of_rows h (row := 1693) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_17
  simp only [← c103536, ← c103537, k2, ← c103538] at e_1693_17
  have hr := e_1693_17
  simp only [e_1693_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1693 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17017)) := by
  have c103539 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103540 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103541 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103542 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103543 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103544 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1693_18 := arithEq_of_rows h (row := 1693) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_18
  simp only [← c103539, ← c103540, ← c103541] at e_1693_18
  have e_1693_19 := arithEq_of_rows h (row := 1693) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1693_19
  simp only [← c103542, ← c103543, k2, ← c103544] at e_1693_19
  have hr := e_1693_19
  simp only [e_1693_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17018)) := by
  have c103545 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103546 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103547 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103548 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103549 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103550 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_0 := arithEq_of_rows h (row := 1694) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_0
  simp only [← c103545, ← c103546, ← c103547] at e_1694_0
  have e_1694_1 := arithEq_of_rows h (row := 1694) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_1
  simp only [← c103548, ← c103549, k2, ← c103550] at e_1694_1
  have hr := e_1694_1
  simp only [e_1694_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17019)) := by
  have c103551 := (publicBatchWrapper86_copies3235 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103552 := (publicBatchWrapper86_copies3236 a h).1
  have c103553 := (publicBatchWrapper86_copies3236 a h).2.1
  have c103554 := (publicBatchWrapper86_copies3236 a h).2.2.1
  have c103555 := (publicBatchWrapper86_copies3236 a h).2.2.2.1
  have c103556 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_2 := arithEq_of_rows h (row := 1694) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_2
  simp only [← c103551, ← c103552, ← c103553] at e_1694_2
  have e_1694_3 := arithEq_of_rows h (row := 1694) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_3
  simp only [← c103554, ← c103555, k2, ← c103556] at e_1694_3
  have hr := e_1694_3
  simp only [e_1694_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17020)) := by
  have c103557 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.1
  have c103558 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.1
  have c103559 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.1
  have c103560 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.1
  have c103561 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.1
  have c103562 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_4 := arithEq_of_rows h (row := 1694) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_4
  simp only [← c103557, ← c103558, ← c103559] at e_1694_4
  have e_1694_5 := arithEq_of_rows h (row := 1694) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_5
  simp only [← c103560, ← c103561, k2, ← c103562] at e_1694_5
  have hr := e_1694_5
  simp only [e_1694_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17021)) := by
  have c103563 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103564 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103565 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103566 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103567 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103568 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_6 := arithEq_of_rows h (row := 1694) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_6
  simp only [← c103563, ← c103564, ← c103565] at e_1694_6
  have e_1694_7 := arithEq_of_rows h (row := 1694) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_7
  simp only [← c103566, ← c103567, k2, ← c103568] at e_1694_7
  have hr := e_1694_7
  simp only [e_1694_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17022)) := by
  have c103569 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103570 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103571 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103572 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103573 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103574 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_8 := arithEq_of_rows h (row := 1694) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_8
  simp only [← c103569, ← c103570, ← c103571] at e_1694_8
  have e_1694_9 := arithEq_of_rows h (row := 1694) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_9
  simp only [← c103572, ← c103573, k2, ← c103574] at e_1694_9
  have hr := e_1694_9
  simp only [e_1694_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17023)) := by
  have c103575 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103576 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103577 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103578 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103579 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103580 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_10 := arithEq_of_rows h (row := 1694) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_10
  simp only [← c103575, ← c103576, ← c103577] at e_1694_10
  have e_1694_11 := arithEq_of_rows h (row := 1694) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_11
  simp only [← c103578, ← c103579, k2, ← c103580] at e_1694_11
  have hr := e_1694_11
  simp only [e_1694_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17024)) := by
  have c103581 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103582 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103583 := (publicBatchWrapper86_copies3236 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103584 := (publicBatchWrapper86_copies3237 a h).1
  have c103585 := (publicBatchWrapper86_copies3237 a h).2.1
  have c103586 := (publicBatchWrapper86_copies3237 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_12 := arithEq_of_rows h (row := 1694) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_12
  simp only [← c103581, ← c103582, ← c103583] at e_1694_12
  have e_1694_13 := arithEq_of_rows h (row := 1694) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_13
  simp only [← c103584, ← c103585, k2, ← c103586] at e_1694_13
  have hr := e_1694_13
  simp only [e_1694_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17025)) := by
  have c103587 := (publicBatchWrapper86_copies3237 a h).2.2.2.1
  have c103588 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.1
  have c103589 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.1
  have c103590 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.1
  have c103591 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.1
  have c103592 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_14 := arithEq_of_rows h (row := 1694) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_14
  simp only [← c103587, ← c103588, ← c103589] at e_1694_14
  have e_1694_15 := arithEq_of_rows h (row := 1694) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_15
  simp only [← c103590, ← c103591, k2, ← c103592] at e_1694_15
  have hr := e_1694_15
  simp only [e_1694_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17026)) := by
  have c103593 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.1
  have c103594 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103595 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103596 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103597 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103598 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_16 := arithEq_of_rows h (row := 1694) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_16
  simp only [← c103593, ← c103594, ← c103595] at e_1694_16
  have e_1694_17 := arithEq_of_rows h (row := 1694) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_17
  simp only [← c103596, ← c103597, k2, ← c103598] at e_1694_17
  have hr := e_1694_17
  simp only [e_1694_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1694 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17027)) := by
  have c103599 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103600 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103601 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103602 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103603 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103604 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1694_18 := arithEq_of_rows h (row := 1694) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_18
  simp only [← c103599, ← c103600, ← c103601] at e_1694_18
  have e_1694_19 := arithEq_of_rows h (row := 1694) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1694_19
  simp only [← c103602, ← c103603, k2, ← c103604] at e_1694_19
  have hr := e_1694_19
  simp only [e_1694_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17028)) := by
  have c103605 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103606 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103607 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103608 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103609 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103610 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_0 := arithEq_of_rows h (row := 1695) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_0
  simp only [← c103605, ← c103606, ← c103607] at e_1695_0
  have e_1695_1 := arithEq_of_rows h (row := 1695) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_1
  simp only [← c103608, ← c103609, k2, ← c103610] at e_1695_1
  have hr := e_1695_1
  simp only [e_1695_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17029)) := by
  have c103611 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103612 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103613 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103614 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103615 := (publicBatchWrapper86_copies3237 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103616 := (publicBatchWrapper86_copies3238 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_2 := arithEq_of_rows h (row := 1695) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_2
  simp only [← c103611, ← c103612, ← c103613] at e_1695_2
  have e_1695_3 := arithEq_of_rows h (row := 1695) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_3
  simp only [← c103614, ← c103615, k2, ← c103616] at e_1695_3
  have hr := e_1695_3
  simp only [e_1695_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17030)) := by
  have c103617 := (publicBatchWrapper86_copies3238 a h).2.1
  have c103618 := (publicBatchWrapper86_copies3238 a h).2.2.1
  have c103619 := (publicBatchWrapper86_copies3238 a h).2.2.2.1
  have c103620 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.1
  have c103621 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.1
  have c103622 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_4 := arithEq_of_rows h (row := 1695) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_4
  simp only [← c103617, ← c103618, ← c103619] at e_1695_4
  have e_1695_5 := arithEq_of_rows h (row := 1695) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_5
  simp only [← c103620, ← c103621, k2, ← c103622] at e_1695_5
  have hr := e_1695_5
  simp only [e_1695_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17031)) := by
  have c103623 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.1
  have c103624 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.1
  have c103625 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.1
  have c103626 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103627 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103628 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_6 := arithEq_of_rows h (row := 1695) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_6
  simp only [← c103623, ← c103624, ← c103625] at e_1695_6
  have e_1695_7 := arithEq_of_rows h (row := 1695) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_7
  simp only [← c103626, ← c103627, k2, ← c103628] at e_1695_7
  have hr := e_1695_7
  simp only [e_1695_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17032)) := by
  have c103629 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103630 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103631 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103632 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103633 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103634 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_8 := arithEq_of_rows h (row := 1695) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_8
  simp only [← c103629, ← c103630, ← c103631] at e_1695_8
  have e_1695_9 := arithEq_of_rows h (row := 1695) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_9
  simp only [← c103632, ← c103633, k2, ← c103634] at e_1695_9
  have hr := e_1695_9
  simp only [e_1695_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17033)) := by
  have c103635 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103636 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103637 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103638 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103639 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103640 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_10 := arithEq_of_rows h (row := 1695) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_10
  simp only [← c103635, ← c103636, ← c103637] at e_1695_10
  have e_1695_11 := arithEq_of_rows h (row := 1695) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_11
  simp only [← c103638, ← c103639, k2, ← c103640] at e_1695_11
  have hr := e_1695_11
  simp only [e_1695_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17034)) := by
  have c103641 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103642 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103643 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103644 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103645 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103646 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_12 := arithEq_of_rows h (row := 1695) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_12
  simp only [← c103641, ← c103642, ← c103643] at e_1695_12
  have e_1695_13 := arithEq_of_rows h (row := 1695) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_13
  simp only [← c103644, ← c103645, k2, ← c103646] at e_1695_13
  have hr := e_1695_13
  simp only [e_1695_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17035)) := by
  have c103647 := (publicBatchWrapper86_copies3238 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103648 := (publicBatchWrapper86_copies3239 a h).1
  have c103649 := (publicBatchWrapper86_copies3239 a h).2.1
  have c103650 := (publicBatchWrapper86_copies3239 a h).2.2.1
  have c103651 := (publicBatchWrapper86_copies3239 a h).2.2.2.1
  have c103652 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_14 := arithEq_of_rows h (row := 1695) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_14
  simp only [← c103647, ← c103648, ← c103649] at e_1695_14
  have e_1695_15 := arithEq_of_rows h (row := 1695) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_15
  simp only [← c103650, ← c103651, k2, ← c103652] at e_1695_15
  have hr := e_1695_15
  simp only [e_1695_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17036)) := by
  have c103653 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.1
  have c103654 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.1
  have c103655 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.1
  have c103656 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.1
  have c103657 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.1
  have c103658 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_16 := arithEq_of_rows h (row := 1695) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_16
  simp only [← c103653, ← c103654, ← c103655] at e_1695_16
  have e_1695_17 := arithEq_of_rows h (row := 1695) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_17
  simp only [← c103656, ← c103657, k2, ← c103658] at e_1695_17
  have hr := e_1695_17
  simp only [e_1695_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1695 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17037)) := by
  have c103659 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103660 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103661 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103662 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103663 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103664 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1695_18 := arithEq_of_rows h (row := 1695) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_18
  simp only [← c103659, ← c103660, ← c103661] at e_1695_18
  have e_1695_19 := arithEq_of_rows h (row := 1695) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1695_19
  simp only [← c103662, ← c103663, k2, ← c103664] at e_1695_19
  have hr := e_1695_19
  simp only [e_1695_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17038)) := by
  have c103665 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103666 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103667 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103668 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103669 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103670 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_0 := arithEq_of_rows h (row := 1696) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_0
  simp only [← c103665, ← c103666, ← c103667] at e_1696_0
  have e_1696_1 := arithEq_of_rows h (row := 1696) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_1
  simp only [← c103668, ← c103669, k2, ← c103670] at e_1696_1
  have hr := e_1696_1
  simp only [e_1696_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17039)) := by
  have c103671 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103672 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103673 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103674 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103675 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103676 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_2 := arithEq_of_rows h (row := 1696) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_2
  simp only [← c103671, ← c103672, ← c103673] at e_1696_2
  have e_1696_3 := arithEq_of_rows h (row := 1696) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_3
  simp only [← c103674, ← c103675, k2, ← c103676] at e_1696_3
  have hr := e_1696_3
  simp only [e_1696_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17040)) := by
  have c103677 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103678 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103679 := (publicBatchWrapper86_copies3239 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103680 := (publicBatchWrapper86_copies3240 a h).1
  have c103681 := (publicBatchWrapper86_copies3240 a h).2.1
  have c103682 := (publicBatchWrapper86_copies3240 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_4 := arithEq_of_rows h (row := 1696) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_4
  simp only [← c103677, ← c103678, ← c103679] at e_1696_4
  have e_1696_5 := arithEq_of_rows h (row := 1696) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_5
  simp only [← c103680, ← c103681, k2, ← c103682] at e_1696_5
  have hr := e_1696_5
  simp only [e_1696_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17041)) := by
  have c103683 := (publicBatchWrapper86_copies3240 a h).2.2.2.1
  have c103684 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.1
  have c103685 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.1
  have c103686 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.1
  have c103687 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.1
  have c103688 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_6 := arithEq_of_rows h (row := 1696) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_6
  simp only [← c103683, ← c103684, ← c103685] at e_1696_6
  have e_1696_7 := arithEq_of_rows h (row := 1696) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_7
  simp only [← c103686, ← c103687, k2, ← c103688] at e_1696_7
  have hr := e_1696_7
  simp only [e_1696_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17042)) := by
  have c103689 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.1
  have c103690 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103691 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103692 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103693 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103694 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_8 := arithEq_of_rows h (row := 1696) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_8
  simp only [← c103689, ← c103690, ← c103691] at e_1696_8
  have e_1696_9 := arithEq_of_rows h (row := 1696) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_9
  simp only [← c103692, ← c103693, k2, ← c103694] at e_1696_9
  have hr := e_1696_9
  simp only [e_1696_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17043)) := by
  have c103695 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103696 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103697 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103698 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103699 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103700 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_10 := arithEq_of_rows h (row := 1696) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_10
  simp only [← c103695, ← c103696, ← c103697] at e_1696_10
  have e_1696_11 := arithEq_of_rows h (row := 1696) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_11
  simp only [← c103698, ← c103699, k2, ← c103700] at e_1696_11
  have hr := e_1696_11
  simp only [e_1696_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17044)) := by
  have c103701 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103702 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103703 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103704 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103705 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103706 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_12 := arithEq_of_rows h (row := 1696) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_12
  simp only [← c103701, ← c103702, ← c103703] at e_1696_12
  have e_1696_13 := arithEq_of_rows h (row := 1696) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_13
  simp only [← c103704, ← c103705, k2, ← c103706] at e_1696_13
  have hr := e_1696_13
  simp only [e_1696_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17045)) := by
  have c103707 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103708 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103709 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103710 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103711 := (publicBatchWrapper86_copies3240 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103712 := (publicBatchWrapper86_copies3241 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_14 := arithEq_of_rows h (row := 1696) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_14
  simp only [← c103707, ← c103708, ← c103709] at e_1696_14
  have e_1696_15 := arithEq_of_rows h (row := 1696) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_15
  simp only [← c103710, ← c103711, k2, ← c103712] at e_1696_15
  have hr := e_1696_15
  simp only [e_1696_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17046)) := by
  have c103713 := (publicBatchWrapper86_copies3241 a h).2.1
  have c103714 := (publicBatchWrapper86_copies3241 a h).2.2.1
  have c103715 := (publicBatchWrapper86_copies3241 a h).2.2.2.1
  have c103716 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.1
  have c103717 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.1
  have c103718 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_16 := arithEq_of_rows h (row := 1696) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_16
  simp only [← c103713, ← c103714, ← c103715] at e_1696_16
  have e_1696_17 := arithEq_of_rows h (row := 1696) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_17
  simp only [← c103716, ← c103717, k2, ← c103718] at e_1696_17
  have hr := e_1696_17
  simp only [e_1696_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1696 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17047)) := by
  have c103719 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.1
  have c103720 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.1
  have c103721 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.1
  have c103722 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103723 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103724 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1696_18 := arithEq_of_rows h (row := 1696) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_18
  simp only [← c103719, ← c103720, ← c103721] at e_1696_18
  have e_1696_19 := arithEq_of_rows h (row := 1696) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1696_19
  simp only [← c103722, ← c103723, k2, ← c103724] at e_1696_19
  have hr := e_1696_19
  simp only [e_1696_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17048)) := by
  have c103725 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103726 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103727 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103728 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103729 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103730 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_0 := arithEq_of_rows h (row := 1697) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_0
  simp only [← c103725, ← c103726, ← c103727] at e_1697_0
  have e_1697_1 := arithEq_of_rows h (row := 1697) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_1
  simp only [← c103728, ← c103729, k2, ← c103730] at e_1697_1
  have hr := e_1697_1
  simp only [e_1697_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17049)) := by
  have c103731 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103732 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103733 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103734 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103735 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103736 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_2 := arithEq_of_rows h (row := 1697) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_2
  simp only [← c103731, ← c103732, ← c103733] at e_1697_2
  have e_1697_3 := arithEq_of_rows h (row := 1697) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_3
  simp only [← c103734, ← c103735, k2, ← c103736] at e_1697_3
  have hr := e_1697_3
  simp only [e_1697_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17050)) := by
  have c103737 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103738 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103739 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103740 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103741 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103742 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_4 := arithEq_of_rows h (row := 1697) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_4
  simp only [← c103737, ← c103738, ← c103739] at e_1697_4
  have e_1697_5 := arithEq_of_rows h (row := 1697) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_5
  simp only [← c103740, ← c103741, k2, ← c103742] at e_1697_5
  have hr := e_1697_5
  simp only [e_1697_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17051)) := by
  have c103743 := (publicBatchWrapper86_copies3241 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103744 := (publicBatchWrapper86_copies3242 a h).1
  have c103745 := (publicBatchWrapper86_copies3242 a h).2.1
  have c103746 := (publicBatchWrapper86_copies3242 a h).2.2.1
  have c103747 := (publicBatchWrapper86_copies3242 a h).2.2.2.1
  have c103748 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_6 := arithEq_of_rows h (row := 1697) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_6
  simp only [← c103743, ← c103744, ← c103745] at e_1697_6
  have e_1697_7 := arithEq_of_rows h (row := 1697) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_7
  simp only [← c103746, ← c103747, k2, ← c103748] at e_1697_7
  have hr := e_1697_7
  simp only [e_1697_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17052)) := by
  have c103749 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.1
  have c103750 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.1
  have c103751 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.1
  have c103752 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.1
  have c103753 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.1
  have c103754 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_8 := arithEq_of_rows h (row := 1697) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_8
  simp only [← c103749, ← c103750, ← c103751] at e_1697_8
  have e_1697_9 := arithEq_of_rows h (row := 1697) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_9
  simp only [← c103752, ← c103753, k2, ← c103754] at e_1697_9
  have hr := e_1697_9
  simp only [e_1697_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17053)) := by
  have c103755 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103756 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103757 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103758 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103759 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103760 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_10 := arithEq_of_rows h (row := 1697) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_10
  simp only [← c103755, ← c103756, ← c103757] at e_1697_10
  have e_1697_11 := arithEq_of_rows h (row := 1697) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_11
  simp only [← c103758, ← c103759, k2, ← c103760] at e_1697_11
  have hr := e_1697_11
  simp only [e_1697_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17054)) := by
  have c103761 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103762 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103763 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103764 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103765 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103766 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_12 := arithEq_of_rows h (row := 1697) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_12
  simp only [← c103761, ← c103762, ← c103763] at e_1697_12
  have e_1697_13 := arithEq_of_rows h (row := 1697) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_13
  simp only [← c103764, ← c103765, k2, ← c103766] at e_1697_13
  have hr := e_1697_13
  simp only [e_1697_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17055)) := by
  have c103767 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103768 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103769 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103770 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103771 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103772 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_14 := arithEq_of_rows h (row := 1697) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_14
  simp only [← c103767, ← c103768, ← c103769] at e_1697_14
  have e_1697_15 := arithEq_of_rows h (row := 1697) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_15
  simp only [← c103770, ← c103771, k2, ← c103772] at e_1697_15
  have hr := e_1697_15
  simp only [e_1697_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17056)) := by
  have c103773 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103774 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103775 := (publicBatchWrapper86_copies3242 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103776 := (publicBatchWrapper86_copies3243 a h).1
  have c103777 := (publicBatchWrapper86_copies3243 a h).2.1
  have c103778 := (publicBatchWrapper86_copies3243 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_16 := arithEq_of_rows h (row := 1697) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_16
  simp only [← c103773, ← c103774, ← c103775] at e_1697_16
  have e_1697_17 := arithEq_of_rows h (row := 1697) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_17
  simp only [← c103776, ← c103777, k2, ← c103778] at e_1697_17
  have hr := e_1697_17
  simp only [e_1697_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1697 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17057)) := by
  have c103779 := (publicBatchWrapper86_copies3243 a h).2.2.2.1
  have c103780 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.1
  have c103781 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.1
  have c103782 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.1
  have c103783 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.1
  have c103784 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1697_18 := arithEq_of_rows h (row := 1697) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_18
  simp only [← c103779, ← c103780, ← c103781] at e_1697_18
  have e_1697_19 := arithEq_of_rows h (row := 1697) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1697_19
  simp only [← c103782, ← c103783, k2, ← c103784] at e_1697_19
  have hr := e_1697_19
  simp only [e_1697_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17058)) := by
  have c103785 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.1
  have c103786 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103787 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103788 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103789 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103790 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_0 := arithEq_of_rows h (row := 1698) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_0
  simp only [← c103785, ← c103786, ← c103787] at e_1698_0
  have e_1698_1 := arithEq_of_rows h (row := 1698) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_1
  simp only [← c103788, ← c103789, k2, ← c103790] at e_1698_1
  have hr := e_1698_1
  simp only [e_1698_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17059)) := by
  have c103791 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103792 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103793 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103794 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103795 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103796 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_2 := arithEq_of_rows h (row := 1698) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_2
  simp only [← c103791, ← c103792, ← c103793] at e_1698_2
  have e_1698_3 := arithEq_of_rows h (row := 1698) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_3
  simp only [← c103794, ← c103795, k2, ← c103796] at e_1698_3
  have hr := e_1698_3
  simp only [e_1698_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17060)) := by
  have c103797 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103798 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103799 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103800 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103801 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103802 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_4 := arithEq_of_rows h (row := 1698) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_4
  simp only [← c103797, ← c103798, ← c103799] at e_1698_4
  have e_1698_5 := arithEq_of_rows h (row := 1698) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_5
  simp only [← c103800, ← c103801, k2, ← c103802] at e_1698_5
  have hr := e_1698_5
  simp only [e_1698_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17061)) := by
  have c103803 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103804 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103805 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103806 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103807 := (publicBatchWrapper86_copies3243 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103808 := (publicBatchWrapper86_copies3244 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_6 := arithEq_of_rows h (row := 1698) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_6
  simp only [← c103803, ← c103804, ← c103805] at e_1698_6
  have e_1698_7 := arithEq_of_rows h (row := 1698) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_7
  simp only [← c103806, ← c103807, k2, ← c103808] at e_1698_7
  have hr := e_1698_7
  simp only [e_1698_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17062)) := by
  have c103809 := (publicBatchWrapper86_copies3244 a h).2.1
  have c103810 := (publicBatchWrapper86_copies3244 a h).2.2.1
  have c103811 := (publicBatchWrapper86_copies3244 a h).2.2.2.1
  have c103812 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.1
  have c103813 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.1
  have c103814 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_8 := arithEq_of_rows h (row := 1698) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_8
  simp only [← c103809, ← c103810, ← c103811] at e_1698_8
  have e_1698_9 := arithEq_of_rows h (row := 1698) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_9
  simp only [← c103812, ← c103813, k2, ← c103814] at e_1698_9
  have hr := e_1698_9
  simp only [e_1698_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17063)) := by
  have c103815 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.1
  have c103816 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.1
  have c103817 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.1
  have c103818 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103819 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103820 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_10 := arithEq_of_rows h (row := 1698) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_10
  simp only [← c103815, ← c103816, ← c103817] at e_1698_10
  have e_1698_11 := arithEq_of_rows h (row := 1698) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_11
  simp only [← c103818, ← c103819, k2, ← c103820] at e_1698_11
  have hr := e_1698_11
  simp only [e_1698_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17064)) := by
  have c103821 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103822 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103823 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103824 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103825 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103826 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_12 := arithEq_of_rows h (row := 1698) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_12
  simp only [← c103821, ← c103822, ← c103823] at e_1698_12
  have e_1698_13 := arithEq_of_rows h (row := 1698) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_13
  simp only [← c103824, ← c103825, k2, ← c103826] at e_1698_13
  have hr := e_1698_13
  simp only [e_1698_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17065)) := by
  have c103827 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103828 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103829 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103830 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103831 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103832 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_14 := arithEq_of_rows h (row := 1698) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_14
  simp only [← c103827, ← c103828, ← c103829] at e_1698_14
  have e_1698_15 := arithEq_of_rows h (row := 1698) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_15
  simp only [← c103830, ← c103831, k2, ← c103832] at e_1698_15
  have hr := e_1698_15
  simp only [e_1698_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17066)) := by
  have c103833 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103834 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103835 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103836 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103837 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103838 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_16 := arithEq_of_rows h (row := 1698) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_16
  simp only [← c103833, ← c103834, ← c103835] at e_1698_16
  have e_1698_17 := arithEq_of_rows h (row := 1698) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_17
  simp only [← c103836, ← c103837, k2, ← c103838] at e_1698_17
  have hr := e_1698_17
  simp only [e_1698_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1698 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17067)) := by
  have c103839 := (publicBatchWrapper86_copies3244 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103840 := (publicBatchWrapper86_copies3245 a h).1
  have c103841 := (publicBatchWrapper86_copies3245 a h).2.1
  have c103842 := (publicBatchWrapper86_copies3245 a h).2.2.1
  have c103843 := (publicBatchWrapper86_copies3245 a h).2.2.2.1
  have c103844 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1698_18 := arithEq_of_rows h (row := 1698) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_18
  simp only [← c103839, ← c103840, ← c103841] at e_1698_18
  have e_1698_19 := arithEq_of_rows h (row := 1698) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1698_19
  simp only [← c103842, ← c103843, k2, ← c103844] at e_1698_19
  have hr := e_1698_19
  simp only [e_1698_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17068)) := by
  have c103845 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.1
  have c103846 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.1
  have c103847 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.1
  have c103848 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.1
  have c103849 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.1
  have c103850 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_0 := arithEq_of_rows h (row := 1699) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_0
  simp only [← c103845, ← c103846, ← c103847] at e_1699_0
  have e_1699_1 := arithEq_of_rows h (row := 1699) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_1
  simp only [← c103848, ← c103849, k2, ← c103850] at e_1699_1
  have hr := e_1699_1
  simp only [e_1699_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17069)) := by
  have c103851 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103852 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103853 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103854 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103855 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103856 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_2 := arithEq_of_rows h (row := 1699) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_2
  simp only [← c103851, ← c103852, ← c103853] at e_1699_2
  have e_1699_3 := arithEq_of_rows h (row := 1699) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_3
  simp only [← c103854, ← c103855, k2, ← c103856] at e_1699_3
  have hr := e_1699_3
  simp only [e_1699_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17070)) := by
  have c103857 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103858 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103859 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103860 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103861 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103862 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_4 := arithEq_of_rows h (row := 1699) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_4
  simp only [← c103857, ← c103858, ← c103859] at e_1699_4
  have e_1699_5 := arithEq_of_rows h (row := 1699) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_5
  simp only [← c103860, ← c103861, k2, ← c103862] at e_1699_5
  have hr := e_1699_5
  simp only [e_1699_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17071)) := by
  have c103863 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103864 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103865 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103866 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103867 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103868 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_6 := arithEq_of_rows h (row := 1699) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_6
  simp only [← c103863, ← c103864, ← c103865] at e_1699_6
  have e_1699_7 := arithEq_of_rows h (row := 1699) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_7
  simp only [← c103866, ← c103867, k2, ← c103868] at e_1699_7
  have hr := e_1699_7
  simp only [e_1699_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17072)) := by
  have c103869 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103870 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103871 := (publicBatchWrapper86_copies3245 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103872 := (publicBatchWrapper86_copies3246 a h).1
  have c103873 := (publicBatchWrapper86_copies3246 a h).2.1
  have c103874 := (publicBatchWrapper86_copies3246 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_8 := arithEq_of_rows h (row := 1699) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_8
  simp only [← c103869, ← c103870, ← c103871] at e_1699_8
  have e_1699_9 := arithEq_of_rows h (row := 1699) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_9
  simp only [← c103872, ← c103873, k2, ← c103874] at e_1699_9
  have hr := e_1699_9
  simp only [e_1699_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17073)) := by
  have c103875 := (publicBatchWrapper86_copies3246 a h).2.2.2.1
  have c103876 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.1
  have c103877 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.1
  have c103878 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.1
  have c103879 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.1
  have c103880 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_10 := arithEq_of_rows h (row := 1699) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_10
  simp only [← c103875, ← c103876, ← c103877] at e_1699_10
  have e_1699_11 := arithEq_of_rows h (row := 1699) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_11
  simp only [← c103878, ← c103879, k2, ← c103880] at e_1699_11
  have hr := e_1699_11
  simp only [e_1699_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17074)) := by
  have c103881 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.1
  have c103882 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103883 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103884 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103885 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103886 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_12 := arithEq_of_rows h (row := 1699) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_12
  simp only [← c103881, ← c103882, ← c103883] at e_1699_12
  have e_1699_13 := arithEq_of_rows h (row := 1699) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_13
  simp only [← c103884, ← c103885, k2, ← c103886] at e_1699_13
  have hr := e_1699_13
  simp only [e_1699_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17075)) := by
  have c103887 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103888 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103889 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103890 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103891 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103892 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_14 := arithEq_of_rows h (row := 1699) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_14
  simp only [← c103887, ← c103888, ← c103889] at e_1699_14
  have e_1699_15 := arithEq_of_rows h (row := 1699) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_15
  simp only [← c103890, ← c103891, k2, ← c103892] at e_1699_15
  have hr := e_1699_15
  simp only [e_1699_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17076)) := by
  have c103893 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103894 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103895 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103896 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103897 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103898 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_16 := arithEq_of_rows h (row := 1699) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_16
  simp only [← c103893, ← c103894, ← c103895] at e_1699_16
  have e_1699_17 := arithEq_of_rows h (row := 1699) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_17
  simp only [← c103896, ← c103897, k2, ← c103898] at e_1699_17
  have hr := e_1699_17
  simp only [e_1699_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1699 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17077)) := by
  have c103899 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103900 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103901 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103902 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103903 := (publicBatchWrapper86_copies3246 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103904 := (publicBatchWrapper86_copies3247 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1699_18 := arithEq_of_rows h (row := 1699) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_18
  simp only [← c103899, ← c103900, ← c103901] at e_1699_18
  have e_1699_19 := arithEq_of_rows h (row := 1699) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1699_19
  simp only [← c103902, ← c103903, k2, ← c103904] at e_1699_19
  have hr := e_1699_19
  simp only [e_1699_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17078)) := by
  have c103905 := (publicBatchWrapper86_copies3247 a h).2.1
  have c103906 := (publicBatchWrapper86_copies3247 a h).2.2.1
  have c103907 := (publicBatchWrapper86_copies3247 a h).2.2.2.1
  have c103908 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.1
  have c103909 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.1
  have c103910 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_0 := arithEq_of_rows h (row := 1700) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_0
  simp only [← c103905, ← c103906, ← c103907] at e_1700_0
  have e_1700_1 := arithEq_of_rows h (row := 1700) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_1
  simp only [← c103908, ← c103909, k2, ← c103910] at e_1700_1
  have hr := e_1700_1
  simp only [e_1700_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17079)) := by
  have c103911 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.1
  have c103912 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.1
  have c103913 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.1
  have c103914 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103915 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103916 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_2 := arithEq_of_rows h (row := 1700) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_2
  simp only [← c103911, ← c103912, ← c103913] at e_1700_2
  have e_1700_3 := arithEq_of_rows h (row := 1700) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_3
  simp only [← c103914, ← c103915, k2, ← c103916] at e_1700_3
  have hr := e_1700_3
  simp only [e_1700_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17080)) := by
  have c103917 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103918 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103919 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103920 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103921 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103922 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_4 := arithEq_of_rows h (row := 1700) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_4
  simp only [← c103917, ← c103918, ← c103919] at e_1700_4
  have e_1700_5 := arithEq_of_rows h (row := 1700) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_5
  simp only [← c103920, ← c103921, k2, ← c103922] at e_1700_5
  have hr := e_1700_5
  simp only [e_1700_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17081)) := by
  have c103923 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103924 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103925 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103926 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103927 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103928 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_6 := arithEq_of_rows h (row := 1700) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_6
  simp only [← c103923, ← c103924, ← c103925] at e_1700_6
  have e_1700_7 := arithEq_of_rows h (row := 1700) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_7
  simp only [← c103926, ← c103927, k2, ← c103928] at e_1700_7
  have hr := e_1700_7
  simp only [e_1700_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17082)) := by
  have c103929 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103930 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103931 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103932 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103933 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103934 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_8 := arithEq_of_rows h (row := 1700) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_8
  simp only [← c103929, ← c103930, ← c103931] at e_1700_8
  have e_1700_9 := arithEq_of_rows h (row := 1700) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_9
  simp only [← c103932, ← c103933, k2, ← c103934] at e_1700_9
  have hr := e_1700_9
  simp only [e_1700_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17083)) := by
  have c103935 := (publicBatchWrapper86_copies3247 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103936 := (publicBatchWrapper86_copies3248 a h).1
  have c103937 := (publicBatchWrapper86_copies3248 a h).2.1
  have c103938 := (publicBatchWrapper86_copies3248 a h).2.2.1
  have c103939 := (publicBatchWrapper86_copies3248 a h).2.2.2.1
  have c103940 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_10 := arithEq_of_rows h (row := 1700) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_10
  simp only [← c103935, ← c103936, ← c103937] at e_1700_10
  have e_1700_11 := arithEq_of_rows h (row := 1700) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_11
  simp only [← c103938, ← c103939, k2, ← c103940] at e_1700_11
  have hr := e_1700_11
  simp only [e_1700_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17084)) := by
  have c103941 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.1
  have c103942 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.1
  have c103943 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.1
  have c103944 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.1
  have c103945 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.1
  have c103946 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_12 := arithEq_of_rows h (row := 1700) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_12
  simp only [← c103941, ← c103942, ← c103943] at e_1700_12
  have e_1700_13 := arithEq_of_rows h (row := 1700) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_13
  simp only [← c103944, ← c103945, k2, ← c103946] at e_1700_13
  have hr := e_1700_13
  simp only [e_1700_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17085)) := by
  have c103947 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103948 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103949 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103950 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103951 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103952 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_14 := arithEq_of_rows h (row := 1700) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_14
  simp only [← c103947, ← c103948, ← c103949] at e_1700_14
  have e_1700_15 := arithEq_of_rows h (row := 1700) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_15
  simp only [← c103950, ← c103951, k2, ← c103952] at e_1700_15
  have hr := e_1700_15
  simp only [e_1700_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17086)) := by
  have c103953 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103954 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103955 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103956 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103957 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103958 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_16 := arithEq_of_rows h (row := 1700) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_16
  simp only [← c103953, ← c103954, ← c103955] at e_1700_16
  have e_1700_17 := arithEq_of_rows h (row := 1700) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_17
  simp only [← c103956, ← c103957, k2, ← c103958] at e_1700_17
  have hr := e_1700_17
  simp only [e_1700_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1700 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17087)) := by
  have c103959 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103960 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103961 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103962 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103963 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103964 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1700_18 := arithEq_of_rows h (row := 1700) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_18
  simp only [← c103959, ← c103960, ← c103961] at e_1700_18
  have e_1700_19 := arithEq_of_rows h (row := 1700) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1700_19
  simp only [← c103962, ← c103963, k2, ← c103964] at e_1700_19
  have hr := e_1700_19
  simp only [e_1700_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17088)) := by
  have c103965 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103966 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103967 := (publicBatchWrapper86_copies3248 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c103968 := (publicBatchWrapper86_copies3249 a h).1
  have c103969 := (publicBatchWrapper86_copies3249 a h).2.1
  have c103970 := (publicBatchWrapper86_copies3249 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_0 := arithEq_of_rows h (row := 1701) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_0
  simp only [← c103965, ← c103966, ← c103967] at e_1701_0
  have e_1701_1 := arithEq_of_rows h (row := 1701) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_1
  simp only [← c103968, ← c103969, k2, ← c103970] at e_1701_1
  have hr := e_1701_1
  simp only [e_1701_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17089)) := by
  have c103971 := (publicBatchWrapper86_copies3249 a h).2.2.2.1
  have c103972 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.1
  have c103973 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.1
  have c103974 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.1
  have c103975 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.1
  have c103976 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_2 := arithEq_of_rows h (row := 1701) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_2
  simp only [← c103971, ← c103972, ← c103973] at e_1701_2
  have e_1701_3 := arithEq_of_rows h (row := 1701) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_3
  simp only [← c103974, ← c103975, k2, ← c103976] at e_1701_3
  have hr := e_1701_3
  simp only [e_1701_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17090)) := by
  have c103977 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.1
  have c103978 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.1
  have c103979 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c103980 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103981 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103982 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_4 := arithEq_of_rows h (row := 1701) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_4
  simp only [← c103977, ← c103978, ← c103979] at e_1701_4
  have e_1701_5 := arithEq_of_rows h (row := 1701) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_5
  simp only [← c103980, ← c103981, k2, ← c103982] at e_1701_5
  have hr := e_1701_5
  simp only [e_1701_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17091)) := by
  have c103983 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103984 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103985 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103986 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103987 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103988 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_6 := arithEq_of_rows h (row := 1701) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_6
  simp only [← c103983, ← c103984, ← c103985] at e_1701_6
  have e_1701_7 := arithEq_of_rows h (row := 1701) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_7
  simp only [← c103986, ← c103987, k2, ← c103988] at e_1701_7
  have hr := e_1701_7
  simp only [e_1701_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17092)) := by
  have c103989 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103990 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103991 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103992 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103993 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103994 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_8 := arithEq_of_rows h (row := 1701) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_8
  simp only [← c103989, ← c103990, ← c103991] at e_1701_8
  have e_1701_9 := arithEq_of_rows h (row := 1701) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_9
  simp only [← c103992, ← c103993, k2, ← c103994] at e_1701_9
  have hr := e_1701_9
  simp only [e_1701_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17093)) := by
  have c103995 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103996 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103997 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103998 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c103999 := (publicBatchWrapper86_copies3249 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104000 := (publicBatchWrapper86_copies3250 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_10 := arithEq_of_rows h (row := 1701) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_10
  simp only [← c103995, ← c103996, ← c103997] at e_1701_10
  have e_1701_11 := arithEq_of_rows h (row := 1701) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_11
  simp only [← c103998, ← c103999, k2, ← c104000] at e_1701_11
  have hr := e_1701_11
  simp only [e_1701_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17094)) := by
  have c104001 := (publicBatchWrapper86_copies3250 a h).2.1
  have c104002 := (publicBatchWrapper86_copies3250 a h).2.2.1
  have c104003 := (publicBatchWrapper86_copies3250 a h).2.2.2.1
  have c104004 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.1
  have c104005 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.1
  have c104006 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_12 := arithEq_of_rows h (row := 1701) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_12
  simp only [← c104001, ← c104002, ← c104003] at e_1701_12
  have e_1701_13 := arithEq_of_rows h (row := 1701) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_13
  simp only [← c104004, ← c104005, k2, ← c104006] at e_1701_13
  have hr := e_1701_13
  simp only [e_1701_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17095)) := by
  have c104007 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.1
  have c104008 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.1
  have c104009 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.1
  have c104010 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104011 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104012 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_14 := arithEq_of_rows h (row := 1701) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_14
  simp only [← c104007, ← c104008, ← c104009] at e_1701_14
  have e_1701_15 := arithEq_of_rows h (row := 1701) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_15
  simp only [← c104010, ← c104011, k2, ← c104012] at e_1701_15
  have hr := e_1701_15
  simp only [e_1701_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17096)) := by
  have c104013 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104014 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104015 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104016 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104017 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104018 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_16 := arithEq_of_rows h (row := 1701) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_16
  simp only [← c104013, ← c104014, ← c104015] at e_1701_16
  have e_1701_17 := arithEq_of_rows h (row := 1701) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_17
  simp only [← c104016, ← c104017, k2, ← c104018] at e_1701_17
  have hr := e_1701_17
  simp only [e_1701_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1701 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17097)) := by
  have c104019 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104020 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104021 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104022 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104023 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104024 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1701_18 := arithEq_of_rows h (row := 1701) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_18
  simp only [← c104019, ← c104020, ← c104021] at e_1701_18
  have e_1701_19 := arithEq_of_rows h (row := 1701) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1701_19
  simp only [← c104022, ← c104023, k2, ← c104024] at e_1701_19
  have hr := e_1701_19
  simp only [e_1701_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17098)) := by
  have c104025 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104026 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104027 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104028 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104029 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104030 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_0 := arithEq_of_rows h (row := 1702) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_0
  simp only [← c104025, ← c104026, ← c104027] at e_1702_0
  have e_1702_1 := arithEq_of_rows h (row := 1702) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_1
  simp only [← c104028, ← c104029, k2, ← c104030] at e_1702_1
  have hr := e_1702_1
  simp only [e_1702_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17099)) := by
  have c104031 := (publicBatchWrapper86_copies3250 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104032 := (publicBatchWrapper86_copies3251 a h).1
  have c104033 := (publicBatchWrapper86_copies3251 a h).2.1
  have c104034 := (publicBatchWrapper86_copies3251 a h).2.2.1
  have c104035 := (publicBatchWrapper86_copies3251 a h).2.2.2.1
  have c104036 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_2 := arithEq_of_rows h (row := 1702) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_2
  simp only [← c104031, ← c104032, ← c104033] at e_1702_2
  have e_1702_3 := arithEq_of_rows h (row := 1702) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_3
  simp only [← c104034, ← c104035, k2, ← c104036] at e_1702_3
  have hr := e_1702_3
  simp only [e_1702_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17100)) := by
  have c104037 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.1
  have c104038 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.1
  have c104039 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.1
  have c104040 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.1
  have c104041 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.1
  have c104042 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_4 := arithEq_of_rows h (row := 1702) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_4
  simp only [← c104037, ← c104038, ← c104039] at e_1702_4
  have e_1702_5 := arithEq_of_rows h (row := 1702) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_5
  simp only [← c104040, ← c104041, k2, ← c104042] at e_1702_5
  have hr := e_1702_5
  simp only [e_1702_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17101)) := by
  have c104043 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104044 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104045 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104046 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104047 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104048 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_6 := arithEq_of_rows h (row := 1702) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_6
  simp only [← c104043, ← c104044, ← c104045] at e_1702_6
  have e_1702_7 := arithEq_of_rows h (row := 1702) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_7
  simp only [← c104046, ← c104047, k2, ← c104048] at e_1702_7
  have hr := e_1702_7
  simp only [e_1702_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17102)) := by
  have c104049 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104050 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104051 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104052 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104053 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104054 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_8 := arithEq_of_rows h (row := 1702) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_8
  simp only [← c104049, ← c104050, ← c104051] at e_1702_8
  have e_1702_9 := arithEq_of_rows h (row := 1702) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_9
  simp only [← c104052, ← c104053, k2, ← c104054] at e_1702_9
  have hr := e_1702_9
  simp only [e_1702_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17103)) := by
  have c104055 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104056 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104057 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104058 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104059 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104060 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_10 := arithEq_of_rows h (row := 1702) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_10
  simp only [← c104055, ← c104056, ← c104057] at e_1702_10
  have e_1702_11 := arithEq_of_rows h (row := 1702) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_11
  simp only [← c104058, ← c104059, k2, ← c104060] at e_1702_11
  have hr := e_1702_11
  simp only [e_1702_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17104)) := by
  have c104061 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104062 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104063 := (publicBatchWrapper86_copies3251 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104064 := (publicBatchWrapper86_copies3252 a h).1
  have c104065 := (publicBatchWrapper86_copies3252 a h).2.1
  have c104066 := (publicBatchWrapper86_copies3252 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_12 := arithEq_of_rows h (row := 1702) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_12
  simp only [← c104061, ← c104062, ← c104063] at e_1702_12
  have e_1702_13 := arithEq_of_rows h (row := 1702) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_13
  simp only [← c104064, ← c104065, k2, ← c104066] at e_1702_13
  have hr := e_1702_13
  simp only [e_1702_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17105)) := by
  have c104067 := (publicBatchWrapper86_copies3252 a h).2.2.2.1
  have c104068 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.1
  have c104069 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.1
  have c104070 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.1
  have c104071 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.1
  have c104072 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_14 := arithEq_of_rows h (row := 1702) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_14
  simp only [← c104067, ← c104068, ← c104069] at e_1702_14
  have e_1702_15 := arithEq_of_rows h (row := 1702) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_15
  simp only [← c104070, ← c104071, k2, ← c104072] at e_1702_15
  have hr := e_1702_15
  simp only [e_1702_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17106)) := by
  have c104073 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.1
  have c104074 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104075 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104076 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104077 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104078 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_16 := arithEq_of_rows h (row := 1702) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_16
  simp only [← c104073, ← c104074, ← c104075] at e_1702_16
  have e_1702_17 := arithEq_of_rows h (row := 1702) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_17
  simp only [← c104076, ← c104077, k2, ← c104078] at e_1702_17
  have hr := e_1702_17
  simp only [e_1702_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1702 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17107)) := by
  have c104079 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104080 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104081 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104082 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104083 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104084 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1702_18 := arithEq_of_rows h (row := 1702) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_18
  simp only [← c104079, ← c104080, ← c104081] at e_1702_18
  have e_1702_19 := arithEq_of_rows h (row := 1702) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1702_19
  simp only [← c104082, ← c104083, k2, ← c104084] at e_1702_19
  have hr := e_1702_19
  simp only [e_1702_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17108)) := by
  have c104085 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104086 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104087 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104088 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104089 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104090 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_0 := arithEq_of_rows h (row := 1703) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_0
  simp only [← c104085, ← c104086, ← c104087] at e_1703_0
  have e_1703_1 := arithEq_of_rows h (row := 1703) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_1
  simp only [← c104088, ← c104089, k2, ← c104090] at e_1703_1
  have hr := e_1703_1
  simp only [e_1703_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17109)) := by
  have c104091 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104092 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104093 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104094 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104095 := (publicBatchWrapper86_copies3252 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104096 := (publicBatchWrapper86_copies3253 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_2 := arithEq_of_rows h (row := 1703) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_2
  simp only [← c104091, ← c104092, ← c104093] at e_1703_2
  have e_1703_3 := arithEq_of_rows h (row := 1703) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_3
  simp only [← c104094, ← c104095, k2, ← c104096] at e_1703_3
  have hr := e_1703_3
  simp only [e_1703_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17110)) := by
  have c104097 := (publicBatchWrapper86_copies3253 a h).2.1
  have c104098 := (publicBatchWrapper86_copies3253 a h).2.2.1
  have c104099 := (publicBatchWrapper86_copies3253 a h).2.2.2.1
  have c104100 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.1
  have c104101 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.1
  have c104102 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_4 := arithEq_of_rows h (row := 1703) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_4
  simp only [← c104097, ← c104098, ← c104099] at e_1703_4
  have e_1703_5 := arithEq_of_rows h (row := 1703) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_5
  simp only [← c104100, ← c104101, k2, ← c104102] at e_1703_5
  have hr := e_1703_5
  simp only [e_1703_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17111)) := by
  have c104103 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.1
  have c104104 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.1
  have c104105 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.1
  have c104106 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104107 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104108 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_6 := arithEq_of_rows h (row := 1703) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_6
  simp only [← c104103, ← c104104, ← c104105] at e_1703_6
  have e_1703_7 := arithEq_of_rows h (row := 1703) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_7
  simp only [← c104106, ← c104107, k2, ← c104108] at e_1703_7
  have hr := e_1703_7
  simp only [e_1703_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17112)) := by
  have c104109 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104110 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104111 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104112 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104113 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104114 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_8 := arithEq_of_rows h (row := 1703) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_8
  simp only [← c104109, ← c104110, ← c104111] at e_1703_8
  have e_1703_9 := arithEq_of_rows h (row := 1703) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_9
  simp only [← c104112, ← c104113, k2, ← c104114] at e_1703_9
  have hr := e_1703_9
  simp only [e_1703_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17113)) := by
  have c104115 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104116 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104117 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104118 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104119 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104120 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_10 := arithEq_of_rows h (row := 1703) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_10
  simp only [← c104115, ← c104116, ← c104117] at e_1703_10
  have e_1703_11 := arithEq_of_rows h (row := 1703) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_11
  simp only [← c104118, ← c104119, k2, ← c104120] at e_1703_11
  have hr := e_1703_11
  simp only [e_1703_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17114)) := by
  have c104121 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104122 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104123 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104124 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104125 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104126 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_12 := arithEq_of_rows h (row := 1703) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_12
  simp only [← c104121, ← c104122, ← c104123] at e_1703_12
  have e_1703_13 := arithEq_of_rows h (row := 1703) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_13
  simp only [← c104124, ← c104125, k2, ← c104126] at e_1703_13
  have hr := e_1703_13
  simp only [e_1703_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17115)) := by
  have c104127 := (publicBatchWrapper86_copies3253 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104128 := (publicBatchWrapper86_copies3254 a h).1
  have c104129 := (publicBatchWrapper86_copies3254 a h).2.1
  have c104130 := (publicBatchWrapper86_copies3254 a h).2.2.1
  have c104131 := (publicBatchWrapper86_copies3254 a h).2.2.2.1
  have c104132 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_14 := arithEq_of_rows h (row := 1703) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_14
  simp only [← c104127, ← c104128, ← c104129] at e_1703_14
  have e_1703_15 := arithEq_of_rows h (row := 1703) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_15
  simp only [← c104130, ← c104131, k2, ← c104132] at e_1703_15
  have hr := e_1703_15
  simp only [e_1703_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17116)) := by
  have c104133 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.1
  have c104134 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.1
  have c104135 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.1
  have c104136 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.1
  have c104137 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.1
  have c104138 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_16 := arithEq_of_rows h (row := 1703) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_16
  simp only [← c104133, ← c104134, ← c104135] at e_1703_16
  have e_1703_17 := arithEq_of_rows h (row := 1703) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_17
  simp only [← c104136, ← c104137, k2, ← c104138] at e_1703_17
  have hr := e_1703_17
  simp only [e_1703_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1703 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17117)) := by
  have c104139 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104140 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104141 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104142 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104143 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104144 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1703_18 := arithEq_of_rows h (row := 1703) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_18
  simp only [← c104139, ← c104140, ← c104141] at e_1703_18
  have e_1703_19 := arithEq_of_rows h (row := 1703) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1703_19
  simp only [← c104142, ← c104143, k2, ← c104144] at e_1703_19
  have hr := e_1703_19
  simp only [e_1703_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17118)) := by
  have c104145 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104146 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104147 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104148 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104149 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104150 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_0 := arithEq_of_rows h (row := 1704) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_0
  simp only [← c104145, ← c104146, ← c104147] at e_1704_0
  have e_1704_1 := arithEq_of_rows h (row := 1704) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_1
  simp only [← c104148, ← c104149, k2, ← c104150] at e_1704_1
  have hr := e_1704_1
  simp only [e_1704_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17119)) := by
  have c104151 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104152 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104153 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104154 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104155 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104156 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_2 := arithEq_of_rows h (row := 1704) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_2
  simp only [← c104151, ← c104152, ← c104153] at e_1704_2
  have e_1704_3 := arithEq_of_rows h (row := 1704) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_3
  simp only [← c104154, ← c104155, k2, ← c104156] at e_1704_3
  have hr := e_1704_3
  simp only [e_1704_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17120)) := by
  have c104157 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104158 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104159 := (publicBatchWrapper86_copies3254 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104160 := (publicBatchWrapper86_copies3255 a h).1
  have c104161 := (publicBatchWrapper86_copies3255 a h).2.1
  have c104162 := (publicBatchWrapper86_copies3255 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_4 := arithEq_of_rows h (row := 1704) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_4
  simp only [← c104157, ← c104158, ← c104159] at e_1704_4
  have e_1704_5 := arithEq_of_rows h (row := 1704) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_5
  simp only [← c104160, ← c104161, k2, ← c104162] at e_1704_5
  have hr := e_1704_5
  simp only [e_1704_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17121)) := by
  have c104163 := (publicBatchWrapper86_copies3255 a h).2.2.2.1
  have c104164 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.1
  have c104165 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.1
  have c104166 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.1
  have c104167 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.1
  have c104168 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_6 := arithEq_of_rows h (row := 1704) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_6
  simp only [← c104163, ← c104164, ← c104165] at e_1704_6
  have e_1704_7 := arithEq_of_rows h (row := 1704) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_7
  simp only [← c104166, ← c104167, k2, ← c104168] at e_1704_7
  have hr := e_1704_7
  simp only [e_1704_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17122)) := by
  have c104169 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.1
  have c104170 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104171 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104172 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104173 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104174 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_8 := arithEq_of_rows h (row := 1704) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_8
  simp only [← c104169, ← c104170, ← c104171] at e_1704_8
  have e_1704_9 := arithEq_of_rows h (row := 1704) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_9
  simp only [← c104172, ← c104173, k2, ← c104174] at e_1704_9
  have hr := e_1704_9
  simp only [e_1704_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17123)) := by
  have c104175 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104176 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104177 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104178 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104179 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104180 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_10 := arithEq_of_rows h (row := 1704) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_10
  simp only [← c104175, ← c104176, ← c104177] at e_1704_10
  have e_1704_11 := arithEq_of_rows h (row := 1704) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_11
  simp only [← c104178, ← c104179, k2, ← c104180] at e_1704_11
  have hr := e_1704_11
  simp only [e_1704_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17124)) := by
  have c104181 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104182 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104183 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104184 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104185 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104186 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_12 := arithEq_of_rows h (row := 1704) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_12
  simp only [← c104181, ← c104182, ← c104183] at e_1704_12
  have e_1704_13 := arithEq_of_rows h (row := 1704) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_13
  simp only [← c104184, ← c104185, k2, ← c104186] at e_1704_13
  have hr := e_1704_13
  simp only [e_1704_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17125)) := by
  have c104187 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104188 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104189 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104190 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104191 := (publicBatchWrapper86_copies3255 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104192 := (publicBatchWrapper86_copies3256 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_14 := arithEq_of_rows h (row := 1704) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_14
  simp only [← c104187, ← c104188, ← c104189] at e_1704_14
  have e_1704_15 := arithEq_of_rows h (row := 1704) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_15
  simp only [← c104190, ← c104191, k2, ← c104192] at e_1704_15
  have hr := e_1704_15
  simp only [e_1704_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17126)) := by
  have c104193 := (publicBatchWrapper86_copies3256 a h).2.1
  have c104194 := (publicBatchWrapper86_copies3256 a h).2.2.1
  have c104195 := (publicBatchWrapper86_copies3256 a h).2.2.2.1
  have c104196 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.1
  have c104197 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.1
  have c104198 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_16 := arithEq_of_rows h (row := 1704) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_16
  simp only [← c104193, ← c104194, ← c104195] at e_1704_16
  have e_1704_17 := arithEq_of_rows h (row := 1704) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_17
  simp only [← c104196, ← c104197, k2, ← c104198] at e_1704_17
  have hr := e_1704_17
  simp only [e_1704_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1704 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17127)) := by
  have c104199 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.1
  have c104200 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.1
  have c104201 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.1
  have c104202 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104203 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104204 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1704_18 := arithEq_of_rows h (row := 1704) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_18
  simp only [← c104199, ← c104200, ← c104201] at e_1704_18
  have e_1704_19 := arithEq_of_rows h (row := 1704) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1704_19
  simp only [← c104202, ← c104203, k2, ← c104204] at e_1704_19
  have hr := e_1704_19
  simp only [e_1704_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17128)) := by
  have c104205 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104206 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104207 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104208 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104209 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104210 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_0 := arithEq_of_rows h (row := 1705) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_0
  simp only [← c104205, ← c104206, ← c104207] at e_1705_0
  have e_1705_1 := arithEq_of_rows h (row := 1705) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_1
  simp only [← c104208, ← c104209, k2, ← c104210] at e_1705_1
  have hr := e_1705_1
  simp only [e_1705_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17129)) := by
  have c104211 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104212 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104213 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104214 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104215 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104216 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_2 := arithEq_of_rows h (row := 1705) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_2
  simp only [← c104211, ← c104212, ← c104213] at e_1705_2
  have e_1705_3 := arithEq_of_rows h (row := 1705) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_3
  simp only [← c104214, ← c104215, k2, ← c104216] at e_1705_3
  have hr := e_1705_3
  simp only [e_1705_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17130)) := by
  have c104217 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104218 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104219 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104220 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104221 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104222 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_4 := arithEq_of_rows h (row := 1705) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_4
  simp only [← c104217, ← c104218, ← c104219] at e_1705_4
  have e_1705_5 := arithEq_of_rows h (row := 1705) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_5
  simp only [← c104220, ← c104221, k2, ← c104222] at e_1705_5
  have hr := e_1705_5
  simp only [e_1705_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17131)) := by
  have c104223 := (publicBatchWrapper86_copies3256 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104224 := (publicBatchWrapper86_copies3257 a h).1
  have c104225 := (publicBatchWrapper86_copies3257 a h).2.1
  have c104226 := (publicBatchWrapper86_copies3257 a h).2.2.1
  have c104227 := (publicBatchWrapper86_copies3257 a h).2.2.2.1
  have c104228 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_6 := arithEq_of_rows h (row := 1705) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_6
  simp only [← c104223, ← c104224, ← c104225] at e_1705_6
  have e_1705_7 := arithEq_of_rows h (row := 1705) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_7
  simp only [← c104226, ← c104227, k2, ← c104228] at e_1705_7
  have hr := e_1705_7
  simp only [e_1705_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17132)) := by
  have c104229 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.1
  have c104230 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.1
  have c104231 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.1
  have c104232 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.1
  have c104233 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.1
  have c104234 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_8 := arithEq_of_rows h (row := 1705) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_8
  simp only [← c104229, ← c104230, ← c104231] at e_1705_8
  have e_1705_9 := arithEq_of_rows h (row := 1705) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_9
  simp only [← c104232, ← c104233, k2, ← c104234] at e_1705_9
  have hr := e_1705_9
  simp only [e_1705_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17133)) := by
  have c104235 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104236 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104237 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104238 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104239 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104240 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_10 := arithEq_of_rows h (row := 1705) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_10
  simp only [← c104235, ← c104236, ← c104237] at e_1705_10
  have e_1705_11 := arithEq_of_rows h (row := 1705) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_11
  simp only [← c104238, ← c104239, k2, ← c104240] at e_1705_11
  have hr := e_1705_11
  simp only [e_1705_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17134)) := by
  have c104241 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104242 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104243 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104244 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104245 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104246 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_12 := arithEq_of_rows h (row := 1705) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_12
  simp only [← c104241, ← c104242, ← c104243] at e_1705_12
  have e_1705_13 := arithEq_of_rows h (row := 1705) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_13
  simp only [← c104244, ← c104245, k2, ← c104246] at e_1705_13
  have hr := e_1705_13
  simp only [e_1705_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17135)) := by
  have c104247 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104248 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104249 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104250 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104251 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104252 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_14 := arithEq_of_rows h (row := 1705) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_14
  simp only [← c104247, ← c104248, ← c104249] at e_1705_14
  have e_1705_15 := arithEq_of_rows h (row := 1705) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_15
  simp only [← c104250, ← c104251, k2, ← c104252] at e_1705_15
  have hr := e_1705_15
  simp only [e_1705_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17136)) := by
  have c104253 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104254 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104255 := (publicBatchWrapper86_copies3257 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104256 := (publicBatchWrapper86_copies3258 a h).1
  have c104257 := (publicBatchWrapper86_copies3258 a h).2.1
  have c104258 := (publicBatchWrapper86_copies3258 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_16 := arithEq_of_rows h (row := 1705) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_16
  simp only [← c104253, ← c104254, ← c104255] at e_1705_16
  have e_1705_17 := arithEq_of_rows h (row := 1705) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_17
  simp only [← c104256, ← c104257, k2, ← c104258] at e_1705_17
  have hr := e_1705_17
  simp only [e_1705_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1705 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17137)) := by
  have c104259 := (publicBatchWrapper86_copies3258 a h).2.2.2.1
  have c104260 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.1
  have c104261 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.1
  have c104262 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.1
  have c104263 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.1
  have c104264 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1705_18 := arithEq_of_rows h (row := 1705) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_18
  simp only [← c104259, ← c104260, ← c104261] at e_1705_18
  have e_1705_19 := arithEq_of_rows h (row := 1705) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1705_19
  simp only [← c104262, ← c104263, k2, ← c104264] at e_1705_19
  have hr := e_1705_19
  simp only [e_1705_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17138)) := by
  have c104265 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.1
  have c104266 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104267 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104268 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104269 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104270 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_0 := arithEq_of_rows h (row := 1706) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_0
  simp only [← c104265, ← c104266, ← c104267] at e_1706_0
  have e_1706_1 := arithEq_of_rows h (row := 1706) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_1
  simp only [← c104268, ← c104269, k2, ← c104270] at e_1706_1
  have hr := e_1706_1
  simp only [e_1706_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17139)) := by
  have c104271 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104272 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104273 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104274 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104275 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104276 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_2 := arithEq_of_rows h (row := 1706) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_2
  simp only [← c104271, ← c104272, ← c104273] at e_1706_2
  have e_1706_3 := arithEq_of_rows h (row := 1706) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_3
  simp only [← c104274, ← c104275, k2, ← c104276] at e_1706_3
  have hr := e_1706_3
  simp only [e_1706_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17140)) := by
  have c104277 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104278 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104279 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104280 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104281 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104282 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_4 := arithEq_of_rows h (row := 1706) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_4
  simp only [← c104277, ← c104278, ← c104279] at e_1706_4
  have e_1706_5 := arithEq_of_rows h (row := 1706) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_5
  simp only [← c104280, ← c104281, k2, ← c104282] at e_1706_5
  have hr := e_1706_5
  simp only [e_1706_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17141)) := by
  have c104283 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104284 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104285 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104286 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104287 := (publicBatchWrapper86_copies3258 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104288 := (publicBatchWrapper86_copies3259 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_6 := arithEq_of_rows h (row := 1706) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_6
  simp only [← c104283, ← c104284, ← c104285] at e_1706_6
  have e_1706_7 := arithEq_of_rows h (row := 1706) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_7
  simp only [← c104286, ← c104287, k2, ← c104288] at e_1706_7
  have hr := e_1706_7
  simp only [e_1706_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17142)) := by
  have c104289 := (publicBatchWrapper86_copies3259 a h).2.1
  have c104290 := (publicBatchWrapper86_copies3259 a h).2.2.1
  have c104291 := (publicBatchWrapper86_copies3259 a h).2.2.2.1
  have c104292 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.1
  have c104293 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.1
  have c104294 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_8 := arithEq_of_rows h (row := 1706) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_8
  simp only [← c104289, ← c104290, ← c104291] at e_1706_8
  have e_1706_9 := arithEq_of_rows h (row := 1706) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_9
  simp only [← c104292, ← c104293, k2, ← c104294] at e_1706_9
  have hr := e_1706_9
  simp only [e_1706_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17143)) := by
  have c104295 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.1
  have c104296 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.1
  have c104297 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.1
  have c104298 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104299 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104300 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_10 := arithEq_of_rows h (row := 1706) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_10
  simp only [← c104295, ← c104296, ← c104297] at e_1706_10
  have e_1706_11 := arithEq_of_rows h (row := 1706) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_11
  simp only [← c104298, ← c104299, k2, ← c104300] at e_1706_11
  have hr := e_1706_11
  simp only [e_1706_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17144)) := by
  have c104301 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104302 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104303 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104304 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104305 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104306 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_12 := arithEq_of_rows h (row := 1706) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_12
  simp only [← c104301, ← c104302, ← c104303] at e_1706_12
  have e_1706_13 := arithEq_of_rows h (row := 1706) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_13
  simp only [← c104304, ← c104305, k2, ← c104306] at e_1706_13
  have hr := e_1706_13
  simp only [e_1706_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17145)) := by
  have c104307 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104308 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104309 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104310 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104311 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104312 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_14 := arithEq_of_rows h (row := 1706) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_14
  simp only [← c104307, ← c104308, ← c104309] at e_1706_14
  have e_1706_15 := arithEq_of_rows h (row := 1706) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_15
  simp only [← c104310, ← c104311, k2, ← c104312] at e_1706_15
  have hr := e_1706_15
  simp only [e_1706_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17146)) := by
  have c104313 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104314 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104315 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104316 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104317 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104318 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_16 := arithEq_of_rows h (row := 1706) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_16
  simp only [← c104313, ← c104314, ← c104315] at e_1706_16
  have e_1706_17 := arithEq_of_rows h (row := 1706) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_17
  simp only [← c104316, ← c104317, k2, ← c104318] at e_1706_17
  have hr := e_1706_17
  simp only [e_1706_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1706 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17147)) := by
  have c104319 := (publicBatchWrapper86_copies3259 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104320 := (publicBatchWrapper86_copies3260 a h).1
  have c104321 := (publicBatchWrapper86_copies3260 a h).2.1
  have c104322 := (publicBatchWrapper86_copies3260 a h).2.2.1
  have c104323 := (publicBatchWrapper86_copies3260 a h).2.2.2.1
  have c104324 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1706_18 := arithEq_of_rows h (row := 1706) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_18
  simp only [← c104319, ← c104320, ← c104321] at e_1706_18
  have e_1706_19 := arithEq_of_rows h (row := 1706) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1706_19
  simp only [← c104322, ← c104323, k2, ← c104324] at e_1706_19
  have hr := e_1706_19
  simp only [e_1706_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17148)) := by
  have c104325 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.1
  have c104326 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.1
  have c104327 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.1
  have c104328 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.1
  have c104329 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.1
  have c104330 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_0 := arithEq_of_rows h (row := 1707) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_0
  simp only [← c104325, ← c104326, ← c104327] at e_1707_0
  have e_1707_1 := arithEq_of_rows h (row := 1707) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_1
  simp only [← c104328, ← c104329, k2, ← c104330] at e_1707_1
  have hr := e_1707_1
  simp only [e_1707_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17149)) := by
  have c104331 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104332 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104333 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104334 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104335 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104336 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_2 := arithEq_of_rows h (row := 1707) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_2
  simp only [← c104331, ← c104332, ← c104333] at e_1707_2
  have e_1707_3 := arithEq_of_rows h (row := 1707) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_3
  simp only [← c104334, ← c104335, k2, ← c104336] at e_1707_3
  have hr := e_1707_3
  simp only [e_1707_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17150)) := by
  have c104337 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104338 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104339 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104340 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104341 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104342 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_4 := arithEq_of_rows h (row := 1707) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_4
  simp only [← c104337, ← c104338, ← c104339] at e_1707_4
  have e_1707_5 := arithEq_of_rows h (row := 1707) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_5
  simp only [← c104340, ← c104341, k2, ← c104342] at e_1707_5
  have hr := e_1707_5
  simp only [e_1707_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17151)) := by
  have c104343 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104344 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104345 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104346 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104347 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104348 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_6 := arithEq_of_rows h (row := 1707) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_6
  simp only [← c104343, ← c104344, ← c104345] at e_1707_6
  have e_1707_7 := arithEq_of_rows h (row := 1707) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_7
  simp only [← c104346, ← c104347, k2, ← c104348] at e_1707_7
  have hr := e_1707_7
  simp only [e_1707_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17152)) := by
  have c104349 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104350 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104351 := (publicBatchWrapper86_copies3260 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104352 := (publicBatchWrapper86_copies3261 a h).1
  have c104353 := (publicBatchWrapper86_copies3261 a h).2.1
  have c104354 := (publicBatchWrapper86_copies3261 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_8 := arithEq_of_rows h (row := 1707) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_8
  simp only [← c104349, ← c104350, ← c104351] at e_1707_8
  have e_1707_9 := arithEq_of_rows h (row := 1707) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_9
  simp only [← c104352, ← c104353, k2, ← c104354] at e_1707_9
  have hr := e_1707_9
  simp only [e_1707_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16667 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17153)) := by
  have c104355 := (publicBatchWrapper86_copies3261 a h).2.2.2.1
  have c104356 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.1
  have c104357 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.1
  have c104358 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.1
  have c104359 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.1
  have c104360 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_10 := arithEq_of_rows h (row := 1707) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_10
  simp only [← c104355, ← c104356, ← c104357] at e_1707_10
  have e_1707_11 := arithEq_of_rows h (row := 1707) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_11
  simp only [← c104358, ← c104359, k2, ← c104360] at e_1707_11
  have hr := e_1707_11
  simp only [e_1707_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16668 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17154)) := by
  have c104361 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.1
  have c104362 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104363 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104364 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104365 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104366 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_12 := arithEq_of_rows h (row := 1707) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_12
  simp only [← c104361, ← c104362, ← c104363] at e_1707_12
  have e_1707_13 := arithEq_of_rows h (row := 1707) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_13
  simp only [← c104364, ← c104365, k2, ← c104366] at e_1707_13
  have hr := e_1707_13
  simp only [e_1707_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16669 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17155)) := by
  have c104367 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104368 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104369 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104370 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104371 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104372 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_14 := arithEq_of_rows h (row := 1707) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_14
  simp only [← c104367, ← c104368, ← c104369] at e_1707_14
  have e_1707_15 := arithEq_of_rows h (row := 1707) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_15
  simp only [← c104370, ← c104371, k2, ← c104372] at e_1707_15
  have hr := e_1707_15
  simp only [e_1707_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16670 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17156)) := by
  have c104373 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104374 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104375 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104376 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104377 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104378 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_16 := arithEq_of_rows h (row := 1707) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_16
  simp only [← c104373, ← c104374, ← c104375] at e_1707_16
  have e_1707_17 := arithEq_of_rows h (row := 1707) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_17
  simp only [← c104376, ← c104377, k2, ← c104378] at e_1707_17
  have hr := e_1707_17
  simp only [e_1707_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16671 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1707 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17157)) := by
  have c104379 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104380 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104381 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104382 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104383 := (publicBatchWrapper86_copies3261 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104384 := (publicBatchWrapper86_copies3262 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1707_18 := arithEq_of_rows h (row := 1707) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_18
  simp only [← c104379, ← c104380, ← c104381] at e_1707_18
  have e_1707_19 := arithEq_of_rows h (row := 1707) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1707_19
  simp only [← c104382, ← c104383, k2, ← c104384] at e_1707_19
  have hr := e_1707_19
  simp only [e_1707_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16672 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17158)) := by
  have c104385 := (publicBatchWrapper86_copies3262 a h).2.1
  have c104386 := (publicBatchWrapper86_copies3262 a h).2.2.1
  have c104387 := (publicBatchWrapper86_copies3262 a h).2.2.2.1
  have c104388 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.1
  have c104389 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.1
  have c104390 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_0 := arithEq_of_rows h (row := 1708) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_0
  simp only [← c104385, ← c104386, ← c104387] at e_1708_0
  have e_1708_1 := arithEq_of_rows h (row := 1708) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_1
  simp only [← c104388, ← c104389, k2, ← c104390] at e_1708_1
  have hr := e_1708_1
  simp only [e_1708_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16673 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17159)) := by
  have c104391 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.1
  have c104392 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.1
  have c104393 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.1
  have c104394 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104395 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104396 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_2 := arithEq_of_rows h (row := 1708) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_2
  simp only [← c104391, ← c104392, ← c104393] at e_1708_2
  have e_1708_3 := arithEq_of_rows h (row := 1708) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_3
  simp only [← c104394, ← c104395, k2, ← c104396] at e_1708_3
  have hr := e_1708_3
  simp only [e_1708_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16674 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 23) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17160)) := by
  have c104397 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104398 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104399 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104400 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104401 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104402 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_4 := arithEq_of_rows h (row := 1708) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_4
  simp only [← c104397, ← c104398, ← c104399] at e_1708_4
  have e_1708_5 := arithEq_of_rows h (row := 1708) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_5
  simp only [← c104400, ← c104401, k2, ← c104402] at e_1708_5
  have hr := e_1708_5
  simp only [e_1708_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16675 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 31) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17161)) := by
  have c104403 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104404 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104405 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104406 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104407 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104408 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_6 := arithEq_of_rows h (row := 1708) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_6
  simp only [← c104403, ← c104404, ← c104405] at e_1708_6
  have e_1708_7 := arithEq_of_rows h (row := 1708) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_7
  simp only [← c104406, ← c104407, k2, ← c104408] at e_1708_7
  have hr := e_1708_7
  simp only [e_1708_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16676 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 39) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17162)) := by
  have c104409 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104410 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104411 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104412 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104413 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104414 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_8 := arithEq_of_rows h (row := 1708) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_8
  simp only [← c104409, ← c104410, ← c104411] at e_1708_8
  have e_1708_9 := arithEq_of_rows h (row := 1708) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_9
  simp only [← c104412, ← c104413, k2, ← c104414] at e_1708_9
  have hr := e_1708_9
  simp only [e_1708_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16677 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 47) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17163)) := by
  have c104415 := (publicBatchWrapper86_copies3262 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104416 := (publicBatchWrapper86_copies3263 a h).1
  have c104417 := (publicBatchWrapper86_copies3263 a h).2.1
  have c104418 := (publicBatchWrapper86_copies3263 a h).2.2.1
  have c104419 := (publicBatchWrapper86_copies3263 a h).2.2.2.1
  have c104420 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_10 := arithEq_of_rows h (row := 1708) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_10
  simp only [← c104415, ← c104416, ← c104417] at e_1708_10
  have e_1708_11 := arithEq_of_rows h (row := 1708) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_11
  simp only [← c104418, ← c104419, k2, ← c104420] at e_1708_11
  have hr := e_1708_11
  simp only [e_1708_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16678 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 55) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17164)) := by
  have c104421 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.1
  have c104422 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.1
  have c104423 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.1
  have c104424 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.1
  have c104425 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.1
  have c104426 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_12 := arithEq_of_rows h (row := 1708) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_12
  simp only [← c104421, ← c104422, ← c104423] at e_1708_12
  have e_1708_13 := arithEq_of_rows h (row := 1708) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_13
  simp only [← c104424, ← c104425, k2, ← c104426] at e_1708_13
  have hr := e_1708_13
  simp only [e_1708_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16679 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 63) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17165)) := by
  have c104427 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104428 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104429 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104430 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104431 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104432 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_14 := arithEq_of_rows h (row := 1708) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_14
  simp only [← c104427, ← c104428, ← c104429] at e_1708_14
  have e_1708_15 := arithEq_of_rows h (row := 1708) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_15
  simp only [← c104430, ← c104431, k2, ← c104432] at e_1708_15
  have hr := e_1708_15
  simp only [e_1708_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16680 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 71) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17166)) := by
  have c104433 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104434 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104435 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104436 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104437 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104438 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_16 := arithEq_of_rows h (row := 1708) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_16
  simp only [← c104433, ← c104434, ← c104435] at e_1708_16
  have e_1708_17 := arithEq_of_rows h (row := 1708) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_17
  simp only [← c104436, ← c104437, k2, ← c104438] at e_1708_17
  have hr := e_1708_17
  simp only [e_1708_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16681 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1708 79) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17167)) := by
  have c104439 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104440 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104441 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104442 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104443 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104444 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1708_18 := arithEq_of_rows h (row := 1708) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_18
  simp only [← c104439, ← c104440, ← c104441] at e_1708_18
  have e_1708_19 := arithEq_of_rows h (row := 1708) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1708_19
  simp only [← c104442, ← c104443, k2, ← c104444] at e_1708_19
  have hr := e_1708_19
  simp only [e_1708_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16682 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 7) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17168)) := by
  have c104445 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104446 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104447 := (publicBatchWrapper86_copies3263 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104448 := (publicBatchWrapper86_copies3264 a h).1
  have c104449 := (publicBatchWrapper86_copies3264 a h).2.1
  have c104450 := (publicBatchWrapper86_copies3264 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_0 := arithEq_of_rows h (row := 1709) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_0
  simp only [← c104445, ← c104446, ← c104447] at e_1709_0
  have e_1709_1 := arithEq_of_rows h (row := 1709) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_1
  simp only [← c104448, ← c104449, k2, ← c104450] at e_1709_1
  have hr := e_1709_1
  simp only [e_1709_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16683 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 15) = bselect (a (.wire 84 7)) (a (.virt 200)) (a (.virt 17169)) := by
  have c104451 := (publicBatchWrapper86_copies3264 a h).2.2.2.1
  have c104452 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.1
  have c104453 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.1
  have c104454 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.1
  have c104455 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.1
  have c104456 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_2 := arithEq_of_rows h (row := 1709) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_2
  simp only [← c104451, ← c104452, ← c104453] at e_1709_2
  have e_1709_3 := arithEq_of_rows h (row := 1709) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_3
  simp only [← c104454, ← c104455, k2, ← c104456] at e_1709_3
  have hr := e_1709_3
  simp only [e_1709_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16684 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 168)) := by
  have c104457 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.1
  have c104458 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104459 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104460 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104461 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104462 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_4 := arithEq_of_rows h (row := 1709) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_4
  simp only [← c104457, ← c104458, ← c104459] at e_1709_4
  have e_1709_5 := arithEq_of_rows h (row := 1709) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_5
  simp only [← c104460, ← c104461, k2, ← c104462] at e_1709_5
  have hr := e_1709_5
  simp only [e_1709_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16685 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 169)) := by
  have c104463 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104464 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104465 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104466 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104467 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104468 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_6 := arithEq_of_rows h (row := 1709) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_6
  simp only [← c104463, ← c104464, ← c104465] at e_1709_6
  have e_1709_7 := arithEq_of_rows h (row := 1709) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_7
  simp only [← c104466, ← c104467, k2, ← c104468] at e_1709_7
  have hr := e_1709_7
  simp only [e_1709_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16686 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 170)) := by
  have c104469 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104470 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104471 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104472 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104473 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104474 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_8 := arithEq_of_rows h (row := 1709) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_8
  simp only [← c104469, ← c104470, ← c104471] at e_1709_8
  have e_1709_9 := arithEq_of_rows h (row := 1709) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_9
  simp only [← c104472, ← c104473, k2, ← c104474] at e_1709_9
  have hr := e_1709_9
  simp only [e_1709_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16687 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 171)) := by
  have c104475 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104476 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104477 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104478 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104479 := (publicBatchWrapper86_copies3264 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104480 := (publicBatchWrapper86_copies3265 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_10 := arithEq_of_rows h (row := 1709) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_10
  simp only [← c104475, ← c104476, ← c104477] at e_1709_10
  have e_1709_11 := arithEq_of_rows h (row := 1709) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_11
  simp only [← c104478, ← c104479, k2, ← c104480] at e_1709_11
  have hr := e_1709_11
  simp only [e_1709_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16688 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 172)) := by
  have c104481 := (publicBatchWrapper86_copies3265 a h).2.1
  have c104482 := (publicBatchWrapper86_copies3265 a h).2.2.1
  have c104483 := (publicBatchWrapper86_copies3265 a h).2.2.2.1
  have c104484 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.1
  have c104485 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.1
  have c104486 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_12 := arithEq_of_rows h (row := 1709) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_12
  simp only [← c104481, ← c104482, ← c104483] at e_1709_12
  have e_1709_13 := arithEq_of_rows h (row := 1709) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_13
  simp only [← c104484, ← c104485, k2, ← c104486] at e_1709_13
  have hr := e_1709_13
  simp only [e_1709_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16689 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 173)) := by
  have c104487 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.1
  have c104488 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.1
  have c104489 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.1
  have c104490 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104491 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104492 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_14 := arithEq_of_rows h (row := 1709) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_14
  simp only [← c104487, ← c104488, ← c104489] at e_1709_14
  have e_1709_15 := arithEq_of_rows h (row := 1709) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_15
  simp only [← c104490, ← c104491, k2, ← c104492] at e_1709_15
  have hr := e_1709_15
  simp only [e_1709_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16690 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 174)) := by
  have c104493 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104494 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104495 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104496 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104497 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104498 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_16 := arithEq_of_rows h (row := 1709) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_16
  simp only [← c104493, ← c104494, ← c104495] at e_1709_16
  have e_1709_17 := arithEq_of_rows h (row := 1709) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_17
  simp only [← c104496, ← c104497, k2, ← c104498] at e_1709_17
  have hr := e_1709_17
  simp only [e_1709_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16691 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1709 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 175)) := by
  have c104499 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104500 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104501 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104502 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104503 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104504 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1709_18 := arithEq_of_rows h (row := 1709) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_18
  simp only [← c104499, ← c104500, ← c104501] at e_1709_18
  have e_1709_19 := arithEq_of_rows h (row := 1709) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1709_19
  simp only [← c104502, ← c104503, k2, ← c104504] at e_1709_19
  have hr := e_1709_19
  simp only [e_1709_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16692 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 176)) := by
  have c104505 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104506 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104507 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104508 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104509 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104510 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_0 := arithEq_of_rows h (row := 1710) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_0
  simp only [← c104505, ← c104506, ← c104507] at e_1710_0
  have e_1710_1 := arithEq_of_rows h (row := 1710) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_1
  simp only [← c104508, ← c104509, k2, ← c104510] at e_1710_1
  have hr := e_1710_1
  simp only [e_1710_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16693 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 177)) := by
  have c104511 := (publicBatchWrapper86_copies3265 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104512 := (publicBatchWrapper86_copies3266 a h).1
  have c104513 := (publicBatchWrapper86_copies3266 a h).2.1
  have c104514 := (publicBatchWrapper86_copies3266 a h).2.2.1
  have c104515 := (publicBatchWrapper86_copies3266 a h).2.2.2.1
  have c104516 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_2 := arithEq_of_rows h (row := 1710) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_2
  simp only [← c104511, ← c104512, ← c104513] at e_1710_2
  have e_1710_3 := arithEq_of_rows h (row := 1710) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_3
  simp only [← c104514, ← c104515, k2, ← c104516] at e_1710_3
  have hr := e_1710_3
  simp only [e_1710_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16694 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 178)) := by
  have c104517 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.1
  have c104518 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.1
  have c104519 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.1
  have c104520 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.1
  have c104521 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.1
  have c104522 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_4 := arithEq_of_rows h (row := 1710) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_4
  simp only [← c104517, ← c104518, ← c104519] at e_1710_4
  have e_1710_5 := arithEq_of_rows h (row := 1710) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_5
  simp only [← c104520, ← c104521, k2, ← c104522] at e_1710_5
  have hr := e_1710_5
  simp only [e_1710_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16695 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 179)) := by
  have c104523 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104524 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104525 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104526 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104527 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104528 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_6 := arithEq_of_rows h (row := 1710) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_6
  simp only [← c104523, ← c104524, ← c104525] at e_1710_6
  have e_1710_7 := arithEq_of_rows h (row := 1710) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_7
  simp only [← c104526, ← c104527, k2, ← c104528] at e_1710_7
  have hr := e_1710_7
  simp only [e_1710_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16696 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 180)) := by
  have c104529 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104530 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104531 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104532 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104533 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104534 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_8 := arithEq_of_rows h (row := 1710) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_8
  simp only [← c104529, ← c104530, ← c104531] at e_1710_8
  have e_1710_9 := arithEq_of_rows h (row := 1710) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_9
  simp only [← c104532, ← c104533, k2, ← c104534] at e_1710_9
  have hr := e_1710_9
  simp only [e_1710_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16697 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 181)) := by
  have c104535 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104536 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104537 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104538 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104539 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104540 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_10 := arithEq_of_rows h (row := 1710) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_10
  simp only [← c104535, ← c104536, ← c104537] at e_1710_10
  have e_1710_11 := arithEq_of_rows h (row := 1710) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_11
  simp only [← c104538, ← c104539, k2, ← c104540] at e_1710_11
  have hr := e_1710_11
  simp only [e_1710_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16698 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 182)) := by
  have c104541 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104542 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104543 := (publicBatchWrapper86_copies3266 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104544 := (publicBatchWrapper86_copies3267 a h).1
  have c104545 := (publicBatchWrapper86_copies3267 a h).2.1
  have c104546 := (publicBatchWrapper86_copies3267 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_12 := arithEq_of_rows h (row := 1710) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_12
  simp only [← c104541, ← c104542, ← c104543] at e_1710_12
  have e_1710_13 := arithEq_of_rows h (row := 1710) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_13
  simp only [← c104544, ← c104545, k2, ← c104546] at e_1710_13
  have hr := e_1710_13
  simp only [e_1710_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16699 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 183)) := by
  have c104547 := (publicBatchWrapper86_copies3267 a h).2.2.2.1
  have c104548 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.1
  have c104549 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.1
  have c104550 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.1
  have c104551 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.1
  have c104552 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_14 := arithEq_of_rows h (row := 1710) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_14
  simp only [← c104547, ← c104548, ← c104549] at e_1710_14
  have e_1710_15 := arithEq_of_rows h (row := 1710) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_15
  simp only [← c104550, ← c104551, k2, ← c104552] at e_1710_15
  have hr := e_1710_15
  simp only [e_1710_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16700 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 184)) := by
  have c104553 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.1
  have c104554 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104555 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104556 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104557 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104558 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_16 := arithEq_of_rows h (row := 1710) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_16
  simp only [← c104553, ← c104554, ← c104555] at e_1710_16
  have e_1710_17 := arithEq_of_rows h (row := 1710) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_17
  simp only [← c104556, ← c104557, k2, ← c104558] at e_1710_17
  have hr := e_1710_17
  simp only [e_1710_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16701 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1710 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 185)) := by
  have c104559 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104560 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104561 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104562 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104563 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104564 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1710_18 := arithEq_of_rows h (row := 1710) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_18
  simp only [← c104559, ← c104560, ← c104561] at e_1710_18
  have e_1710_19 := arithEq_of_rows h (row := 1710) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1710_19
  simp only [← c104562, ← c104563, k2, ← c104564] at e_1710_19
  have hr := e_1710_19
  simp only [e_1710_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16702 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 186)) := by
  have c104565 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104566 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104567 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104568 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104569 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104570 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_0 := arithEq_of_rows h (row := 1711) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_0
  simp only [← c104565, ← c104566, ← c104567] at e_1711_0
  have e_1711_1 := arithEq_of_rows h (row := 1711) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_1
  simp only [← c104568, ← c104569, k2, ← c104570] at e_1711_1
  have hr := e_1711_1
  simp only [e_1711_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16703 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 187)) := by
  have c104571 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104572 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104573 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104574 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104575 := (publicBatchWrapper86_copies3267 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104576 := (publicBatchWrapper86_copies3268 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_2 := arithEq_of_rows h (row := 1711) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_2
  simp only [← c104571, ← c104572, ← c104573] at e_1711_2
  have e_1711_3 := arithEq_of_rows h (row := 1711) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_3
  simp only [← c104574, ← c104575, k2, ← c104576] at e_1711_3
  have hr := e_1711_3
  simp only [e_1711_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16704 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 188)) := by
  have c104577 := (publicBatchWrapper86_copies3268 a h).2.1
  have c104578 := (publicBatchWrapper86_copies3268 a h).2.2.1
  have c104579 := (publicBatchWrapper86_copies3268 a h).2.2.2.1
  have c104580 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.1
  have c104581 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.1
  have c104582 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_4 := arithEq_of_rows h (row := 1711) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_4
  simp only [← c104577, ← c104578, ← c104579] at e_1711_4
  have e_1711_5 := arithEq_of_rows h (row := 1711) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_5
  simp only [← c104580, ← c104581, k2, ← c104582] at e_1711_5
  have hr := e_1711_5
  simp only [e_1711_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16705 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 189)) := by
  have c104583 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.1
  have c104584 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.1
  have c104585 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.1
  have c104586 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104587 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104588 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_6 := arithEq_of_rows h (row := 1711) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_6
  simp only [← c104583, ← c104584, ← c104585] at e_1711_6
  have e_1711_7 := arithEq_of_rows h (row := 1711) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_7
  simp only [← c104586, ← c104587, k2, ← c104588] at e_1711_7
  have hr := e_1711_7
  simp only [e_1711_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16706 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 39) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 190)) := by
  have c104589 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104590 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104591 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104592 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104593 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104594 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_8 := arithEq_of_rows h (row := 1711) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_8
  simp only [← c104589, ← c104590, ← c104591] at e_1711_8
  have e_1711_9 := arithEq_of_rows h (row := 1711) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_9
  simp only [← c104592, ← c104593, k2, ← c104594] at e_1711_9
  have hr := e_1711_9
  simp only [e_1711_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16707 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 47) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 191)) := by
  have c104595 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104596 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104597 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104598 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104599 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104600 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_10 := arithEq_of_rows h (row := 1711) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_10
  simp only [← c104595, ← c104596, ← c104597] at e_1711_10
  have e_1711_11 := arithEq_of_rows h (row := 1711) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_11
  simp only [← c104598, ← c104599, k2, ← c104600] at e_1711_11
  have hr := e_1711_11
  simp only [e_1711_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16708 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 55) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 192)) := by
  have c104601 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104602 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104603 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104604 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104605 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104606 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_12 := arithEq_of_rows h (row := 1711) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_12
  simp only [← c104601, ← c104602, ← c104603] at e_1711_12
  have e_1711_13 := arithEq_of_rows h (row := 1711) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_13
  simp only [← c104604, ← c104605, k2, ← c104606] at e_1711_13
  have hr := e_1711_13
  simp only [e_1711_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16709 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 63) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 193)) := by
  have c104607 := (publicBatchWrapper86_copies3268 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104608 := (publicBatchWrapper86_copies3269 a h).1
  have c104609 := (publicBatchWrapper86_copies3269 a h).2.1
  have c104610 := (publicBatchWrapper86_copies3269 a h).2.2.1
  have c104611 := (publicBatchWrapper86_copies3269 a h).2.2.2.1
  have c104612 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_14 := arithEq_of_rows h (row := 1711) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_14
  simp only [← c104607, ← c104608, ← c104609] at e_1711_14
  have e_1711_15 := arithEq_of_rows h (row := 1711) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_15
  simp only [← c104610, ← c104611, k2, ← c104612] at e_1711_15
  have hr := e_1711_15
  simp only [e_1711_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16710 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 71) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 194)) := by
  have c104613 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.1
  have c104614 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.1
  have c104615 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.1
  have c104616 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.1
  have c104617 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.1
  have c104618 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_16 := arithEq_of_rows h (row := 1711) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_16
  simp only [← c104613, ← c104614, ← c104615] at e_1711_16
  have e_1711_17 := arithEq_of_rows h (row := 1711) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_17
  simp only [← c104616, ← c104617, k2, ← c104618] at e_1711_17
  have hr := e_1711_17
  simp only [e_1711_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16711 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1711 79) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 195)) := by
  have c104619 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104620 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104621 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104622 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104623 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104624 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1711_18 := arithEq_of_rows h (row := 1711) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_18
  simp only [← c104619, ← c104620, ← c104621] at e_1711_18
  have e_1711_19 := arithEq_of_rows h (row := 1711) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1711_19
  simp only [← c104622, ← c104623, k2, ← c104624] at e_1711_19
  have hr := e_1711_19
  simp only [e_1711_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16712 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1712 7) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 196)) := by
  have c104625 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104626 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104627 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104628 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104629 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104630 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1712_0 := arithEq_of_rows h (row := 1712) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_0
  simp only [← c104625, ← c104626, ← c104627] at e_1712_0
  have e_1712_1 := arithEq_of_rows h (row := 1712) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_1
  simp only [← c104628, ← c104629, k2, ← c104630] at e_1712_1
  have hr := e_1712_1
  simp only [e_1712_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16713 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1712 15) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 197)) := by
  have c104631 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104632 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104633 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104634 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104635 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104636 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1712_2 := arithEq_of_rows h (row := 1712) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_2
  simp only [← c104631, ← c104632, ← c104633] at e_1712_2
  have e_1712_3 := arithEq_of_rows h (row := 1712) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_3
  simp only [← c104634, ← c104635, k2, ← c104636] at e_1712_3
  have hr := e_1712_3
  simp only [e_1712_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16714 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1712 23) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 198)) := by
  have c104637 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104638 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104639 := (publicBatchWrapper86_copies3269 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104640 := (publicBatchWrapper86_copies3270 a h).1
  have c104641 := (publicBatchWrapper86_copies3270 a h).2.1
  have c104642 := (publicBatchWrapper86_copies3270 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1712_4 := arithEq_of_rows h (row := 1712) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_4
  simp only [← c104637, ← c104638, ← c104639] at e_1712_4
  have e_1712_5 := arithEq_of_rows h (row := 1712) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_5
  simp only [← c104640, ← c104641, k2, ← c104642] at e_1712_5
  have hr := e_1712_5
  simp only [e_1712_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16715 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1712 31) = bselect (a (.wire 3 27)) (a (.virt 200)) (a (.virt 199)) := by
  have c104643 := (publicBatchWrapper86_copies3270 a h).2.2.2.1
  have c104644 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.1
  have c104645 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.1
  have c104646 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.1
  have c104647 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.1
  have c104648 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1712_6 := arithEq_of_rows h (row := 1712) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_6
  simp only [← c104643, ← c104644, ← c104645] at e_1712_6
  have e_1712_7 := arithEq_of_rows h (row := 1712) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_7
  simp only [← c104646, ← c104647, k2, ← c104648] at e_1712_7
  have hr := e_1712_7
  simp only [e_1712_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16716 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1712 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 370)) := by
  have c104649 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.1
  have c104650 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104651 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104652 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104653 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104654 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1712_8 := arithEq_of_rows h (row := 1712) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_8
  simp only [← c104649, ← c104650, ← c104651] at e_1712_8
  have e_1712_9 := arithEq_of_rows h (row := 1712) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_9
  simp only [← c104652, ← c104653, k2, ← c104654] at e_1712_9
  have hr := e_1712_9
  simp only [e_1712_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16717 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1712 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 371)) := by
  have c104655 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104656 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104657 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104658 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104659 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104660 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1712_10 := arithEq_of_rows h (row := 1712) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_10
  simp only [← c104655, ← c104656, ← c104657] at e_1712_10
  have e_1712_11 := arithEq_of_rows h (row := 1712) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_11
  simp only [← c104658, ← c104659, k2, ← c104660] at e_1712_11
  have hr := e_1712_11
  simp only [e_1712_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16718 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1712 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 372)) := by
  have c104661 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104662 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104663 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104664 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104665 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104666 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1712_12 := arithEq_of_rows h (row := 1712) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_12
  simp only [← c104661, ← c104662, ← c104663] at e_1712_12
  have e_1712_13 := arithEq_of_rows h (row := 1712) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_13
  simp only [← c104664, ← c104665, k2, ← c104666] at e_1712_13
  have hr := e_1712_13
  simp only [e_1712_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16719 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1712 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 373)) := by
  have c104667 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104668 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104669 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104670 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104671 := (publicBatchWrapper86_copies3270 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104672 := (publicBatchWrapper86_copies3271 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1712_14 := arithEq_of_rows h (row := 1712) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_14
  simp only [← c104667, ← c104668, ← c104669] at e_1712_14
  have e_1712_15 := arithEq_of_rows h (row := 1712) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_15
  simp only [← c104670, ← c104671, k2, ← c104672] at e_1712_15
  have hr := e_1712_15
  simp only [e_1712_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16720 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1712 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 374)) := by
  have c104673 := (publicBatchWrapper86_copies3271 a h).2.1
  have c104674 := (publicBatchWrapper86_copies3271 a h).2.2.1
  have c104675 := (publicBatchWrapper86_copies3271 a h).2.2.2.1
  have c104676 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.1
  have c104677 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.1
  have c104678 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1712_16 := arithEq_of_rows h (row := 1712) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_16
  simp only [← c104673, ← c104674, ← c104675] at e_1712_16
  have e_1712_17 := arithEq_of_rows h (row := 1712) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_17
  simp only [← c104676, ← c104677, k2, ← c104678] at e_1712_17
  have hr := e_1712_17
  simp only [e_1712_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16721 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1712 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 375)) := by
  have c104679 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.1
  have c104680 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.1
  have c104681 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.1
  have c104682 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104683 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104684 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1712_18 := arithEq_of_rows h (row := 1712) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_18
  simp only [← c104679, ← c104680, ← c104681] at e_1712_18
  have e_1712_19 := arithEq_of_rows h (row := 1712) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1712_19
  simp only [← c104682, ← c104683, k2, ← c104684] at e_1712_19
  have hr := e_1712_19
  simp only [e_1712_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16722 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1713 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 376)) := by
  have c104685 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104686 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104687 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104688 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104689 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104690 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1713_0 := arithEq_of_rows h (row := 1713) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_0
  simp only [← c104685, ← c104686, ← c104687] at e_1713_0
  have e_1713_1 := arithEq_of_rows h (row := 1713) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_1
  simp only [← c104688, ← c104689, k2, ← c104690] at e_1713_1
  have hr := e_1713_1
  simp only [e_1713_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16723 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1713 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 377)) := by
  have c104691 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104692 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104693 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104694 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104695 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104696 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1713_2 := arithEq_of_rows h (row := 1713) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_2
  simp only [← c104691, ← c104692, ← c104693] at e_1713_2
  have e_1713_3 := arithEq_of_rows h (row := 1713) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_3
  simp only [← c104694, ← c104695, k2, ← c104696] at e_1713_3
  have hr := e_1713_3
  simp only [e_1713_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16724 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1713 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 378)) := by
  have c104697 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104698 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104699 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104700 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104701 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104702 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1713_4 := arithEq_of_rows h (row := 1713) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_4
  simp only [← c104697, ← c104698, ← c104699] at e_1713_4
  have e_1713_5 := arithEq_of_rows h (row := 1713) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_5
  simp only [← c104700, ← c104701, k2, ← c104702] at e_1713_5
  have hr := e_1713_5
  simp only [e_1713_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16725 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1713 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 379)) := by
  have c104703 := (publicBatchWrapper86_copies3271 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104704 := (publicBatchWrapper86_copies3272 a h).1
  have c104705 := (publicBatchWrapper86_copies3272 a h).2.1
  have c104706 := (publicBatchWrapper86_copies3272 a h).2.2.1
  have c104707 := (publicBatchWrapper86_copies3272 a h).2.2.2.1
  have c104708 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1713_6 := arithEq_of_rows h (row := 1713) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_6
  simp only [← c104703, ← c104704, ← c104705] at e_1713_6
  have e_1713_7 := arithEq_of_rows h (row := 1713) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_7
  simp only [← c104706, ← c104707, k2, ← c104708] at e_1713_7
  have hr := e_1713_7
  simp only [e_1713_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16726 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1713 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 380)) := by
  have c104709 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.1
  have c104710 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.1
  have c104711 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.1
  have c104712 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.1
  have c104713 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.1
  have c104714 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1713_8 := arithEq_of_rows h (row := 1713) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_8
  simp only [← c104709, ← c104710, ← c104711] at e_1713_8
  have e_1713_9 := arithEq_of_rows h (row := 1713) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_9
  simp only [← c104712, ← c104713, k2, ← c104714] at e_1713_9
  have hr := e_1713_9
  simp only [e_1713_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16727 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1713 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 381)) := by
  have c104715 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104716 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104717 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104718 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104719 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104720 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1713_10 := arithEq_of_rows h (row := 1713) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_10
  simp only [← c104715, ← c104716, ← c104717] at e_1713_10
  have e_1713_11 := arithEq_of_rows h (row := 1713) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_11
  simp only [← c104718, ← c104719, k2, ← c104720] at e_1713_11
  have hr := e_1713_11
  simp only [e_1713_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16728 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1713 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 382)) := by
  have c104721 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104722 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104723 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104724 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104725 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104726 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1713_12 := arithEq_of_rows h (row := 1713) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_12
  simp only [← c104721, ← c104722, ← c104723] at e_1713_12
  have e_1713_13 := arithEq_of_rows h (row := 1713) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_13
  simp only [← c104724, ← c104725, k2, ← c104726] at e_1713_13
  have hr := e_1713_13
  simp only [e_1713_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16729 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1713 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 383)) := by
  have c104727 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104728 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104729 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104730 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104731 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104732 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1713_14 := arithEq_of_rows h (row := 1713) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_14
  simp only [← c104727, ← c104728, ← c104729] at e_1713_14
  have e_1713_15 := arithEq_of_rows h (row := 1713) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_15
  simp only [← c104730, ← c104731, k2, ← c104732] at e_1713_15
  have hr := e_1713_15
  simp only [e_1713_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16730 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1713 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 384)) := by
  have c104733 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104734 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104735 := (publicBatchWrapper86_copies3272 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104736 := (publicBatchWrapper86_copies3273 a h).1
  have c104737 := (publicBatchWrapper86_copies3273 a h).2.1
  have c104738 := (publicBatchWrapper86_copies3273 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1713_16 := arithEq_of_rows h (row := 1713) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_16
  simp only [← c104733, ← c104734, ← c104735] at e_1713_16
  have e_1713_17 := arithEq_of_rows h (row := 1713) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_17
  simp only [← c104736, ← c104737, k2, ← c104738] at e_1713_17
  have hr := e_1713_17
  simp only [e_1713_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16731 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1713 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 385)) := by
  have c104739 := (publicBatchWrapper86_copies3273 a h).2.2.2.1
  have c104740 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.1
  have c104741 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.1
  have c104742 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.1
  have c104743 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.1
  have c104744 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1713_18 := arithEq_of_rows h (row := 1713) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_18
  simp only [← c104739, ← c104740, ← c104741] at e_1713_18
  have e_1713_19 := arithEq_of_rows h (row := 1713) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1713_19
  simp only [← c104742, ← c104743, k2, ← c104744] at e_1713_19
  have hr := e_1713_19
  simp only [e_1713_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16732 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1714 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 386)) := by
  have c104745 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.1
  have c104746 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104747 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104748 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104749 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104750 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1714_0 := arithEq_of_rows h (row := 1714) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_0
  simp only [← c104745, ← c104746, ← c104747] at e_1714_0
  have e_1714_1 := arithEq_of_rows h (row := 1714) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_1
  simp only [← c104748, ← c104749, k2, ← c104750] at e_1714_1
  have hr := e_1714_1
  simp only [e_1714_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16733 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1714 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 387)) := by
  have c104751 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104752 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104753 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104754 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104755 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104756 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1714_2 := arithEq_of_rows h (row := 1714) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_2
  simp only [← c104751, ← c104752, ← c104753] at e_1714_2
  have e_1714_3 := arithEq_of_rows h (row := 1714) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_3
  simp only [← c104754, ← c104755, k2, ← c104756] at e_1714_3
  have hr := e_1714_3
  simp only [e_1714_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16734 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1714 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 388)) := by
  have c104757 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104758 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104759 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104760 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104761 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104762 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1714_4 := arithEq_of_rows h (row := 1714) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_4
  simp only [← c104757, ← c104758, ← c104759] at e_1714_4
  have e_1714_5 := arithEq_of_rows h (row := 1714) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_5
  simp only [← c104760, ← c104761, k2, ← c104762] at e_1714_5
  have hr := e_1714_5
  simp only [e_1714_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16735 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1714 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 389)) := by
  have c104763 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104764 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104765 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104766 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104767 := (publicBatchWrapper86_copies3273 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104768 := (publicBatchWrapper86_copies3274 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1714_6 := arithEq_of_rows h (row := 1714) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_6
  simp only [← c104763, ← c104764, ← c104765] at e_1714_6
  have e_1714_7 := arithEq_of_rows h (row := 1714) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_7
  simp only [← c104766, ← c104767, k2, ← c104768] at e_1714_7
  have hr := e_1714_7
  simp only [e_1714_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16736 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1714 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 390)) := by
  have c104769 := (publicBatchWrapper86_copies3274 a h).2.1
  have c104770 := (publicBatchWrapper86_copies3274 a h).2.2.1
  have c104771 := (publicBatchWrapper86_copies3274 a h).2.2.2.1
  have c104772 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.1
  have c104773 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.1
  have c104774 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1714_8 := arithEq_of_rows h (row := 1714) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_8
  simp only [← c104769, ← c104770, ← c104771] at e_1714_8
  have e_1714_9 := arithEq_of_rows h (row := 1714) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_9
  simp only [← c104772, ← c104773, k2, ← c104774] at e_1714_9
  have hr := e_1714_9
  simp only [e_1714_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16737 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1714 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 391)) := by
  have c104775 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.1
  have c104776 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.1
  have c104777 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.1
  have c104778 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104779 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104780 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1714_10 := arithEq_of_rows h (row := 1714) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_10
  simp only [← c104775, ← c104776, ← c104777] at e_1714_10
  have e_1714_11 := arithEq_of_rows h (row := 1714) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_11
  simp only [← c104778, ← c104779, k2, ← c104780] at e_1714_11
  have hr := e_1714_11
  simp only [e_1714_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16738 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1714 55) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 392)) := by
  have c104781 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104782 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104783 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104784 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104785 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104786 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1714_12 := arithEq_of_rows h (row := 1714) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_12
  simp only [← c104781, ← c104782, ← c104783] at e_1714_12
  have e_1714_13 := arithEq_of_rows h (row := 1714) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_13
  simp only [← c104784, ← c104785, k2, ← c104786] at e_1714_13
  have hr := e_1714_13
  simp only [e_1714_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16739 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1714 63) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 393)) := by
  have c104787 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104788 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104789 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104790 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104791 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104792 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1714_14 := arithEq_of_rows h (row := 1714) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_14
  simp only [← c104787, ← c104788, ← c104789] at e_1714_14
  have e_1714_15 := arithEq_of_rows h (row := 1714) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_15
  simp only [← c104790, ← c104791, k2, ← c104792] at e_1714_15
  have hr := e_1714_15
  simp only [e_1714_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16740 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1714 71) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 394)) := by
  have c104793 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104794 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104795 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104796 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104797 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104798 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1714_16 := arithEq_of_rows h (row := 1714) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_16
  simp only [← c104793, ← c104794, ← c104795] at e_1714_16
  have e_1714_17 := arithEq_of_rows h (row := 1714) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_17
  simp only [← c104796, ← c104797, k2, ← c104798] at e_1714_17
  have hr := e_1714_17
  simp only [e_1714_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16741 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1714 79) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 395)) := by
  have c104799 := (publicBatchWrapper86_copies3274 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104800 := (publicBatchWrapper86_copies3275 a h).1
  have c104801 := (publicBatchWrapper86_copies3275 a h).2.1
  have c104802 := (publicBatchWrapper86_copies3275 a h).2.2.1
  have c104803 := (publicBatchWrapper86_copies3275 a h).2.2.2.1
  have c104804 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1714_18 := arithEq_of_rows h (row := 1714) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_18
  simp only [← c104799, ← c104800, ← c104801] at e_1714_18
  have e_1714_19 := arithEq_of_rows h (row := 1714) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1714_19
  simp only [← c104802, ← c104803, k2, ← c104804] at e_1714_19
  have hr := e_1714_19
  simp only [e_1714_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16742 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1715 7) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 396)) := by
  have c104805 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.1
  have c104806 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.1
  have c104807 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.1
  have c104808 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.1
  have c104809 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.1
  have c104810 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1715_0 := arithEq_of_rows h (row := 1715) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_0
  simp only [← c104805, ← c104806, ← c104807] at e_1715_0
  have e_1715_1 := arithEq_of_rows h (row := 1715) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_1
  simp only [← c104808, ← c104809, k2, ← c104810] at e_1715_1
  have hr := e_1715_1
  simp only [e_1715_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16743 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1715 15) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 397)) := by
  have c104811 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104812 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104813 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104814 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104815 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104816 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1715_2 := arithEq_of_rows h (row := 1715) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_2
  simp only [← c104811, ← c104812, ← c104813] at e_1715_2
  have e_1715_3 := arithEq_of_rows h (row := 1715) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_3
  simp only [← c104814, ← c104815, k2, ← c104816] at e_1715_3
  have hr := e_1715_3
  simp only [e_1715_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16744 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1715 23) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 398)) := by
  have c104817 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104818 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104819 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104820 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104821 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104822 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1715_4 := arithEq_of_rows h (row := 1715) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_4
  simp only [← c104817, ← c104818, ← c104819] at e_1715_4
  have e_1715_5 := arithEq_of_rows h (row := 1715) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_5
  simp only [← c104820, ← c104821, k2, ← c104822] at e_1715_5
  have hr := e_1715_5
  simp only [e_1715_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16745 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1715 31) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 399)) := by
  have c104823 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104824 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104825 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104826 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104827 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104828 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1715_6 := arithEq_of_rows h (row := 1715) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_6
  simp only [← c104823, ← c104824, ← c104825] at e_1715_6
  have e_1715_7 := arithEq_of_rows h (row := 1715) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_7
  simp only [← c104826, ← c104827, k2, ← c104828] at e_1715_7
  have hr := e_1715_7
  simp only [e_1715_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16746 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1715 39) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 400)) := by
  have c104829 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104830 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104831 := (publicBatchWrapper86_copies3275 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104832 := (publicBatchWrapper86_copies3276 a h).1
  have c104833 := (publicBatchWrapper86_copies3276 a h).2.1
  have c104834 := (publicBatchWrapper86_copies3276 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1715_8 := arithEq_of_rows h (row := 1715) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_8
  simp only [← c104829, ← c104830, ← c104831] at e_1715_8
  have e_1715_9 := arithEq_of_rows h (row := 1715) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_9
  simp only [← c104832, ← c104833, k2, ← c104834] at e_1715_9
  have hr := e_1715_9
  simp only [e_1715_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16747 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1715 47) = bselect (a (.wire 3 71)) (a (.virt 200)) (a (.virt 401)) := by
  have c104835 := (publicBatchWrapper86_copies3276 a h).2.2.2.1
  have c104836 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.1
  have c104837 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.1
  have c104838 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.1
  have c104839 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.1
  have c104840 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1715_10 := arithEq_of_rows h (row := 1715) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_10
  simp only [← c104835, ← c104836, ← c104837] at e_1715_10
  have e_1715_11 := arithEq_of_rows h (row := 1715) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_11
  simp only [← c104838, ← c104839, k2, ← c104840] at e_1715_11
  have hr := e_1715_11
  simp only [e_1715_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16748 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1715 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 570)) := by
  have c104841 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.1
  have c104842 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104843 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104844 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104845 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104846 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1715_12 := arithEq_of_rows h (row := 1715) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_12
  simp only [← c104841, ← c104842, ← c104843] at e_1715_12
  have e_1715_13 := arithEq_of_rows h (row := 1715) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_13
  simp only [← c104844, ← c104845, k2, ← c104846] at e_1715_13
  have hr := e_1715_13
  simp only [e_1715_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16749 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1715 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 571)) := by
  have c104847 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104848 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104849 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104850 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104851 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104852 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1715_14 := arithEq_of_rows h (row := 1715) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_14
  simp only [← c104847, ← c104848, ← c104849] at e_1715_14
  have e_1715_15 := arithEq_of_rows h (row := 1715) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_15
  simp only [← c104850, ← c104851, k2, ← c104852] at e_1715_15
  have hr := e_1715_15
  simp only [e_1715_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16750 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1715 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 572)) := by
  have c104853 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104854 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104855 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104856 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104857 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104858 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1715_16 := arithEq_of_rows h (row := 1715) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_16
  simp only [← c104853, ← c104854, ← c104855] at e_1715_16
  have e_1715_17 := arithEq_of_rows h (row := 1715) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_17
  simp only [← c104856, ← c104857, k2, ← c104858] at e_1715_17
  have hr := e_1715_17
  simp only [e_1715_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16751 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1715 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 573)) := by
  have c104859 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104860 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104861 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104862 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104863 := (publicBatchWrapper86_copies3276 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104864 := (publicBatchWrapper86_copies3277 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1715_18 := arithEq_of_rows h (row := 1715) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_18
  simp only [← c104859, ← c104860, ← c104861] at e_1715_18
  have e_1715_19 := arithEq_of_rows h (row := 1715) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1715_19
  simp only [← c104862, ← c104863, k2, ← c104864] at e_1715_19
  have hr := e_1715_19
  simp only [e_1715_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16752 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1716 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 574)) := by
  have c104865 := (publicBatchWrapper86_copies3277 a h).2.1
  have c104866 := (publicBatchWrapper86_copies3277 a h).2.2.1
  have c104867 := (publicBatchWrapper86_copies3277 a h).2.2.2.1
  have c104868 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.1
  have c104869 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.1
  have c104870 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1716_0 := arithEq_of_rows h (row := 1716) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_0
  simp only [← c104865, ← c104866, ← c104867] at e_1716_0
  have e_1716_1 := arithEq_of_rows h (row := 1716) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_1
  simp only [← c104868, ← c104869, k2, ← c104870] at e_1716_1
  have hr := e_1716_1
  simp only [e_1716_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16753 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1716 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 575)) := by
  have c104871 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.1
  have c104872 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.1
  have c104873 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.1
  have c104874 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104875 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104876 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1716_2 := arithEq_of_rows h (row := 1716) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_2
  simp only [← c104871, ← c104872, ← c104873] at e_1716_2
  have e_1716_3 := arithEq_of_rows h (row := 1716) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_3
  simp only [← c104874, ← c104875, k2, ← c104876] at e_1716_3
  have hr := e_1716_3
  simp only [e_1716_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16754 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1716 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 576)) := by
  have c104877 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104878 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104879 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104880 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104881 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104882 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1716_4 := arithEq_of_rows h (row := 1716) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_4
  simp only [← c104877, ← c104878, ← c104879] at e_1716_4
  have e_1716_5 := arithEq_of_rows h (row := 1716) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_5
  simp only [← c104880, ← c104881, k2, ← c104882] at e_1716_5
  have hr := e_1716_5
  simp only [e_1716_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16755 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1716 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 577)) := by
  have c104883 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104884 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104885 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104886 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104887 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104888 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1716_6 := arithEq_of_rows h (row := 1716) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_6
  simp only [← c104883, ← c104884, ← c104885] at e_1716_6
  have e_1716_7 := arithEq_of_rows h (row := 1716) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_7
  simp only [← c104886, ← c104887, k2, ← c104888] at e_1716_7
  have hr := e_1716_7
  simp only [e_1716_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16756 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1716 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 578)) := by
  have c104889 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104890 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104891 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104892 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104893 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104894 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1716_8 := arithEq_of_rows h (row := 1716) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_8
  simp only [← c104889, ← c104890, ← c104891] at e_1716_8
  have e_1716_9 := arithEq_of_rows h (row := 1716) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_9
  simp only [← c104892, ← c104893, k2, ← c104894] at e_1716_9
  have hr := e_1716_9
  simp only [e_1716_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16757 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1716 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 579)) := by
  have c104895 := (publicBatchWrapper86_copies3277 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104896 := (publicBatchWrapper86_copies3278 a h).1
  have c104897 := (publicBatchWrapper86_copies3278 a h).2.1
  have c104898 := (publicBatchWrapper86_copies3278 a h).2.2.1
  have c104899 := (publicBatchWrapper86_copies3278 a h).2.2.2.1
  have c104900 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1716_10 := arithEq_of_rows h (row := 1716) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_10
  simp only [← c104895, ← c104896, ← c104897] at e_1716_10
  have e_1716_11 := arithEq_of_rows h (row := 1716) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_11
  simp only [← c104898, ← c104899, k2, ← c104900] at e_1716_11
  have hr := e_1716_11
  simp only [e_1716_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16758 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1716 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 580)) := by
  have c104901 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.1
  have c104902 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.1
  have c104903 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.1
  have c104904 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.1
  have c104905 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.1
  have c104906 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1716_12 := arithEq_of_rows h (row := 1716) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_12
  simp only [← c104901, ← c104902, ← c104903] at e_1716_12
  have e_1716_13 := arithEq_of_rows h (row := 1716) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_13
  simp only [← c104904, ← c104905, k2, ← c104906] at e_1716_13
  have hr := e_1716_13
  simp only [e_1716_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16759 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1716 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 581)) := by
  have c104907 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104908 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104909 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104910 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104911 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104912 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1716_14 := arithEq_of_rows h (row := 1716) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_14
  simp only [← c104907, ← c104908, ← c104909] at e_1716_14
  have e_1716_15 := arithEq_of_rows h (row := 1716) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_15
  simp only [← c104910, ← c104911, k2, ← c104912] at e_1716_15
  have hr := e_1716_15
  simp only [e_1716_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16760 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1716 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 582)) := by
  have c104913 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104914 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104915 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104916 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104917 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104918 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1716_16 := arithEq_of_rows h (row := 1716) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_16
  simp only [← c104913, ← c104914, ← c104915] at e_1716_16
  have e_1716_17 := arithEq_of_rows h (row := 1716) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_17
  simp only [← c104916, ← c104917, k2, ← c104918] at e_1716_17
  have hr := e_1716_17
  simp only [e_1716_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16761 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1716 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 583)) := by
  have c104919 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104920 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104921 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104922 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104923 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104924 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1716_18 := arithEq_of_rows h (row := 1716) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_18
  simp only [← c104919, ← c104920, ← c104921] at e_1716_18
  have e_1716_19 := arithEq_of_rows h (row := 1716) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1716_19
  simp only [← c104922, ← c104923, k2, ← c104924] at e_1716_19
  have hr := e_1716_19
  simp only [e_1716_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16762 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1717 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 584)) := by
  have c104925 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104926 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104927 := (publicBatchWrapper86_copies3278 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104928 := (publicBatchWrapper86_copies3279 a h).1
  have c104929 := (publicBatchWrapper86_copies3279 a h).2.1
  have c104930 := (publicBatchWrapper86_copies3279 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1717_0 := arithEq_of_rows h (row := 1717) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_0
  simp only [← c104925, ← c104926, ← c104927] at e_1717_0
  have e_1717_1 := arithEq_of_rows h (row := 1717) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_1
  simp only [← c104928, ← c104929, k2, ← c104930] at e_1717_1
  have hr := e_1717_1
  simp only [e_1717_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16763 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1717 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 585)) := by
  have c104931 := (publicBatchWrapper86_copies3279 a h).2.2.2.1
  have c104932 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.1
  have c104933 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.1
  have c104934 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.1
  have c104935 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.1
  have c104936 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1717_2 := arithEq_of_rows h (row := 1717) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_2
  simp only [← c104931, ← c104932, ← c104933] at e_1717_2
  have e_1717_3 := arithEq_of_rows h (row := 1717) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_3
  simp only [← c104934, ← c104935, k2, ← c104936] at e_1717_3
  have hr := e_1717_3
  simp only [e_1717_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16764 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1717 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 586)) := by
  have c104937 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.1
  have c104938 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104939 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104940 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104941 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104942 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1717_4 := arithEq_of_rows h (row := 1717) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_4
  simp only [← c104937, ← c104938, ← c104939] at e_1717_4
  have e_1717_5 := arithEq_of_rows h (row := 1717) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_5
  simp only [← c104940, ← c104941, k2, ← c104942] at e_1717_5
  have hr := e_1717_5
  simp only [e_1717_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16765 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1717 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 587)) := by
  have c104943 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104944 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104945 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104946 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104947 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104948 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1717_6 := arithEq_of_rows h (row := 1717) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_6
  simp only [← c104943, ← c104944, ← c104945] at e_1717_6
  have e_1717_7 := arithEq_of_rows h (row := 1717) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_7
  simp only [← c104946, ← c104947, k2, ← c104948] at e_1717_7
  have hr := e_1717_7
  simp only [e_1717_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16766 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1717 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 588)) := by
  have c104949 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104950 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104951 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104952 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104953 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104954 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1717_8 := arithEq_of_rows h (row := 1717) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_8
  simp only [← c104949, ← c104950, ← c104951] at e_1717_8
  have e_1717_9 := arithEq_of_rows h (row := 1717) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_9
  simp only [← c104952, ← c104953, k2, ← c104954] at e_1717_9
  have hr := e_1717_9
  simp only [e_1717_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16767 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1717 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 589)) := by
  have c104955 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104956 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104957 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104958 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104959 := (publicBatchWrapper86_copies3279 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104960 := (publicBatchWrapper86_copies3280 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1717_10 := arithEq_of_rows h (row := 1717) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_10
  simp only [← c104955, ← c104956, ← c104957] at e_1717_10
  have e_1717_11 := arithEq_of_rows h (row := 1717) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_11
  simp only [← c104958, ← c104959, k2, ← c104960] at e_1717_11
  have hr := e_1717_11
  simp only [e_1717_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16768 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1717 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 590)) := by
  have c104961 := (publicBatchWrapper86_copies3280 a h).2.1
  have c104962 := (publicBatchWrapper86_copies3280 a h).2.2.1
  have c104963 := (publicBatchWrapper86_copies3280 a h).2.2.2.1
  have c104964 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.1
  have c104965 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.1
  have c104966 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1717_12 := arithEq_of_rows h (row := 1717) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_12
  simp only [← c104961, ← c104962, ← c104963] at e_1717_12
  have e_1717_13 := arithEq_of_rows h (row := 1717) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_13
  simp only [← c104964, ← c104965, k2, ← c104966] at e_1717_13
  have hr := e_1717_13
  simp only [e_1717_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16769 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1717 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 591)) := by
  have c104967 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.1
  have c104968 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.1
  have c104969 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.1
  have c104970 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.1
  have c104971 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c104972 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1717_14 := arithEq_of_rows h (row := 1717) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_14
  simp only [← c104967, ← c104968, ← c104969] at e_1717_14
  have e_1717_15 := arithEq_of_rows h (row := 1717) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_15
  simp only [← c104970, ← c104971, k2, ← c104972] at e_1717_15
  have hr := e_1717_15
  simp only [e_1717_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16770 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1717 71) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 592)) := by
  have c104973 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104974 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104975 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104976 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104977 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104978 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1717_16 := arithEq_of_rows h (row := 1717) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_16
  simp only [← c104973, ← c104974, ← c104975] at e_1717_16
  have e_1717_17 := arithEq_of_rows h (row := 1717) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_17
  simp only [← c104976, ← c104977, k2, ← c104978] at e_1717_17
  have hr := e_1717_17
  simp only [e_1717_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16771 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1717 79) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 593)) := by
  have c104979 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104980 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104981 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104982 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104983 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104984 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1717_18 := arithEq_of_rows h (row := 1717) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_18
  simp only [← c104979, ← c104980, ← c104981] at e_1717_18
  have e_1717_19 := arithEq_of_rows h (row := 1717) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1717_19
  simp only [← c104982, ← c104983, k2, ← c104984] at e_1717_19
  have hr := e_1717_19
  simp only [e_1717_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16772 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1718 7) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 594)) := by
  have c104985 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104986 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104987 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104988 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104989 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c104990 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1718_0 := arithEq_of_rows h (row := 1718) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_0
  simp only [← c104985, ← c104986, ← c104987] at e_1718_0
  have e_1718_1 := arithEq_of_rows h (row := 1718) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_1
  simp only [← c104988, ← c104989, k2, ← c104990] at e_1718_1
  have hr := e_1718_1
  simp only [e_1718_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16773 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1718 15) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 595)) := by
  have c104991 := (publicBatchWrapper86_copies3280 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c104992 := (publicBatchWrapper86_copies3281 a h).1
  have c104993 := (publicBatchWrapper86_copies3281 a h).2.1
  have c104994 := (publicBatchWrapper86_copies3281 a h).2.2.1
  have c104995 := (publicBatchWrapper86_copies3281 a h).2.2.2.1
  have c104996 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1718_2 := arithEq_of_rows h (row := 1718) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_2
  simp only [← c104991, ← c104992, ← c104993] at e_1718_2
  have e_1718_3 := arithEq_of_rows h (row := 1718) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_3
  simp only [← c104994, ← c104995, k2, ← c104996] at e_1718_3
  have hr := e_1718_3
  simp only [e_1718_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16774 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1718 23) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 596)) := by
  have c104997 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.1
  have c104998 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.1
  have c104999 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.1
  have c105000 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.1
  have c105001 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.1
  have c105002 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1718_4 := arithEq_of_rows h (row := 1718) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_4
  simp only [← c104997, ← c104998, ← c104999] at e_1718_4
  have e_1718_5 := arithEq_of_rows h (row := 1718) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_5
  simp only [← c105000, ← c105001, k2, ← c105002] at e_1718_5
  have hr := e_1718_5
  simp only [e_1718_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16775 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1718 31) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 597)) := by
  have c105003 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105004 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105005 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105006 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105007 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105008 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1718_6 := arithEq_of_rows h (row := 1718) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_6
  simp only [← c105003, ← c105004, ← c105005] at e_1718_6
  have e_1718_7 := arithEq_of_rows h (row := 1718) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_7
  simp only [← c105006, ← c105007, k2, ← c105008] at e_1718_7
  have hr := e_1718_7
  simp only [e_1718_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16776 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1718 39) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 598)) := by
  have c105009 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105010 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105011 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105012 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105013 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105014 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1718_8 := arithEq_of_rows h (row := 1718) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_8
  simp only [← c105009, ← c105010, ← c105011] at e_1718_8
  have e_1718_9 := arithEq_of_rows h (row := 1718) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_9
  simp only [← c105012, ← c105013, k2, ← c105014] at e_1718_9
  have hr := e_1718_9
  simp only [e_1718_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16777 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1718 47) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 599)) := by
  have c105015 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105016 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105017 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105018 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105019 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105020 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1718_10 := arithEq_of_rows h (row := 1718) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_10
  simp only [← c105015, ← c105016, ← c105017] at e_1718_10
  have e_1718_11 := arithEq_of_rows h (row := 1718) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_11
  simp only [← c105018, ← c105019, k2, ← c105020] at e_1718_11
  have hr := e_1718_11
  simp only [e_1718_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16778 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1718 55) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 600)) := by
  have c105021 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105022 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105023 := (publicBatchWrapper86_copies3281 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105024 := (publicBatchWrapper86_copies3282 a h).1
  have c105025 := (publicBatchWrapper86_copies3282 a h).2.1
  have c105026 := (publicBatchWrapper86_copies3282 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1718_12 := arithEq_of_rows h (row := 1718) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_12
  simp only [← c105021, ← c105022, ← c105023] at e_1718_12
  have e_1718_13 := arithEq_of_rows h (row := 1718) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_13
  simp only [← c105024, ← c105025, k2, ← c105026] at e_1718_13
  have hr := e_1718_13
  simp only [e_1718_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16779 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1718 63) = bselect (a (.wire 4 35)) (a (.virt 200)) (a (.virt 601)) := by
  have c105027 := (publicBatchWrapper86_copies3282 a h).2.2.2.1
  have c105028 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.1
  have c105029 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.1
  have c105030 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.1
  have c105031 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.1
  have c105032 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1718_14 := arithEq_of_rows h (row := 1718) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_14
  simp only [← c105027, ← c105028, ← c105029] at e_1718_14
  have e_1718_15 := arithEq_of_rows h (row := 1718) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_15
  simp only [← c105030, ← c105031, k2, ← c105032] at e_1718_15
  have hr := e_1718_15
  simp only [e_1718_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16780 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1718 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 770)) := by
  have c105033 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.1
  have c105034 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105035 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105036 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105037 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105038 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1718_16 := arithEq_of_rows h (row := 1718) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_16
  simp only [← c105033, ← c105034, ← c105035] at e_1718_16
  have e_1718_17 := arithEq_of_rows h (row := 1718) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_17
  simp only [← c105036, ← c105037, k2, ← c105038] at e_1718_17
  have hr := e_1718_17
  simp only [e_1718_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16781 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1718 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 771)) := by
  have c105039 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105040 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105041 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105042 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105043 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105044 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1718_18 := arithEq_of_rows h (row := 1718) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_18
  simp only [← c105039, ← c105040, ← c105041] at e_1718_18
  have e_1718_19 := arithEq_of_rows h (row := 1718) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1718_19
  simp only [← c105042, ← c105043, k2, ← c105044] at e_1718_19
  have hr := e_1718_19
  simp only [e_1718_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16782 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1719 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 772)) := by
  have c105045 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105046 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105047 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105048 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105049 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105050 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1719_0 := arithEq_of_rows h (row := 1719) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_0
  simp only [← c105045, ← c105046, ← c105047] at e_1719_0
  have e_1719_1 := arithEq_of_rows h (row := 1719) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_1
  simp only [← c105048, ← c105049, k2, ← c105050] at e_1719_1
  have hr := e_1719_1
  simp only [e_1719_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16783 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1719 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 773)) := by
  have c105051 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105052 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105053 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105054 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105055 := (publicBatchWrapper86_copies3282 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105056 := (publicBatchWrapper86_copies3283 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1719_2 := arithEq_of_rows h (row := 1719) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_2
  simp only [← c105051, ← c105052, ← c105053] at e_1719_2
  have e_1719_3 := arithEq_of_rows h (row := 1719) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_3
  simp only [← c105054, ← c105055, k2, ← c105056] at e_1719_3
  have hr := e_1719_3
  simp only [e_1719_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16784 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1719 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 774)) := by
  have c105057 := (publicBatchWrapper86_copies3283 a h).2.1
  have c105058 := (publicBatchWrapper86_copies3283 a h).2.2.1
  have c105059 := (publicBatchWrapper86_copies3283 a h).2.2.2.1
  have c105060 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.1
  have c105061 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.1
  have c105062 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1719_4 := arithEq_of_rows h (row := 1719) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_4
  simp only [← c105057, ← c105058, ← c105059] at e_1719_4
  have e_1719_5 := arithEq_of_rows h (row := 1719) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_5
  simp only [← c105060, ← c105061, k2, ← c105062] at e_1719_5
  have hr := e_1719_5
  simp only [e_1719_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16785 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1719 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 775)) := by
  have c105063 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.1
  have c105064 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.1
  have c105065 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.1
  have c105066 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105067 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105068 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1719_6 := arithEq_of_rows h (row := 1719) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_6
  simp only [← c105063, ← c105064, ← c105065] at e_1719_6
  have e_1719_7 := arithEq_of_rows h (row := 1719) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_7
  simp only [← c105066, ← c105067, k2, ← c105068] at e_1719_7
  have hr := e_1719_7
  simp only [e_1719_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16786 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1719 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 776)) := by
  have c105069 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105070 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105071 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105072 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105073 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105074 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1719_8 := arithEq_of_rows h (row := 1719) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_8
  simp only [← c105069, ← c105070, ← c105071] at e_1719_8
  have e_1719_9 := arithEq_of_rows h (row := 1719) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_9
  simp only [← c105072, ← c105073, k2, ← c105074] at e_1719_9
  have hr := e_1719_9
  simp only [e_1719_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16787 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1719 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 777)) := by
  have c105075 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105076 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105077 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105078 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105079 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105080 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1719_10 := arithEq_of_rows h (row := 1719) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_10
  simp only [← c105075, ← c105076, ← c105077] at e_1719_10
  have e_1719_11 := arithEq_of_rows h (row := 1719) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_11
  simp only [← c105078, ← c105079, k2, ← c105080] at e_1719_11
  have hr := e_1719_11
  simp only [e_1719_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16788 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1719 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 778)) := by
  have c105081 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105082 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105083 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105084 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105085 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105086 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1719_12 := arithEq_of_rows h (row := 1719) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_12
  simp only [← c105081, ← c105082, ← c105083] at e_1719_12
  have e_1719_13 := arithEq_of_rows h (row := 1719) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_13
  simp only [← c105084, ← c105085, k2, ← c105086] at e_1719_13
  have hr := e_1719_13
  simp only [e_1719_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16789 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1719 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 779)) := by
  have c105087 := (publicBatchWrapper86_copies3283 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105088 := (publicBatchWrapper86_copies3284 a h).1
  have c105089 := (publicBatchWrapper86_copies3284 a h).2.1
  have c105090 := (publicBatchWrapper86_copies3284 a h).2.2.1
  have c105091 := (publicBatchWrapper86_copies3284 a h).2.2.2.1
  have c105092 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1719_14 := arithEq_of_rows h (row := 1719) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_14
  simp only [← c105087, ← c105088, ← c105089] at e_1719_14
  have e_1719_15 := arithEq_of_rows h (row := 1719) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_15
  simp only [← c105090, ← c105091, k2, ← c105092] at e_1719_15
  have hr := e_1719_15
  simp only [e_1719_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16790 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1719 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 780)) := by
  have c105093 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.1
  have c105094 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.1
  have c105095 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.1
  have c105096 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.1
  have c105097 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.1
  have c105098 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1719_16 := arithEq_of_rows h (row := 1719) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_16
  simp only [← c105093, ← c105094, ← c105095] at e_1719_16
  have e_1719_17 := arithEq_of_rows h (row := 1719) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_17
  simp only [← c105096, ← c105097, k2, ← c105098] at e_1719_17
  have hr := e_1719_17
  simp only [e_1719_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16791 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1719 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 781)) := by
  have c105099 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105100 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105101 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105102 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105103 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105104 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1719_18 := arithEq_of_rows h (row := 1719) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_18
  simp only [← c105099, ← c105100, ← c105101] at e_1719_18
  have e_1719_19 := arithEq_of_rows h (row := 1719) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1719_19
  simp only [← c105102, ← c105103, k2, ← c105104] at e_1719_19
  have hr := e_1719_19
  simp only [e_1719_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16792 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1720 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 782)) := by
  have c105105 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105106 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105107 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105108 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105109 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105110 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1720_0 := arithEq_of_rows h (row := 1720) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_0
  simp only [← c105105, ← c105106, ← c105107] at e_1720_0
  have e_1720_1 := arithEq_of_rows h (row := 1720) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_1
  simp only [← c105108, ← c105109, k2, ← c105110] at e_1720_1
  have hr := e_1720_1
  simp only [e_1720_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16793 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1720 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 783)) := by
  have c105111 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105112 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105113 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105114 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105115 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105116 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1720_2 := arithEq_of_rows h (row := 1720) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_2
  simp only [← c105111, ← c105112, ← c105113] at e_1720_2
  have e_1720_3 := arithEq_of_rows h (row := 1720) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_3
  simp only [← c105114, ← c105115, k2, ← c105116] at e_1720_3
  have hr := e_1720_3
  simp only [e_1720_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16794 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1720 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 784)) := by
  have c105117 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105118 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105119 := (publicBatchWrapper86_copies3284 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105120 := (publicBatchWrapper86_copies3285 a h).1
  have c105121 := (publicBatchWrapper86_copies3285 a h).2.1
  have c105122 := (publicBatchWrapper86_copies3285 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1720_4 := arithEq_of_rows h (row := 1720) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_4
  simp only [← c105117, ← c105118, ← c105119] at e_1720_4
  have e_1720_5 := arithEq_of_rows h (row := 1720) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_5
  simp only [← c105120, ← c105121, k2, ← c105122] at e_1720_5
  have hr := e_1720_5
  simp only [e_1720_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16795 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1720 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 785)) := by
  have c105123 := (publicBatchWrapper86_copies3285 a h).2.2.2.1
  have c105124 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.1
  have c105125 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.1
  have c105126 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.1
  have c105127 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.1
  have c105128 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1720_6 := arithEq_of_rows h (row := 1720) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_6
  simp only [← c105123, ← c105124, ← c105125] at e_1720_6
  have e_1720_7 := arithEq_of_rows h (row := 1720) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_7
  simp only [← c105126, ← c105127, k2, ← c105128] at e_1720_7
  have hr := e_1720_7
  simp only [e_1720_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16796 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1720 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 786)) := by
  have c105129 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.1
  have c105130 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105131 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105132 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105133 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105134 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1720_8 := arithEq_of_rows h (row := 1720) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_8
  simp only [← c105129, ← c105130, ← c105131] at e_1720_8
  have e_1720_9 := arithEq_of_rows h (row := 1720) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_9
  simp only [← c105132, ← c105133, k2, ← c105134] at e_1720_9
  have hr := e_1720_9
  simp only [e_1720_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16797 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1720 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 787)) := by
  have c105135 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105136 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105137 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105138 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105139 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105140 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1720_10 := arithEq_of_rows h (row := 1720) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_10
  simp only [← c105135, ← c105136, ← c105137] at e_1720_10
  have e_1720_11 := arithEq_of_rows h (row := 1720) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_11
  simp only [← c105138, ← c105139, k2, ← c105140] at e_1720_11
  have hr := e_1720_11
  simp only [e_1720_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16798 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1720 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 788)) := by
  have c105141 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105142 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105143 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105144 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105145 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105146 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1720_12 := arithEq_of_rows h (row := 1720) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_12
  simp only [← c105141, ← c105142, ← c105143] at e_1720_12
  have e_1720_13 := arithEq_of_rows h (row := 1720) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_13
  simp only [← c105144, ← c105145, k2, ← c105146] at e_1720_13
  have hr := e_1720_13
  simp only [e_1720_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16799 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1720 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 789)) := by
  have c105147 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105148 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105149 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105150 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105151 := (publicBatchWrapper86_copies3285 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105152 := (publicBatchWrapper86_copies3286 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1720_14 := arithEq_of_rows h (row := 1720) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_14
  simp only [← c105147, ← c105148, ← c105149] at e_1720_14
  have e_1720_15 := arithEq_of_rows h (row := 1720) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_15
  simp only [← c105150, ← c105151, k2, ← c105152] at e_1720_15
  have hr := e_1720_15
  simp only [e_1720_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16800 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1720 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 790)) := by
  have c105153 := (publicBatchWrapper86_copies3286 a h).2.1
  have c105154 := (publicBatchWrapper86_copies3286 a h).2.2.1
  have c105155 := (publicBatchWrapper86_copies3286 a h).2.2.2.1
  have c105156 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.1
  have c105157 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.1
  have c105158 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1720_16 := arithEq_of_rows h (row := 1720) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_16
  simp only [← c105153, ← c105154, ← c105155] at e_1720_16
  have e_1720_17 := arithEq_of_rows h (row := 1720) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_17
  simp only [← c105156, ← c105157, k2, ← c105158] at e_1720_17
  have hr := e_1720_17
  simp only [e_1720_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16801 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1720 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 791)) := by
  have c105159 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.1
  have c105160 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.1
  have c105161 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.1
  have c105162 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105163 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105164 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1720_18 := arithEq_of_rows h (row := 1720) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_18
  simp only [← c105159, ← c105160, ← c105161] at e_1720_18
  have e_1720_19 := arithEq_of_rows h (row := 1720) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1720_19
  simp only [← c105162, ← c105163, k2, ← c105164] at e_1720_19
  have hr := e_1720_19
  simp only [e_1720_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16802 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1721 7) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 792)) := by
  have c105165 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105166 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105167 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105168 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105169 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105170 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1721_0 := arithEq_of_rows h (row := 1721) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_0
  simp only [← c105165, ← c105166, ← c105167] at e_1721_0
  have e_1721_1 := arithEq_of_rows h (row := 1721) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_1
  simp only [← c105168, ← c105169, k2, ← c105170] at e_1721_1
  have hr := e_1721_1
  simp only [e_1721_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16803 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1721 15) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 793)) := by
  have c105171 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105172 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105173 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105174 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105175 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105176 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1721_2 := arithEq_of_rows h (row := 1721) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_2
  simp only [← c105171, ← c105172, ← c105173] at e_1721_2
  have e_1721_3 := arithEq_of_rows h (row := 1721) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_3
  simp only [← c105174, ← c105175, k2, ← c105176] at e_1721_3
  have hr := e_1721_3
  simp only [e_1721_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16804 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1721 23) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 794)) := by
  have c105177 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105178 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105179 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105180 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105181 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105182 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1721_4 := arithEq_of_rows h (row := 1721) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_4
  simp only [← c105177, ← c105178, ← c105179] at e_1721_4
  have e_1721_5 := arithEq_of_rows h (row := 1721) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_5
  simp only [← c105180, ← c105181, k2, ← c105182] at e_1721_5
  have hr := e_1721_5
  simp only [e_1721_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16805 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1721 31) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 795)) := by
  have c105183 := (publicBatchWrapper86_copies3286 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105184 := (publicBatchWrapper86_copies3287 a h).1
  have c105185 := (publicBatchWrapper86_copies3287 a h).2.1
  have c105186 := (publicBatchWrapper86_copies3287 a h).2.2.1
  have c105187 := (publicBatchWrapper86_copies3287 a h).2.2.2.1
  have c105188 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1721_6 := arithEq_of_rows h (row := 1721) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_6
  simp only [← c105183, ← c105184, ← c105185] at e_1721_6
  have e_1721_7 := arithEq_of_rows h (row := 1721) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_7
  simp only [← c105186, ← c105187, k2, ← c105188] at e_1721_7
  have hr := e_1721_7
  simp only [e_1721_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16806 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1721 39) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 796)) := by
  have c105189 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.1
  have c105190 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.1
  have c105191 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.1
  have c105192 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.1
  have c105193 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.1
  have c105194 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1721_8 := arithEq_of_rows h (row := 1721) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_8
  simp only [← c105189, ← c105190, ← c105191] at e_1721_8
  have e_1721_9 := arithEq_of_rows h (row := 1721) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_9
  simp only [← c105192, ← c105193, k2, ← c105194] at e_1721_9
  have hr := e_1721_9
  simp only [e_1721_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16807 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1721 47) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 797)) := by
  have c105195 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105196 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105197 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105198 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105199 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105200 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1721_10 := arithEq_of_rows h (row := 1721) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_10
  simp only [← c105195, ← c105196, ← c105197] at e_1721_10
  have e_1721_11 := arithEq_of_rows h (row := 1721) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_11
  simp only [← c105198, ← c105199, k2, ← c105200] at e_1721_11
  have hr := e_1721_11
  simp only [e_1721_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16808 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1721 55) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 798)) := by
  have c105201 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105202 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105203 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105204 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105205 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105206 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1721_12 := arithEq_of_rows h (row := 1721) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_12
  simp only [← c105201, ← c105202, ← c105203] at e_1721_12
  have e_1721_13 := arithEq_of_rows h (row := 1721) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_13
  simp only [← c105204, ← c105205, k2, ← c105206] at e_1721_13
  have hr := e_1721_13
  simp only [e_1721_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16809 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1721 63) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 799)) := by
  have c105207 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105208 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105209 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105210 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105211 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105212 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1721_14 := arithEq_of_rows h (row := 1721) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_14
  simp only [← c105207, ← c105208, ← c105209] at e_1721_14
  have e_1721_15 := arithEq_of_rows h (row := 1721) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_15
  simp only [← c105210, ← c105211, k2, ← c105212] at e_1721_15
  have hr := e_1721_15
  simp only [e_1721_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16810 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1721 71) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 800)) := by
  have c105213 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105214 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105215 := (publicBatchWrapper86_copies3287 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105216 := (publicBatchWrapper86_copies3288 a h).1
  have c105217 := (publicBatchWrapper86_copies3288 a h).2.1
  have c105218 := (publicBatchWrapper86_copies3288 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1721_16 := arithEq_of_rows h (row := 1721) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_16
  simp only [← c105213, ← c105214, ← c105215] at e_1721_16
  have e_1721_17 := arithEq_of_rows h (row := 1721) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_17
  simp only [← c105216, ← c105217, k2, ← c105218] at e_1721_17
  have hr := e_1721_17
  simp only [e_1721_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16811 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1721 79) = bselect (a (.wire 4 79)) (a (.virt 200)) (a (.virt 801)) := by
  have c105219 := (publicBatchWrapper86_copies3288 a h).2.2.2.1
  have c105220 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.1
  have c105221 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.1
  have c105222 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.1
  have c105223 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.1
  have c105224 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1721_18 := arithEq_of_rows h (row := 1721) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_18
  simp only [← c105219, ← c105220, ← c105221] at e_1721_18
  have e_1721_19 := arithEq_of_rows h (row := 1721) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1721_19
  simp only [← c105222, ← c105223, k2, ← c105224] at e_1721_19
  have hr := e_1721_19
  simp only [e_1721_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16812 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1722 7) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 970)) := by
  have c105225 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.1
  have c105226 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105227 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105228 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105229 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105230 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1722_0 := arithEq_of_rows h (row := 1722) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_0
  simp only [← c105225, ← c105226, ← c105227] at e_1722_0
  have e_1722_1 := arithEq_of_rows h (row := 1722) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_1
  simp only [← c105228, ← c105229, k2, ← c105230] at e_1722_1
  have hr := e_1722_1
  simp only [e_1722_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16813 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1722 15) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 971)) := by
  have c105231 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105232 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105233 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105234 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105235 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105236 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1722_2 := arithEq_of_rows h (row := 1722) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_2
  simp only [← c105231, ← c105232, ← c105233] at e_1722_2
  have e_1722_3 := arithEq_of_rows h (row := 1722) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_3
  simp only [← c105234, ← c105235, k2, ← c105236] at e_1722_3
  have hr := e_1722_3
  simp only [e_1722_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16814 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1722 23) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 972)) := by
  have c105237 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105238 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105239 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105240 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105241 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105242 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1722_4 := arithEq_of_rows h (row := 1722) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_4
  simp only [← c105237, ← c105238, ← c105239] at e_1722_4
  have e_1722_5 := arithEq_of_rows h (row := 1722) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_5
  simp only [← c105240, ← c105241, k2, ← c105242] at e_1722_5
  have hr := e_1722_5
  simp only [e_1722_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16815 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1722 31) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 973)) := by
  have c105243 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105244 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105245 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105246 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105247 := (publicBatchWrapper86_copies3288 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105248 := (publicBatchWrapper86_copies3289 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1722_6 := arithEq_of_rows h (row := 1722) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_6
  simp only [← c105243, ← c105244, ← c105245] at e_1722_6
  have e_1722_7 := arithEq_of_rows h (row := 1722) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_7
  simp only [← c105246, ← c105247, k2, ← c105248] at e_1722_7
  have hr := e_1722_7
  simp only [e_1722_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16816 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1722 39) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 974)) := by
  have c105249 := (publicBatchWrapper86_copies3289 a h).2.1
  have c105250 := (publicBatchWrapper86_copies3289 a h).2.2.1
  have c105251 := (publicBatchWrapper86_copies3289 a h).2.2.2.1
  have c105252 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.1
  have c105253 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.1
  have c105254 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1722_8 := arithEq_of_rows h (row := 1722) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_8
  simp only [← c105249, ← c105250, ← c105251] at e_1722_8
  have e_1722_9 := arithEq_of_rows h (row := 1722) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_9
  simp only [← c105252, ← c105253, k2, ← c105254] at e_1722_9
  have hr := e_1722_9
  simp only [e_1722_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16817 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1722 47) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 975)) := by
  have c105255 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.1
  have c105256 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.1
  have c105257 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.1
  have c105258 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105259 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105260 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1722_10 := arithEq_of_rows h (row := 1722) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_10
  simp only [← c105255, ← c105256, ← c105257] at e_1722_10
  have e_1722_11 := arithEq_of_rows h (row := 1722) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_11
  simp only [← c105258, ← c105259, k2, ← c105260] at e_1722_11
  have hr := e_1722_11
  simp only [e_1722_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16818 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1722 55) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 976)) := by
  have c105261 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105262 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105263 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105264 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105265 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105266 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1722_12 := arithEq_of_rows h (row := 1722) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_12
  simp only [← c105261, ← c105262, ← c105263] at e_1722_12
  have e_1722_13 := arithEq_of_rows h (row := 1722) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_13
  simp only [← c105264, ← c105265, k2, ← c105266] at e_1722_13
  have hr := e_1722_13
  simp only [e_1722_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16819 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1722 63) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 977)) := by
  have c105267 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105268 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105269 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105270 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105271 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105272 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1722_14 := arithEq_of_rows h (row := 1722) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_14
  simp only [← c105267, ← c105268, ← c105269] at e_1722_14
  have e_1722_15 := arithEq_of_rows h (row := 1722) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_15
  simp only [← c105270, ← c105271, k2, ← c105272] at e_1722_15
  have hr := e_1722_15
  simp only [e_1722_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16820 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1722 71) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 978)) := by
  have c105273 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105274 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105275 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105276 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105277 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105278 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1722_16 := arithEq_of_rows h (row := 1722) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_16
  simp only [← c105273, ← c105274, ← c105275] at e_1722_16
  have e_1722_17 := arithEq_of_rows h (row := 1722) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_17
  simp only [← c105276, ← c105277, k2, ← c105278] at e_1722_17
  have hr := e_1722_17
  simp only [e_1722_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16821 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1722 79) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 979)) := by
  have c105279 := (publicBatchWrapper86_copies3289 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105280 := (publicBatchWrapper86_copies3290 a h).1
  have c105281 := (publicBatchWrapper86_copies3290 a h).2.1
  have c105282 := (publicBatchWrapper86_copies3290 a h).2.2.1
  have c105283 := (publicBatchWrapper86_copies3290 a h).2.2.2.1
  have c105284 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1722_18 := arithEq_of_rows h (row := 1722) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_18
  simp only [← c105279, ← c105280, ← c105281] at e_1722_18
  have e_1722_19 := arithEq_of_rows h (row := 1722) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1722_19
  simp only [← c105282, ← c105283, k2, ← c105284] at e_1722_19
  have hr := e_1722_19
  simp only [e_1722_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16822 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1723 7) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 980)) := by
  have c105285 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.1
  have c105286 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.1
  have c105287 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.1
  have c105288 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.1
  have c105289 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.1
  have c105290 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1723_0 := arithEq_of_rows h (row := 1723) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_0
  simp only [← c105285, ← c105286, ← c105287] at e_1723_0
  have e_1723_1 := arithEq_of_rows h (row := 1723) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_1
  simp only [← c105288, ← c105289, k2, ← c105290] at e_1723_1
  have hr := e_1723_1
  simp only [e_1723_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16823 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1723 15) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 981)) := by
  have c105291 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105292 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105293 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105294 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105295 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105296 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1723_2 := arithEq_of_rows h (row := 1723) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_2
  simp only [← c105291, ← c105292, ← c105293] at e_1723_2
  have e_1723_3 := arithEq_of_rows h (row := 1723) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_3
  simp only [← c105294, ← c105295, k2, ← c105296] at e_1723_3
  have hr := e_1723_3
  simp only [e_1723_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16824 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1723 23) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 982)) := by
  have c105297 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105298 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105299 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105300 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105301 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105302 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1723_4 := arithEq_of_rows h (row := 1723) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_4
  simp only [← c105297, ← c105298, ← c105299] at e_1723_4
  have e_1723_5 := arithEq_of_rows h (row := 1723) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_5
  simp only [← c105300, ← c105301, k2, ← c105302] at e_1723_5
  have hr := e_1723_5
  simp only [e_1723_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16825 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1723 31) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 983)) := by
  have c105303 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105304 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105305 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105306 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105307 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105308 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1723_6 := arithEq_of_rows h (row := 1723) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_6
  simp only [← c105303, ← c105304, ← c105305] at e_1723_6
  have e_1723_7 := arithEq_of_rows h (row := 1723) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_7
  simp only [← c105306, ← c105307, k2, ← c105308] at e_1723_7
  have hr := e_1723_7
  simp only [e_1723_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16826 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1723 39) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 984)) := by
  have c105309 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105310 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105311 := (publicBatchWrapper86_copies3290 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105312 := (publicBatchWrapper86_copies3291 a h).1
  have c105313 := (publicBatchWrapper86_copies3291 a h).2.1
  have c105314 := (publicBatchWrapper86_copies3291 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1723_8 := arithEq_of_rows h (row := 1723) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_8
  simp only [← c105309, ← c105310, ← c105311] at e_1723_8
  have e_1723_9 := arithEq_of_rows h (row := 1723) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_9
  simp only [← c105312, ← c105313, k2, ← c105314] at e_1723_9
  have hr := e_1723_9
  simp only [e_1723_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16827 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1723 47) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 985)) := by
  have c105315 := (publicBatchWrapper86_copies3291 a h).2.2.2.1
  have c105316 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.1
  have c105317 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.1
  have c105318 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.1
  have c105319 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.1
  have c105320 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1723_10 := arithEq_of_rows h (row := 1723) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_10
  simp only [← c105315, ← c105316, ← c105317] at e_1723_10
  have e_1723_11 := arithEq_of_rows h (row := 1723) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_11
  simp only [← c105318, ← c105319, k2, ← c105320] at e_1723_11
  have hr := e_1723_11
  simp only [e_1723_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16828 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1723 55) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 986)) := by
  have c105321 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.1
  have c105322 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105323 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105324 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105325 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105326 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1723_12 := arithEq_of_rows h (row := 1723) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_12
  simp only [← c105321, ← c105322, ← c105323] at e_1723_12
  have e_1723_13 := arithEq_of_rows h (row := 1723) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_13
  simp only [← c105324, ← c105325, k2, ← c105326] at e_1723_13
  have hr := e_1723_13
  simp only [e_1723_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16829 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1723 63) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 987)) := by
  have c105327 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105328 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105329 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105330 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105331 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105332 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1723_14 := arithEq_of_rows h (row := 1723) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_14
  simp only [← c105327, ← c105328, ← c105329] at e_1723_14
  have e_1723_15 := arithEq_of_rows h (row := 1723) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_15
  simp only [← c105330, ← c105331, k2, ← c105332] at e_1723_15
  have hr := e_1723_15
  simp only [e_1723_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16830 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1723 71) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 988)) := by
  have c105333 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105334 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105335 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105336 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105337 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105338 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1723_16 := arithEq_of_rows h (row := 1723) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_16
  simp only [← c105333, ← c105334, ← c105335] at e_1723_16
  have e_1723_17 := arithEq_of_rows h (row := 1723) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_17
  simp only [← c105336, ← c105337, k2, ← c105338] at e_1723_17
  have hr := e_1723_17
  simp only [e_1723_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16831 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1723 79) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 989)) := by
  have c105339 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105340 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105341 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105342 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105343 := (publicBatchWrapper86_copies3291 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105344 := (publicBatchWrapper86_copies3292 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1723_18 := arithEq_of_rows h (row := 1723) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_18
  simp only [← c105339, ← c105340, ← c105341] at e_1723_18
  have e_1723_19 := arithEq_of_rows h (row := 1723) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1723_19
  simp only [← c105342, ← c105343, k2, ← c105344] at e_1723_19
  have hr := e_1723_19
  simp only [e_1723_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16832 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1724 7) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 990)) := by
  have c105345 := (publicBatchWrapper86_copies3292 a h).2.1
  have c105346 := (publicBatchWrapper86_copies3292 a h).2.2.1
  have c105347 := (publicBatchWrapper86_copies3292 a h).2.2.2.1
  have c105348 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.1
  have c105349 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.1
  have c105350 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1724_0 := arithEq_of_rows h (row := 1724) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_0
  simp only [← c105345, ← c105346, ← c105347] at e_1724_0
  have e_1724_1 := arithEq_of_rows h (row := 1724) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_1
  simp only [← c105348, ← c105349, k2, ← c105350] at e_1724_1
  have hr := e_1724_1
  simp only [e_1724_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16833 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1724 15) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 991)) := by
  have c105351 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.1
  have c105352 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.1
  have c105353 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.1
  have c105354 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105355 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105356 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1724_2 := arithEq_of_rows h (row := 1724) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_2
  simp only [← c105351, ← c105352, ← c105353] at e_1724_2
  have e_1724_3 := arithEq_of_rows h (row := 1724) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_3
  simp only [← c105354, ← c105355, k2, ← c105356] at e_1724_3
  have hr := e_1724_3
  simp only [e_1724_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16834 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1724 23) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 992)) := by
  have c105357 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105358 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105359 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105360 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105361 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105362 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1724_4 := arithEq_of_rows h (row := 1724) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_4
  simp only [← c105357, ← c105358, ← c105359] at e_1724_4
  have e_1724_5 := arithEq_of_rows h (row := 1724) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_5
  simp only [← c105360, ← c105361, k2, ← c105362] at e_1724_5
  have hr := e_1724_5
  simp only [e_1724_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16835 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1724 31) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 993)) := by
  have c105363 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105364 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105365 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105366 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105367 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105368 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1724_6 := arithEq_of_rows h (row := 1724) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_6
  simp only [← c105363, ← c105364, ← c105365] at e_1724_6
  have e_1724_7 := arithEq_of_rows h (row := 1724) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_7
  simp only [← c105366, ← c105367, k2, ← c105368] at e_1724_7
  have hr := e_1724_7
  simp only [e_1724_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16836 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1724 39) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 994)) := by
  have c105369 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105370 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105371 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105372 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105373 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105374 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1724_8 := arithEq_of_rows h (row := 1724) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_8
  simp only [← c105369, ← c105370, ← c105371] at e_1724_8
  have e_1724_9 := arithEq_of_rows h (row := 1724) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_9
  simp only [← c105372, ← c105373, k2, ← c105374] at e_1724_9
  have hr := e_1724_9
  simp only [e_1724_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16837 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1724 47) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 995)) := by
  have c105375 := (publicBatchWrapper86_copies3292 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105376 := (publicBatchWrapper86_copies3293 a h).1
  have c105377 := (publicBatchWrapper86_copies3293 a h).2.1
  have c105378 := (publicBatchWrapper86_copies3293 a h).2.2.1
  have c105379 := (publicBatchWrapper86_copies3293 a h).2.2.2.1
  have c105380 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1724_10 := arithEq_of_rows h (row := 1724) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_10
  simp only [← c105375, ← c105376, ← c105377] at e_1724_10
  have e_1724_11 := arithEq_of_rows h (row := 1724) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_11
  simp only [← c105378, ← c105379, k2, ← c105380] at e_1724_11
  have hr := e_1724_11
  simp only [e_1724_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16838 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1724 55) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 996)) := by
  have c105381 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.1
  have c105382 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.1
  have c105383 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.1
  have c105384 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.1
  have c105385 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.1
  have c105386 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1724_12 := arithEq_of_rows h (row := 1724) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_12
  simp only [← c105381, ← c105382, ← c105383] at e_1724_12
  have e_1724_13 := arithEq_of_rows h (row := 1724) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_13
  simp only [← c105384, ← c105385, k2, ← c105386] at e_1724_13
  have hr := e_1724_13
  simp only [e_1724_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16839 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1724 63) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 997)) := by
  have c105387 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105388 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105389 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105390 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105391 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105392 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1724_14 := arithEq_of_rows h (row := 1724) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_14
  simp only [← c105387, ← c105388, ← c105389] at e_1724_14
  have e_1724_15 := arithEq_of_rows h (row := 1724) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_15
  simp only [← c105390, ← c105391, k2, ← c105392] at e_1724_15
  have hr := e_1724_15
  simp only [e_1724_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16840 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1724 71) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 998)) := by
  have c105393 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105394 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105395 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105396 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105397 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105398 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1724_16 := arithEq_of_rows h (row := 1724) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_16
  simp only [← c105393, ← c105394, ← c105395] at e_1724_16
  have e_1724_17 := arithEq_of_rows h (row := 1724) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_17
  simp only [← c105396, ← c105397, k2, ← c105398] at e_1724_17
  have hr := e_1724_17
  simp only [e_1724_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16841 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1724 79) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 999)) := by
  have c105399 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105400 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105401 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105402 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105403 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105404 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1724_18 := arithEq_of_rows h (row := 1724) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_18
  simp only [← c105399, ← c105400, ← c105401] at e_1724_18
  have e_1724_19 := arithEq_of_rows h (row := 1724) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1724_19
  simp only [← c105402, ← c105403, k2, ← c105404] at e_1724_19
  have hr := e_1724_19
  simp only [e_1724_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16842 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1725 7) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 1000)) := by
  have c105405 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105406 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105407 := (publicBatchWrapper86_copies3293 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105408 := (publicBatchWrapper86_copies3294 a h).1
  have c105409 := (publicBatchWrapper86_copies3294 a h).2.1
  have c105410 := (publicBatchWrapper86_copies3294 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1725_0 := arithEq_of_rows h (row := 1725) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_0
  simp only [← c105405, ← c105406, ← c105407] at e_1725_0
  have e_1725_1 := arithEq_of_rows h (row := 1725) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_1
  simp only [← c105408, ← c105409, k2, ← c105410] at e_1725_1
  have hr := e_1725_1
  simp only [e_1725_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16843 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1725 15) = bselect (a (.wire 6 43)) (a (.virt 200)) (a (.virt 1001)) := by
  have c105411 := (publicBatchWrapper86_copies3294 a h).2.2.2.1
  have c105412 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.1
  have c105413 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.1
  have c105414 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.1
  have c105415 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.1
  have c105416 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1725_2 := arithEq_of_rows h (row := 1725) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_2
  simp only [← c105411, ← c105412, ← c105413] at e_1725_2
  have e_1725_3 := arithEq_of_rows h (row := 1725) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_3
  simp only [← c105414, ← c105415, k2, ← c105416] at e_1725_3
  have hr := e_1725_3
  simp only [e_1725_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16844 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1725 23) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1170)) := by
  have c105417 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.1
  have c105418 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105419 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105420 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105421 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105422 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1725_4 := arithEq_of_rows h (row := 1725) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_4
  simp only [← c105417, ← c105418, ← c105419] at e_1725_4
  have e_1725_5 := arithEq_of_rows h (row := 1725) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_5
  simp only [← c105420, ← c105421, k2, ← c105422] at e_1725_5
  have hr := e_1725_5
  simp only [e_1725_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16845 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1725 31) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1171)) := by
  have c105423 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105424 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105425 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105426 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105427 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105428 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1725_6 := arithEq_of_rows h (row := 1725) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_6
  simp only [← c105423, ← c105424, ← c105425] at e_1725_6
  have e_1725_7 := arithEq_of_rows h (row := 1725) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_7
  simp only [← c105426, ← c105427, k2, ← c105428] at e_1725_7
  have hr := e_1725_7
  simp only [e_1725_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16846 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1725 39) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1172)) := by
  have c105429 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105430 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105431 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105432 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105433 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105434 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1725_8 := arithEq_of_rows h (row := 1725) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_8
  simp only [← c105429, ← c105430, ← c105431] at e_1725_8
  have e_1725_9 := arithEq_of_rows h (row := 1725) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_9
  simp only [← c105432, ← c105433, k2, ← c105434] at e_1725_9
  have hr := e_1725_9
  simp only [e_1725_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16847 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1725 47) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1173)) := by
  have c105435 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105436 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105437 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105438 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105439 := (publicBatchWrapper86_copies3294 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105440 := (publicBatchWrapper86_copies3295 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1725_10 := arithEq_of_rows h (row := 1725) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_10
  simp only [← c105435, ← c105436, ← c105437] at e_1725_10
  have e_1725_11 := arithEq_of_rows h (row := 1725) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_11
  simp only [← c105438, ← c105439, k2, ← c105440] at e_1725_11
  have hr := e_1725_11
  simp only [e_1725_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16848 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1725 55) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1174)) := by
  have c105441 := (publicBatchWrapper86_copies3295 a h).2.1
  have c105442 := (publicBatchWrapper86_copies3295 a h).2.2.1
  have c105443 := (publicBatchWrapper86_copies3295 a h).2.2.2.1
  have c105444 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.1
  have c105445 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.1
  have c105446 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1725_12 := arithEq_of_rows h (row := 1725) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_12
  simp only [← c105441, ← c105442, ← c105443] at e_1725_12
  have e_1725_13 := arithEq_of_rows h (row := 1725) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_13
  simp only [← c105444, ← c105445, k2, ← c105446] at e_1725_13
  have hr := e_1725_13
  simp only [e_1725_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16849 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1725 63) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1175)) := by
  have c105447 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.1
  have c105448 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.1
  have c105449 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.1
  have c105450 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105451 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105452 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1725_14 := arithEq_of_rows h (row := 1725) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_14
  simp only [← c105447, ← c105448, ← c105449] at e_1725_14
  have e_1725_15 := arithEq_of_rows h (row := 1725) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_15
  simp only [← c105450, ← c105451, k2, ← c105452] at e_1725_15
  have hr := e_1725_15
  simp only [e_1725_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16850 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1725 71) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1176)) := by
  have c105453 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105454 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105455 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105456 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105457 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105458 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1725_16 := arithEq_of_rows h (row := 1725) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_16
  simp only [← c105453, ← c105454, ← c105455] at e_1725_16
  have e_1725_17 := arithEq_of_rows h (row := 1725) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_17
  simp only [← c105456, ← c105457, k2, ← c105458] at e_1725_17
  have hr := e_1725_17
  simp only [e_1725_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16851 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1725 79) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1177)) := by
  have c105459 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105460 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105461 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105462 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105463 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105464 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1725_18 := arithEq_of_rows h (row := 1725) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_18
  simp only [← c105459, ← c105460, ← c105461] at e_1725_18
  have e_1725_19 := arithEq_of_rows h (row := 1725) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1725_19
  simp only [← c105462, ← c105463, k2, ← c105464] at e_1725_19
  have hr := e_1725_19
  simp only [e_1725_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16852 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1726 7) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1178)) := by
  have c105465 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105466 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105467 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105468 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105469 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105470 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1726_0 := arithEq_of_rows h (row := 1726) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_0
  simp only [← c105465, ← c105466, ← c105467] at e_1726_0
  have e_1726_1 := arithEq_of_rows h (row := 1726) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_1
  simp only [← c105468, ← c105469, k2, ← c105470] at e_1726_1
  have hr := e_1726_1
  simp only [e_1726_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16853 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1726 15) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1179)) := by
  have c105471 := (publicBatchWrapper86_copies3295 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105472 := (publicBatchWrapper86_copies3296 a h).1
  have c105473 := (publicBatchWrapper86_copies3296 a h).2.1
  have c105474 := (publicBatchWrapper86_copies3296 a h).2.2.1
  have c105475 := (publicBatchWrapper86_copies3296 a h).2.2.2.1
  have c105476 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1726_2 := arithEq_of_rows h (row := 1726) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_2
  simp only [← c105471, ← c105472, ← c105473] at e_1726_2
  have e_1726_3 := arithEq_of_rows h (row := 1726) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_3
  simp only [← c105474, ← c105475, k2, ← c105476] at e_1726_3
  have hr := e_1726_3
  simp only [e_1726_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16854 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1726 23) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1180)) := by
  have c105477 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.1
  have c105478 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.1
  have c105479 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.1
  have c105480 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.1
  have c105481 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.1
  have c105482 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1726_4 := arithEq_of_rows h (row := 1726) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_4
  simp only [← c105477, ← c105478, ← c105479] at e_1726_4
  have e_1726_5 := arithEq_of_rows h (row := 1726) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_5
  simp only [← c105480, ← c105481, k2, ← c105482] at e_1726_5
  have hr := e_1726_5
  simp only [e_1726_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16855 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1726 31) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1181)) := by
  have c105483 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105484 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105485 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105486 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105487 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105488 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1726_6 := arithEq_of_rows h (row := 1726) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_6
  simp only [← c105483, ← c105484, ← c105485] at e_1726_6
  have e_1726_7 := arithEq_of_rows h (row := 1726) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_7
  simp only [← c105486, ← c105487, k2, ← c105488] at e_1726_7
  have hr := e_1726_7
  simp only [e_1726_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16856 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1726 39) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1182)) := by
  have c105489 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105490 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105491 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105492 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105493 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105494 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1726_8 := arithEq_of_rows h (row := 1726) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_8
  simp only [← c105489, ← c105490, ← c105491] at e_1726_8
  have e_1726_9 := arithEq_of_rows h (row := 1726) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_9
  simp only [← c105492, ← c105493, k2, ← c105494] at e_1726_9
  have hr := e_1726_9
  simp only [e_1726_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16857 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1726 47) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1183)) := by
  have c105495 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105496 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105497 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105498 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105499 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105500 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1726_10 := arithEq_of_rows h (row := 1726) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_10
  simp only [← c105495, ← c105496, ← c105497] at e_1726_10
  have e_1726_11 := arithEq_of_rows h (row := 1726) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_11
  simp only [← c105498, ← c105499, k2, ← c105500] at e_1726_11
  have hr := e_1726_11
  simp only [e_1726_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16858 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1726 55) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1184)) := by
  have c105501 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105502 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105503 := (publicBatchWrapper86_copies3296 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105504 := (publicBatchWrapper86_copies3297 a h).1
  have c105505 := (publicBatchWrapper86_copies3297 a h).2.1
  have c105506 := (publicBatchWrapper86_copies3297 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1726_12 := arithEq_of_rows h (row := 1726) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_12
  simp only [← c105501, ← c105502, ← c105503] at e_1726_12
  have e_1726_13 := arithEq_of_rows h (row := 1726) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_13
  simp only [← c105504, ← c105505, k2, ← c105506] at e_1726_13
  have hr := e_1726_13
  simp only [e_1726_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16859 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1726 63) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1185)) := by
  have c105507 := (publicBatchWrapper86_copies3297 a h).2.2.2.1
  have c105508 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.1
  have c105509 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.1
  have c105510 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.1
  have c105511 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.1
  have c105512 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1726_14 := arithEq_of_rows h (row := 1726) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_14
  simp only [← c105507, ← c105508, ← c105509] at e_1726_14
  have e_1726_15 := arithEq_of_rows h (row := 1726) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_15
  simp only [← c105510, ← c105511, k2, ← c105512] at e_1726_15
  have hr := e_1726_15
  simp only [e_1726_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16860 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1726 71) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1186)) := by
  have c105513 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.1
  have c105514 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105515 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105516 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105517 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105518 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1726_16 := arithEq_of_rows h (row := 1726) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_16
  simp only [← c105513, ← c105514, ← c105515] at e_1726_16
  have e_1726_17 := arithEq_of_rows h (row := 1726) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_17
  simp only [← c105516, ← c105517, k2, ← c105518] at e_1726_17
  have hr := e_1726_17
  simp only [e_1726_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16861 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1726 79) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1187)) := by
  have c105519 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105520 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105521 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105522 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105523 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105524 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1726_18 := arithEq_of_rows h (row := 1726) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_18
  simp only [← c105519, ← c105520, ← c105521] at e_1726_18
  have e_1726_19 := arithEq_of_rows h (row := 1726) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1726_19
  simp only [← c105522, ← c105523, k2, ← c105524] at e_1726_19
  have hr := e_1726_19
  simp only [e_1726_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16862 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1727 7) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1188)) := by
  have c105525 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105526 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105527 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105528 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105529 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105530 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1727_0 := arithEq_of_rows h (row := 1727) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_0
  simp only [← c105525, ← c105526, ← c105527] at e_1727_0
  have e_1727_1 := arithEq_of_rows h (row := 1727) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_1
  simp only [← c105528, ← c105529, k2, ← c105530] at e_1727_1
  have hr := e_1727_1
  simp only [e_1727_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16863 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1727 15) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1189)) := by
  have c105531 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105532 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105533 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105534 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105535 := (publicBatchWrapper86_copies3297 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105536 := (publicBatchWrapper86_copies3298 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1727_2 := arithEq_of_rows h (row := 1727) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_2
  simp only [← c105531, ← c105532, ← c105533] at e_1727_2
  have e_1727_3 := arithEq_of_rows h (row := 1727) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_3
  simp only [← c105534, ← c105535, k2, ← c105536] at e_1727_3
  have hr := e_1727_3
  simp only [e_1727_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16864 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1727 23) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1190)) := by
  have c105537 := (publicBatchWrapper86_copies3298 a h).2.1
  have c105538 := (publicBatchWrapper86_copies3298 a h).2.2.1
  have c105539 := (publicBatchWrapper86_copies3298 a h).2.2.2.1
  have c105540 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.1
  have c105541 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.1
  have c105542 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1727_4 := arithEq_of_rows h (row := 1727) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_4
  simp only [← c105537, ← c105538, ← c105539] at e_1727_4
  have e_1727_5 := arithEq_of_rows h (row := 1727) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_5
  simp only [← c105540, ← c105541, k2, ← c105542] at e_1727_5
  have hr := e_1727_5
  simp only [e_1727_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16865 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1727 31) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1191)) := by
  have c105543 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.1
  have c105544 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.1
  have c105545 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.1
  have c105546 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105547 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105548 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1727_6 := arithEq_of_rows h (row := 1727) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_6
  simp only [← c105543, ← c105544, ← c105545] at e_1727_6
  have e_1727_7 := arithEq_of_rows h (row := 1727) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_7
  simp only [← c105546, ← c105547, k2, ← c105548] at e_1727_7
  have hr := e_1727_7
  simp only [e_1727_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16866 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1727 39) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1192)) := by
  have c105549 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105550 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105551 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105552 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105553 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105554 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1727_8 := arithEq_of_rows h (row := 1727) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_8
  simp only [← c105549, ← c105550, ← c105551] at e_1727_8
  have e_1727_9 := arithEq_of_rows h (row := 1727) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_9
  simp only [← c105552, ← c105553, k2, ← c105554] at e_1727_9
  have hr := e_1727_9
  simp only [e_1727_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16867 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1727 47) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1193)) := by
  have c105555 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105556 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105557 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105558 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105559 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105560 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1727_10 := arithEq_of_rows h (row := 1727) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_10
  simp only [← c105555, ← c105556, ← c105557] at e_1727_10
  have e_1727_11 := arithEq_of_rows h (row := 1727) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_11
  simp only [← c105558, ← c105559, k2, ← c105560] at e_1727_11
  have hr := e_1727_11
  simp only [e_1727_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16868 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1727 55) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1194)) := by
  have c105561 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105562 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105563 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105564 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105565 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105566 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1727_12 := arithEq_of_rows h (row := 1727) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_12
  simp only [← c105561, ← c105562, ← c105563] at e_1727_12
  have e_1727_13 := arithEq_of_rows h (row := 1727) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_13
  simp only [← c105564, ← c105565, k2, ← c105566] at e_1727_13
  have hr := e_1727_13
  simp only [e_1727_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16869 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1727 63) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1195)) := by
  have c105567 := (publicBatchWrapper86_copies3298 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105568 := (publicBatchWrapper86_copies3299 a h).1
  have c105569 := (publicBatchWrapper86_copies3299 a h).2.1
  have c105570 := (publicBatchWrapper86_copies3299 a h).2.2.1
  have c105571 := (publicBatchWrapper86_copies3299 a h).2.2.2.1
  have c105572 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1727_14 := arithEq_of_rows h (row := 1727) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_14
  simp only [← c105567, ← c105568, ← c105569] at e_1727_14
  have e_1727_15 := arithEq_of_rows h (row := 1727) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_15
  simp only [← c105570, ← c105571, k2, ← c105572] at e_1727_15
  have hr := e_1727_15
  simp only [e_1727_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16870 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1727 71) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1196)) := by
  have c105573 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.1
  have c105574 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.1
  have c105575 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.1
  have c105576 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.1
  have c105577 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.1
  have c105578 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1727_16 := arithEq_of_rows h (row := 1727) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_16
  simp only [← c105573, ← c105574, ← c105575] at e_1727_16
  have e_1727_17 := arithEq_of_rows h (row := 1727) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_17
  simp only [← c105576, ← c105577, k2, ← c105578] at e_1727_17
  have hr := e_1727_17
  simp only [e_1727_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16871 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1727 79) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1197)) := by
  have c105579 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105580 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105581 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105582 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105583 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105584 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1727_18 := arithEq_of_rows h (row := 1727) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_18
  simp only [← c105579, ← c105580, ← c105581] at e_1727_18
  have e_1727_19 := arithEq_of_rows h (row := 1727) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1727_19
  simp only [← c105582, ← c105583, k2, ← c105584] at e_1727_19
  have hr := e_1727_19
  simp only [e_1727_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16872 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1728 7) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1198)) := by
  have c105585 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105586 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105587 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105588 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105589 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105590 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1728_0 := arithEq_of_rows h (row := 1728) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_0
  simp only [← c105585, ← c105586, ← c105587] at e_1728_0
  have e_1728_1 := arithEq_of_rows h (row := 1728) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_1
  simp only [← c105588, ← c105589, k2, ← c105590] at e_1728_1
  have hr := e_1728_1
  simp only [e_1728_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16873 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1728 15) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1199)) := by
  have c105591 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105592 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105593 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105594 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105595 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105596 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1728_2 := arithEq_of_rows h (row := 1728) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_2
  simp only [← c105591, ← c105592, ← c105593] at e_1728_2
  have e_1728_3 := arithEq_of_rows h (row := 1728) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_3
  simp only [← c105594, ← c105595, k2, ← c105596] at e_1728_3
  have hr := e_1728_3
  simp only [e_1728_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16874 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1728 23) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1200)) := by
  have c105597 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105598 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105599 := (publicBatchWrapper86_copies3299 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105600 := (publicBatchWrapper86_copies3300 a h).1
  have c105601 := (publicBatchWrapper86_copies3300 a h).2.1
  have c105602 := (publicBatchWrapper86_copies3300 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1728_4 := arithEq_of_rows h (row := 1728) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_4
  simp only [← c105597, ← c105598, ← c105599] at e_1728_4
  have e_1728_5 := arithEq_of_rows h (row := 1728) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_5
  simp only [← c105600, ← c105601, k2, ← c105602] at e_1728_5
  have hr := e_1728_5
  simp only [e_1728_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16875 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1728 31) = bselect (a (.wire 8 7)) (a (.virt 200)) (a (.virt 1201)) := by
  have c105603 := (publicBatchWrapper86_copies3300 a h).2.2.2.1
  have c105604 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.1
  have c105605 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.1
  have c105606 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.1
  have c105607 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.1
  have c105608 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1728_6 := arithEq_of_rows h (row := 1728) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_6
  simp only [← c105603, ← c105604, ← c105605] at e_1728_6
  have e_1728_7 := arithEq_of_rows h (row := 1728) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_7
  simp only [← c105606, ← c105607, k2, ← c105608] at e_1728_7
  have hr := e_1728_7
  simp only [e_1728_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16876 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1728 39) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1370)) := by
  have c105609 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.1
  have c105610 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105611 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105612 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105613 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105614 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1728_8 := arithEq_of_rows h (row := 1728) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_8
  simp only [← c105609, ← c105610, ← c105611] at e_1728_8
  have e_1728_9 := arithEq_of_rows h (row := 1728) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_9
  simp only [← c105612, ← c105613, k2, ← c105614] at e_1728_9
  have hr := e_1728_9
  simp only [e_1728_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16877 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1728 47) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1371)) := by
  have c105615 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105616 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105617 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105618 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105619 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105620 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1728_10 := arithEq_of_rows h (row := 1728) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_10
  simp only [← c105615, ← c105616, ← c105617] at e_1728_10
  have e_1728_11 := arithEq_of_rows h (row := 1728) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_11
  simp only [← c105618, ← c105619, k2, ← c105620] at e_1728_11
  have hr := e_1728_11
  simp only [e_1728_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16878 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1728 55) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1372)) := by
  have c105621 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105622 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105623 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105624 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105625 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105626 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1728_12 := arithEq_of_rows h (row := 1728) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_12
  simp only [← c105621, ← c105622, ← c105623] at e_1728_12
  have e_1728_13 := arithEq_of_rows h (row := 1728) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_13
  simp only [← c105624, ← c105625, k2, ← c105626] at e_1728_13
  have hr := e_1728_13
  simp only [e_1728_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16879 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1728 63) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1373)) := by
  have c105627 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105628 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105629 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105630 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105631 := (publicBatchWrapper86_copies3300 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105632 := (publicBatchWrapper86_copies3301 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1728_14 := arithEq_of_rows h (row := 1728) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_14
  simp only [← c105627, ← c105628, ← c105629] at e_1728_14
  have e_1728_15 := arithEq_of_rows h (row := 1728) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_15
  simp only [← c105630, ← c105631, k2, ← c105632] at e_1728_15
  have hr := e_1728_15
  simp only [e_1728_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16880 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1728 71) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1374)) := by
  have c105633 := (publicBatchWrapper86_copies3301 a h).2.1
  have c105634 := (publicBatchWrapper86_copies3301 a h).2.2.1
  have c105635 := (publicBatchWrapper86_copies3301 a h).2.2.2.1
  have c105636 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.1
  have c105637 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.1
  have c105638 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1728_16 := arithEq_of_rows h (row := 1728) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_16
  simp only [← c105633, ← c105634, ← c105635] at e_1728_16
  have e_1728_17 := arithEq_of_rows h (row := 1728) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_17
  simp only [← c105636, ← c105637, k2, ← c105638] at e_1728_17
  have hr := e_1728_17
  simp only [e_1728_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16881 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1728 79) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1375)) := by
  have c105639 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.1
  have c105640 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.1
  have c105641 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.1
  have c105642 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105643 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105644 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1728_18 := arithEq_of_rows h (row := 1728) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_18
  simp only [← c105639, ← c105640, ← c105641] at e_1728_18
  have e_1728_19 := arithEq_of_rows h (row := 1728) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1728_19
  simp only [← c105642, ← c105643, k2, ← c105644] at e_1728_19
  have hr := e_1728_19
  simp only [e_1728_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16882 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1729 7) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1376)) := by
  have c105645 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105646 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105647 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105648 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105649 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105650 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1729_0 := arithEq_of_rows h (row := 1729) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_0
  simp only [← c105645, ← c105646, ← c105647] at e_1729_0
  have e_1729_1 := arithEq_of_rows h (row := 1729) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_1
  simp only [← c105648, ← c105649, k2, ← c105650] at e_1729_1
  have hr := e_1729_1
  simp only [e_1729_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16883 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1729 15) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1377)) := by
  have c105651 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105652 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105653 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105654 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105655 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105656 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1729_2 := arithEq_of_rows h (row := 1729) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_2
  simp only [← c105651, ← c105652, ← c105653] at e_1729_2
  have e_1729_3 := arithEq_of_rows h (row := 1729) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_3
  simp only [← c105654, ← c105655, k2, ← c105656] at e_1729_3
  have hr := e_1729_3
  simp only [e_1729_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16884 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1729 23) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1378)) := by
  have c105657 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105658 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105659 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105660 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105661 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105662 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1729_4 := arithEq_of_rows h (row := 1729) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_4
  simp only [← c105657, ← c105658, ← c105659] at e_1729_4
  have e_1729_5 := arithEq_of_rows h (row := 1729) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_5
  simp only [← c105660, ← c105661, k2, ← c105662] at e_1729_5
  have hr := e_1729_5
  simp only [e_1729_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16885 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1729 31) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1379)) := by
  have c105663 := (publicBatchWrapper86_copies3301 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105664 := (publicBatchWrapper86_copies3302 a h).1
  have c105665 := (publicBatchWrapper86_copies3302 a h).2.1
  have c105666 := (publicBatchWrapper86_copies3302 a h).2.2.1
  have c105667 := (publicBatchWrapper86_copies3302 a h).2.2.2.1
  have c105668 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1729_6 := arithEq_of_rows h (row := 1729) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_6
  simp only [← c105663, ← c105664, ← c105665] at e_1729_6
  have e_1729_7 := arithEq_of_rows h (row := 1729) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_7
  simp only [← c105666, ← c105667, k2, ← c105668] at e_1729_7
  have hr := e_1729_7
  simp only [e_1729_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16886 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1729 39) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1380)) := by
  have c105669 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.1
  have c105670 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.1
  have c105671 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.1
  have c105672 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.1
  have c105673 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.1
  have c105674 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1729_8 := arithEq_of_rows h (row := 1729) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_8
  simp only [← c105669, ← c105670, ← c105671] at e_1729_8
  have e_1729_9 := arithEq_of_rows h (row := 1729) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_9
  simp only [← c105672, ← c105673, k2, ← c105674] at e_1729_9
  have hr := e_1729_9
  simp only [e_1729_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16887 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1729 47) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1381)) := by
  have c105675 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105676 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105677 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105678 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105679 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105680 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1729_10 := arithEq_of_rows h (row := 1729) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_10
  simp only [← c105675, ← c105676, ← c105677] at e_1729_10
  have e_1729_11 := arithEq_of_rows h (row := 1729) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_11
  simp only [← c105678, ← c105679, k2, ← c105680] at e_1729_11
  have hr := e_1729_11
  simp only [e_1729_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16888 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1729 55) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1382)) := by
  have c105681 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105682 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105683 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105684 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105685 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105686 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1729_12 := arithEq_of_rows h (row := 1729) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_12
  simp only [← c105681, ← c105682, ← c105683] at e_1729_12
  have e_1729_13 := arithEq_of_rows h (row := 1729) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_13
  simp only [← c105684, ← c105685, k2, ← c105686] at e_1729_13
  have hr := e_1729_13
  simp only [e_1729_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16889 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1729 63) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1383)) := by
  have c105687 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105688 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105689 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105690 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105691 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105692 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1729_14 := arithEq_of_rows h (row := 1729) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_14
  simp only [← c105687, ← c105688, ← c105689] at e_1729_14
  have e_1729_15 := arithEq_of_rows h (row := 1729) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_15
  simp only [← c105690, ← c105691, k2, ← c105692] at e_1729_15
  have hr := e_1729_15
  simp only [e_1729_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16890 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1729 71) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1384)) := by
  have c105693 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105694 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105695 := (publicBatchWrapper86_copies3302 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105696 := (publicBatchWrapper86_copies3303 a h).1
  have c105697 := (publicBatchWrapper86_copies3303 a h).2.1
  have c105698 := (publicBatchWrapper86_copies3303 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1729_16 := arithEq_of_rows h (row := 1729) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_16
  simp only [← c105693, ← c105694, ← c105695] at e_1729_16
  have e_1729_17 := arithEq_of_rows h (row := 1729) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_17
  simp only [← c105696, ← c105697, k2, ← c105698] at e_1729_17
  have hr := e_1729_17
  simp only [e_1729_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16891 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1729 79) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1385)) := by
  have c105699 := (publicBatchWrapper86_copies3303 a h).2.2.2.1
  have c105700 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.1
  have c105701 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.1
  have c105702 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.1
  have c105703 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.1
  have c105704 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1729_18 := arithEq_of_rows h (row := 1729) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_18
  simp only [← c105699, ← c105700, ← c105701] at e_1729_18
  have e_1729_19 := arithEq_of_rows h (row := 1729) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1729_19
  simp only [← c105702, ← c105703, k2, ← c105704] at e_1729_19
  have hr := e_1729_19
  simp only [e_1729_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16892 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1730 7) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1386)) := by
  have c105705 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.1
  have c105706 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105707 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105708 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105709 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105710 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1730_0 := arithEq_of_rows h (row := 1730) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_0
  simp only [← c105705, ← c105706, ← c105707] at e_1730_0
  have e_1730_1 := arithEq_of_rows h (row := 1730) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_1
  simp only [← c105708, ← c105709, k2, ← c105710] at e_1730_1
  have hr := e_1730_1
  simp only [e_1730_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16893 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1730 15) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1387)) := by
  have c105711 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105712 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105713 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105714 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105715 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105716 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1730_2 := arithEq_of_rows h (row := 1730) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_2
  simp only [← c105711, ← c105712, ← c105713] at e_1730_2
  have e_1730_3 := arithEq_of_rows h (row := 1730) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_3
  simp only [← c105714, ← c105715, k2, ← c105716] at e_1730_3
  have hr := e_1730_3
  simp only [e_1730_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16894 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1730 23) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1388)) := by
  have c105717 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105718 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105719 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105720 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105721 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105722 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1730_4 := arithEq_of_rows h (row := 1730) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_4
  simp only [← c105717, ← c105718, ← c105719] at e_1730_4
  have e_1730_5 := arithEq_of_rows h (row := 1730) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_5
  simp only [← c105720, ← c105721, k2, ← c105722] at e_1730_5
  have hr := e_1730_5
  simp only [e_1730_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16895 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1730 31) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1389)) := by
  have c105723 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105724 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105725 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105726 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105727 := (publicBatchWrapper86_copies3303 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105728 := (publicBatchWrapper86_copies3304 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1730_6 := arithEq_of_rows h (row := 1730) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_6
  simp only [← c105723, ← c105724, ← c105725] at e_1730_6
  have e_1730_7 := arithEq_of_rows h (row := 1730) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_7
  simp only [← c105726, ← c105727, k2, ← c105728] at e_1730_7
  have hr := e_1730_7
  simp only [e_1730_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16896 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1730 39) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1390)) := by
  have c105729 := (publicBatchWrapper86_copies3304 a h).2.1
  have c105730 := (publicBatchWrapper86_copies3304 a h).2.2.1
  have c105731 := (publicBatchWrapper86_copies3304 a h).2.2.2.1
  have c105732 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.1
  have c105733 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.1
  have c105734 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1730_8 := arithEq_of_rows h (row := 1730) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_8
  simp only [← c105729, ← c105730, ← c105731] at e_1730_8
  have e_1730_9 := arithEq_of_rows h (row := 1730) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_9
  simp only [← c105732, ← c105733, k2, ← c105734] at e_1730_9
  have hr := e_1730_9
  simp only [e_1730_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16897 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1730 47) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1391)) := by
  have c105735 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.1
  have c105736 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.1
  have c105737 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.1
  have c105738 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105739 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105740 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1730_10 := arithEq_of_rows h (row := 1730) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_10
  simp only [← c105735, ← c105736, ← c105737] at e_1730_10
  have e_1730_11 := arithEq_of_rows h (row := 1730) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_11
  simp only [← c105738, ← c105739, k2, ← c105740] at e_1730_11
  have hr := e_1730_11
  simp only [e_1730_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16898 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1730 55) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1392)) := by
  have c105741 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105742 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105743 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105744 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105745 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105746 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1730_12 := arithEq_of_rows h (row := 1730) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_12
  simp only [← c105741, ← c105742, ← c105743] at e_1730_12
  have e_1730_13 := arithEq_of_rows h (row := 1730) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_13
  simp only [← c105744, ← c105745, k2, ← c105746] at e_1730_13
  have hr := e_1730_13
  simp only [e_1730_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16899 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1730 63) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1393)) := by
  have c105747 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105748 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105749 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105750 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105751 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105752 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1730_14 := arithEq_of_rows h (row := 1730) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_14
  simp only [← c105747, ← c105748, ← c105749] at e_1730_14
  have e_1730_15 := arithEq_of_rows h (row := 1730) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_15
  simp only [← c105750, ← c105751, k2, ← c105752] at e_1730_15
  have hr := e_1730_15
  simp only [e_1730_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16900 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1730 71) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1394)) := by
  have c105753 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105754 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105755 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105756 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105757 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105758 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1730_16 := arithEq_of_rows h (row := 1730) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_16
  simp only [← c105753, ← c105754, ← c105755] at e_1730_16
  have e_1730_17 := arithEq_of_rows h (row := 1730) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_17
  simp only [← c105756, ← c105757, k2, ← c105758] at e_1730_17
  have hr := e_1730_17
  simp only [e_1730_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16901 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1730 79) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1395)) := by
  have c105759 := (publicBatchWrapper86_copies3304 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105760 := (publicBatchWrapper86_copies3305 a h).1
  have c105761 := (publicBatchWrapper86_copies3305 a h).2.1
  have c105762 := (publicBatchWrapper86_copies3305 a h).2.2.1
  have c105763 := (publicBatchWrapper86_copies3305 a h).2.2.2.1
  have c105764 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1730_18 := arithEq_of_rows h (row := 1730) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_18
  simp only [← c105759, ← c105760, ← c105761] at e_1730_18
  have e_1730_19 := arithEq_of_rows h (row := 1730) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1730_19
  simp only [← c105762, ← c105763, k2, ← c105764] at e_1730_19
  have hr := e_1730_19
  simp only [e_1730_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16902 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1731 7) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1396)) := by
  have c105765 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.1
  have c105766 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.1
  have c105767 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.1
  have c105768 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.1
  have c105769 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.1
  have c105770 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1731_0 := arithEq_of_rows h (row := 1731) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_0
  simp only [← c105765, ← c105766, ← c105767] at e_1731_0
  have e_1731_1 := arithEq_of_rows h (row := 1731) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_1
  simp only [← c105768, ← c105769, k2, ← c105770] at e_1731_1
  have hr := e_1731_1
  simp only [e_1731_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16903 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1731 15) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1397)) := by
  have c105771 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105772 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105773 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105774 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105775 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105776 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1731_2 := arithEq_of_rows h (row := 1731) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_2
  simp only [← c105771, ← c105772, ← c105773] at e_1731_2
  have e_1731_3 := arithEq_of_rows h (row := 1731) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_3
  simp only [← c105774, ← c105775, k2, ← c105776] at e_1731_3
  have hr := e_1731_3
  simp only [e_1731_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16904 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1731 23) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1398)) := by
  have c105777 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105778 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105779 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105780 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105781 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105782 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1731_4 := arithEq_of_rows h (row := 1731) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_4
  simp only [← c105777, ← c105778, ← c105779] at e_1731_4
  have e_1731_5 := arithEq_of_rows h (row := 1731) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_5
  simp only [← c105780, ← c105781, k2, ← c105782] at e_1731_5
  have hr := e_1731_5
  simp only [e_1731_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16905 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1731 31) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1399)) := by
  have c105783 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105784 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105785 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105786 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105787 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105788 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1731_6 := arithEq_of_rows h (row := 1731) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_6
  simp only [← c105783, ← c105784, ← c105785] at e_1731_6
  have e_1731_7 := arithEq_of_rows h (row := 1731) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_7
  simp only [← c105786, ← c105787, k2, ← c105788] at e_1731_7
  have hr := e_1731_7
  simp only [e_1731_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16906 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1731 39) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1400)) := by
  have c105789 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105790 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105791 := (publicBatchWrapper86_copies3305 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105792 := (publicBatchWrapper86_copies3306 a h).1
  have c105793 := (publicBatchWrapper86_copies3306 a h).2.1
  have c105794 := (publicBatchWrapper86_copies3306 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1731_8 := arithEq_of_rows h (row := 1731) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_8
  simp only [← c105789, ← c105790, ← c105791] at e_1731_8
  have e_1731_9 := arithEq_of_rows h (row := 1731) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_9
  simp only [← c105792, ← c105793, k2, ← c105794] at e_1731_9
  have hr := e_1731_9
  simp only [e_1731_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16907 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1731 47) = bselect (a (.wire 8 51)) (a (.virt 200)) (a (.virt 1401)) := by
  have c105795 := (publicBatchWrapper86_copies3306 a h).2.2.2.1
  have c105796 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.1
  have c105797 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.1
  have c105798 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.1
  have c105799 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.1
  have c105800 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1731_10 := arithEq_of_rows h (row := 1731) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_10
  simp only [← c105795, ← c105796, ← c105797] at e_1731_10
  have e_1731_11 := arithEq_of_rows h (row := 1731) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_11
  simp only [← c105798, ← c105799, k2, ← c105800] at e_1731_11
  have hr := e_1731_11
  simp only [e_1731_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16908 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1731 55) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1570)) := by
  have c105801 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.1
  have c105802 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105803 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105804 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105805 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105806 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1731_12 := arithEq_of_rows h (row := 1731) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_12
  simp only [← c105801, ← c105802, ← c105803] at e_1731_12
  have e_1731_13 := arithEq_of_rows h (row := 1731) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_13
  simp only [← c105804, ← c105805, k2, ← c105806] at e_1731_13
  have hr := e_1731_13
  simp only [e_1731_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16909 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1731 63) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1571)) := by
  have c105807 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105808 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105809 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105810 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105811 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105812 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1731_14 := arithEq_of_rows h (row := 1731) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_14
  simp only [← c105807, ← c105808, ← c105809] at e_1731_14
  have e_1731_15 := arithEq_of_rows h (row := 1731) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_15
  simp only [← c105810, ← c105811, k2, ← c105812] at e_1731_15
  have hr := e_1731_15
  simp only [e_1731_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16910 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1731 71) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1572)) := by
  have c105813 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105814 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105815 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105816 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105817 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105818 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1731_16 := arithEq_of_rows h (row := 1731) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_16
  simp only [← c105813, ← c105814, ← c105815] at e_1731_16
  have e_1731_17 := arithEq_of_rows h (row := 1731) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_17
  simp only [← c105816, ← c105817, k2, ← c105818] at e_1731_17
  have hr := e_1731_17
  simp only [e_1731_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16911 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1731 79) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1573)) := by
  have c105819 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105820 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105821 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105822 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105823 := (publicBatchWrapper86_copies3306 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105824 := (publicBatchWrapper86_copies3307 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1731_18 := arithEq_of_rows h (row := 1731) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_18
  simp only [← c105819, ← c105820, ← c105821] at e_1731_18
  have e_1731_19 := arithEq_of_rows h (row := 1731) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1731_19
  simp only [← c105822, ← c105823, k2, ← c105824] at e_1731_19
  have hr := e_1731_19
  simp only [e_1731_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16912 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1732 7) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1574)) := by
  have c105825 := (publicBatchWrapper86_copies3307 a h).2.1
  have c105826 := (publicBatchWrapper86_copies3307 a h).2.2.1
  have c105827 := (publicBatchWrapper86_copies3307 a h).2.2.2.1
  have c105828 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.1
  have c105829 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.1
  have c105830 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1732_0 := arithEq_of_rows h (row := 1732) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_0
  simp only [← c105825, ← c105826, ← c105827] at e_1732_0
  have e_1732_1 := arithEq_of_rows h (row := 1732) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_1
  simp only [← c105828, ← c105829, k2, ← c105830] at e_1732_1
  have hr := e_1732_1
  simp only [e_1732_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16913 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1732 15) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1575)) := by
  have c105831 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.1
  have c105832 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.1
  have c105833 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.1
  have c105834 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105835 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105836 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1732_2 := arithEq_of_rows h (row := 1732) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_2
  simp only [← c105831, ← c105832, ← c105833] at e_1732_2
  have e_1732_3 := arithEq_of_rows h (row := 1732) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_3
  simp only [← c105834, ← c105835, k2, ← c105836] at e_1732_3
  have hr := e_1732_3
  simp only [e_1732_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16914 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1732 23) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1576)) := by
  have c105837 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105838 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105839 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105840 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105841 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105842 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1732_4 := arithEq_of_rows h (row := 1732) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_4
  simp only [← c105837, ← c105838, ← c105839] at e_1732_4
  have e_1732_5 := arithEq_of_rows h (row := 1732) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_5
  simp only [← c105840, ← c105841, k2, ← c105842] at e_1732_5
  have hr := e_1732_5
  simp only [e_1732_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16915 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1732 31) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1577)) := by
  have c105843 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105844 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105845 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105846 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105847 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105848 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1732_6 := arithEq_of_rows h (row := 1732) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_6
  simp only [← c105843, ← c105844, ← c105845] at e_1732_6
  have e_1732_7 := arithEq_of_rows h (row := 1732) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_7
  simp only [← c105846, ← c105847, k2, ← c105848] at e_1732_7
  have hr := e_1732_7
  simp only [e_1732_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16916 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1732 39) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1578)) := by
  have c105849 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105850 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105851 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105852 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105853 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105854 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1732_8 := arithEq_of_rows h (row := 1732) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_8
  simp only [← c105849, ← c105850, ← c105851] at e_1732_8
  have e_1732_9 := arithEq_of_rows h (row := 1732) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_9
  simp only [← c105852, ← c105853, k2, ← c105854] at e_1732_9
  have hr := e_1732_9
  simp only [e_1732_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16917 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1732 47) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1579)) := by
  have c105855 := (publicBatchWrapper86_copies3307 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105856 := (publicBatchWrapper86_copies3308 a h).1
  have c105857 := (publicBatchWrapper86_copies3308 a h).2.1
  have c105858 := (publicBatchWrapper86_copies3308 a h).2.2.1
  have c105859 := (publicBatchWrapper86_copies3308 a h).2.2.2.1
  have c105860 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1732_10 := arithEq_of_rows h (row := 1732) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_10
  simp only [← c105855, ← c105856, ← c105857] at e_1732_10
  have e_1732_11 := arithEq_of_rows h (row := 1732) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_11
  simp only [← c105858, ← c105859, k2, ← c105860] at e_1732_11
  have hr := e_1732_11
  simp only [e_1732_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16918 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1732 55) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1580)) := by
  have c105861 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.1
  have c105862 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.1
  have c105863 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.1
  have c105864 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.1
  have c105865 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.1
  have c105866 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1732_12 := arithEq_of_rows h (row := 1732) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_12
  simp only [← c105861, ← c105862, ← c105863] at e_1732_12
  have e_1732_13 := arithEq_of_rows h (row := 1732) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_13
  simp only [← c105864, ← c105865, k2, ← c105866] at e_1732_13
  have hr := e_1732_13
  simp only [e_1732_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16919 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1732 63) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1581)) := by
  have c105867 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105868 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105869 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105870 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105871 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105872 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1732_14 := arithEq_of_rows h (row := 1732) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_14
  simp only [← c105867, ← c105868, ← c105869] at e_1732_14
  have e_1732_15 := arithEq_of_rows h (row := 1732) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_15
  simp only [← c105870, ← c105871, k2, ← c105872] at e_1732_15
  have hr := e_1732_15
  simp only [e_1732_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16920 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1732 71) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1582)) := by
  have c105873 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105874 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105875 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105876 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105877 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105878 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1732_16 := arithEq_of_rows h (row := 1732) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_16
  simp only [← c105873, ← c105874, ← c105875] at e_1732_16
  have e_1732_17 := arithEq_of_rows h (row := 1732) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_17
  simp only [← c105876, ← c105877, k2, ← c105878] at e_1732_17
  have hr := e_1732_17
  simp only [e_1732_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16921 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1732 79) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1583)) := by
  have c105879 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105880 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105881 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105882 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105883 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105884 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1732_18 := arithEq_of_rows h (row := 1732) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_18
  simp only [← c105879, ← c105880, ← c105881] at e_1732_18
  have e_1732_19 := arithEq_of_rows h (row := 1732) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1732_19
  simp only [← c105882, ← c105883, k2, ← c105884] at e_1732_19
  have hr := e_1732_19
  simp only [e_1732_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16922 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1733 7) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1584)) := by
  have c105885 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105886 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105887 := (publicBatchWrapper86_copies3308 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105888 := (publicBatchWrapper86_copies3309 a h).1
  have c105889 := (publicBatchWrapper86_copies3309 a h).2.1
  have c105890 := (publicBatchWrapper86_copies3309 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1733_0 := arithEq_of_rows h (row := 1733) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_0
  simp only [← c105885, ← c105886, ← c105887] at e_1733_0
  have e_1733_1 := arithEq_of_rows h (row := 1733) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_1
  simp only [← c105888, ← c105889, k2, ← c105890] at e_1733_1
  have hr := e_1733_1
  simp only [e_1733_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16923 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1733 15) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1585)) := by
  have c105891 := (publicBatchWrapper86_copies3309 a h).2.2.2.1
  have c105892 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.1
  have c105893 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.1
  have c105894 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.1
  have c105895 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.1
  have c105896 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1733_2 := arithEq_of_rows h (row := 1733) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_2
  simp only [← c105891, ← c105892, ← c105893] at e_1733_2
  have e_1733_3 := arithEq_of_rows h (row := 1733) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_3
  simp only [← c105894, ← c105895, k2, ← c105896] at e_1733_3
  have hr := e_1733_3
  simp only [e_1733_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16924 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1733 23) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1586)) := by
  have c105897 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.1
  have c105898 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105899 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105900 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105901 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105902 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1733_4 := arithEq_of_rows h (row := 1733) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_4
  simp only [← c105897, ← c105898, ← c105899] at e_1733_4
  have e_1733_5 := arithEq_of_rows h (row := 1733) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_5
  simp only [← c105900, ← c105901, k2, ← c105902] at e_1733_5
  have hr := e_1733_5
  simp only [e_1733_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16925 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1733 31) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1587)) := by
  have c105903 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105904 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105905 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105906 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105907 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105908 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1733_6 := arithEq_of_rows h (row := 1733) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_6
  simp only [← c105903, ← c105904, ← c105905] at e_1733_6
  have e_1733_7 := arithEq_of_rows h (row := 1733) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_7
  simp only [← c105906, ← c105907, k2, ← c105908] at e_1733_7
  have hr := e_1733_7
  simp only [e_1733_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16926 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1733 39) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1588)) := by
  have c105909 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105910 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105911 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105912 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105913 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105914 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1733_8 := arithEq_of_rows h (row := 1733) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_8
  simp only [← c105909, ← c105910, ← c105911] at e_1733_8
  have e_1733_9 := arithEq_of_rows h (row := 1733) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_9
  simp only [← c105912, ← c105913, k2, ← c105914] at e_1733_9
  have hr := e_1733_9
  simp only [e_1733_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16927 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1733 47) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1589)) := by
  have c105915 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105916 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105917 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105918 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105919 := (publicBatchWrapper86_copies3309 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105920 := (publicBatchWrapper86_copies3310 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1733_10 := arithEq_of_rows h (row := 1733) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_10
  simp only [← c105915, ← c105916, ← c105917] at e_1733_10
  have e_1733_11 := arithEq_of_rows h (row := 1733) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_11
  simp only [← c105918, ← c105919, k2, ← c105920] at e_1733_11
  have hr := e_1733_11
  simp only [e_1733_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16928 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1733 55) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1590)) := by
  have c105921 := (publicBatchWrapper86_copies3310 a h).2.1
  have c105922 := (publicBatchWrapper86_copies3310 a h).2.2.1
  have c105923 := (publicBatchWrapper86_copies3310 a h).2.2.2.1
  have c105924 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.1
  have c105925 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.1
  have c105926 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1733_12 := arithEq_of_rows h (row := 1733) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_12
  simp only [← c105921, ← c105922, ← c105923] at e_1733_12
  have e_1733_13 := arithEq_of_rows h (row := 1733) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_13
  simp only [← c105924, ← c105925, k2, ← c105926] at e_1733_13
  have hr := e_1733_13
  simp only [e_1733_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16929 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1733 63) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1591)) := by
  have c105927 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.1
  have c105928 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.1
  have c105929 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.1
  have c105930 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105931 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105932 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1733_14 := arithEq_of_rows h (row := 1733) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_14
  simp only [← c105927, ← c105928, ← c105929] at e_1733_14
  have e_1733_15 := arithEq_of_rows h (row := 1733) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_15
  simp only [← c105930, ← c105931, k2, ← c105932] at e_1733_15
  have hr := e_1733_15
  simp only [e_1733_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16930 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1733 71) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1592)) := by
  have c105933 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105934 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105935 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105936 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105937 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105938 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1733_16 := arithEq_of_rows h (row := 1733) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_16
  simp only [← c105933, ← c105934, ← c105935] at e_1733_16
  have e_1733_17 := arithEq_of_rows h (row := 1733) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_17
  simp only [← c105936, ← c105937, k2, ← c105938] at e_1733_17
  have hr := e_1733_17
  simp only [e_1733_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16931 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1733 79) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1593)) := by
  have c105939 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105940 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105941 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105942 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105943 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105944 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1733_18 := arithEq_of_rows h (row := 1733) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_18
  simp only [← c105939, ← c105940, ← c105941] at e_1733_18
  have e_1733_19 := arithEq_of_rows h (row := 1733) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1733_19
  simp only [← c105942, ← c105943, k2, ← c105944] at e_1733_19
  have hr := e_1733_19
  simp only [e_1733_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16932 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1734 7) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1594)) := by
  have c105945 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105946 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105947 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105948 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105949 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105950 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1734_0 := arithEq_of_rows h (row := 1734) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_0
  simp only [← c105945, ← c105946, ← c105947] at e_1734_0
  have e_1734_1 := arithEq_of_rows h (row := 1734) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_1
  simp only [← c105948, ← c105949, k2, ← c105950] at e_1734_1
  have hr := e_1734_1
  simp only [e_1734_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16933 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1734 15) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1595)) := by
  have c105951 := (publicBatchWrapper86_copies3310 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105952 := (publicBatchWrapper86_copies3311 a h).1
  have c105953 := (publicBatchWrapper86_copies3311 a h).2.1
  have c105954 := (publicBatchWrapper86_copies3311 a h).2.2.1
  have c105955 := (publicBatchWrapper86_copies3311 a h).2.2.2.1
  have c105956 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1734_2 := arithEq_of_rows h (row := 1734) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_2
  simp only [← c105951, ← c105952, ← c105953] at e_1734_2
  have e_1734_3 := arithEq_of_rows h (row := 1734) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_3
  simp only [← c105954, ← c105955, k2, ← c105956] at e_1734_3
  have hr := e_1734_3
  simp only [e_1734_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16934 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1734 23) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1596)) := by
  have c105957 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.1
  have c105958 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.1
  have c105959 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.1
  have c105960 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.1
  have c105961 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.1
  have c105962 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1734_4 := arithEq_of_rows h (row := 1734) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_4
  simp only [← c105957, ← c105958, ← c105959] at e_1734_4
  have e_1734_5 := arithEq_of_rows h (row := 1734) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_5
  simp only [← c105960, ← c105961, k2, ← c105962] at e_1734_5
  have hr := e_1734_5
  simp only [e_1734_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16935 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1734 31) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1597)) := by
  have c105963 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105964 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105965 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105966 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105967 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105968 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1734_6 := arithEq_of_rows h (row := 1734) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_6
  simp only [← c105963, ← c105964, ← c105965] at e_1734_6
  have e_1734_7 := arithEq_of_rows h (row := 1734) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_7
  simp only [← c105966, ← c105967, k2, ← c105968] at e_1734_7
  have hr := e_1734_7
  simp only [e_1734_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16936 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1734 39) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1598)) := by
  have c105969 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105970 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105971 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105972 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105973 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105974 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1734_8 := arithEq_of_rows h (row := 1734) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_8
  simp only [← c105969, ← c105970, ← c105971] at e_1734_8
  have e_1734_9 := arithEq_of_rows h (row := 1734) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_9
  simp only [← c105972, ← c105973, k2, ← c105974] at e_1734_9
  have hr := e_1734_9
  simp only [e_1734_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16937 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1734 47) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1599)) := by
  have c105975 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105976 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105977 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105978 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105979 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105980 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1734_10 := arithEq_of_rows h (row := 1734) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_10
  simp only [← c105975, ← c105976, ← c105977] at e_1734_10
  have e_1734_11 := arithEq_of_rows h (row := 1734) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_11
  simp only [← c105978, ← c105979, k2, ← c105980] at e_1734_11
  have hr := e_1734_11
  simp only [e_1734_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16938 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1734 55) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1600)) := by
  have c105981 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105982 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105983 := (publicBatchWrapper86_copies3311 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c105984 := (publicBatchWrapper86_copies3312 a h).1
  have c105985 := (publicBatchWrapper86_copies3312 a h).2.1
  have c105986 := (publicBatchWrapper86_copies3312 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1734_12 := arithEq_of_rows h (row := 1734) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_12
  simp only [← c105981, ← c105982, ← c105983] at e_1734_12
  have e_1734_13 := arithEq_of_rows h (row := 1734) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_13
  simp only [← c105984, ← c105985, k2, ← c105986] at e_1734_13
  have hr := e_1734_13
  simp only [e_1734_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16939 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1734 63) = bselect (a (.wire 10 15)) (a (.virt 200)) (a (.virt 1601)) := by
  have c105987 := (publicBatchWrapper86_copies3312 a h).2.2.2.1
  have c105988 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.1
  have c105989 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.1
  have c105990 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.1
  have c105991 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.1
  have c105992 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1734_14 := arithEq_of_rows h (row := 1734) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_14
  simp only [← c105987, ← c105988, ← c105989] at e_1734_14
  have e_1734_15 := arithEq_of_rows h (row := 1734) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_15
  simp only [← c105990, ← c105991, k2, ← c105992] at e_1734_15
  have hr := e_1734_15
  simp only [e_1734_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16940 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1734 71) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1770)) := by
  have c105993 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.1
  have c105994 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.1
  have c105995 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c105996 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105997 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c105998 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1734_16 := arithEq_of_rows h (row := 1734) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_16
  simp only [← c105993, ← c105994, ← c105995] at e_1734_16
  have e_1734_17 := arithEq_of_rows h (row := 1734) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_17
  simp only [← c105996, ← c105997, k2, ← c105998] at e_1734_17
  have hr := e_1734_17
  simp only [e_1734_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16941 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1734 79) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1771)) := by
  have c105999 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106000 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106001 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106002 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106003 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106004 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1734_18 := arithEq_of_rows h (row := 1734) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_18
  simp only [← c105999, ← c106000, ← c106001] at e_1734_18
  have e_1734_19 := arithEq_of_rows h (row := 1734) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1734_19
  simp only [← c106002, ← c106003, k2, ← c106004] at e_1734_19
  have hr := e_1734_19
  simp only [e_1734_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16942 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1735 7) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1772)) := by
  have c106005 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106006 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106007 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106008 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106009 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106010 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1735_0 := arithEq_of_rows h (row := 1735) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_0
  simp only [← c106005, ← c106006, ← c106007] at e_1735_0
  have e_1735_1 := arithEq_of_rows h (row := 1735) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_1
  simp only [← c106008, ← c106009, k2, ← c106010] at e_1735_1
  have hr := e_1735_1
  simp only [e_1735_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16943 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1735 15) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1773)) := by
  have c106011 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106012 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106013 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106014 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106015 := (publicBatchWrapper86_copies3312 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106016 := (publicBatchWrapper86_copies3313 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1735_2 := arithEq_of_rows h (row := 1735) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_2
  simp only [← c106011, ← c106012, ← c106013] at e_1735_2
  have e_1735_3 := arithEq_of_rows h (row := 1735) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_3
  simp only [← c106014, ← c106015, k2, ← c106016] at e_1735_3
  have hr := e_1735_3
  simp only [e_1735_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16944 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1735 23) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1774)) := by
  have c106017 := (publicBatchWrapper86_copies3313 a h).2.1
  have c106018 := (publicBatchWrapper86_copies3313 a h).2.2.1
  have c106019 := (publicBatchWrapper86_copies3313 a h).2.2.2.1
  have c106020 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.1
  have c106021 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.1
  have c106022 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1735_4 := arithEq_of_rows h (row := 1735) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_4
  simp only [← c106017, ← c106018, ← c106019] at e_1735_4
  have e_1735_5 := arithEq_of_rows h (row := 1735) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_5
  simp only [← c106020, ← c106021, k2, ← c106022] at e_1735_5
  have hr := e_1735_5
  simp only [e_1735_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16945 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1735 31) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1775)) := by
  have c106023 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.1
  have c106024 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.1
  have c106025 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.1
  have c106026 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106027 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106028 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1735_6 := arithEq_of_rows h (row := 1735) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_6
  simp only [← c106023, ← c106024, ← c106025] at e_1735_6
  have e_1735_7 := arithEq_of_rows h (row := 1735) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_7
  simp only [← c106026, ← c106027, k2, ← c106028] at e_1735_7
  have hr := e_1735_7
  simp only [e_1735_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16946 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1735 39) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1776)) := by
  have c106029 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106030 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106031 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106032 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106033 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106034 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1735_8 := arithEq_of_rows h (row := 1735) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_8
  simp only [← c106029, ← c106030, ← c106031] at e_1735_8
  have e_1735_9 := arithEq_of_rows h (row := 1735) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_9
  simp only [← c106032, ← c106033, k2, ← c106034] at e_1735_9
  have hr := e_1735_9
  simp only [e_1735_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16947 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1735 47) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1777)) := by
  have c106035 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106036 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106037 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106038 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106039 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106040 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1735_10 := arithEq_of_rows h (row := 1735) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_10
  simp only [← c106035, ← c106036, ← c106037] at e_1735_10
  have e_1735_11 := arithEq_of_rows h (row := 1735) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_11
  simp only [← c106038, ← c106039, k2, ← c106040] at e_1735_11
  have hr := e_1735_11
  simp only [e_1735_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16948 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1735 55) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1778)) := by
  have c106041 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106042 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106043 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106044 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106045 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106046 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1735_12 := arithEq_of_rows h (row := 1735) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_12
  simp only [← c106041, ← c106042, ← c106043] at e_1735_12
  have e_1735_13 := arithEq_of_rows h (row := 1735) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_13
  simp only [← c106044, ← c106045, k2, ← c106046] at e_1735_13
  have hr := e_1735_13
  simp only [e_1735_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16949 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1735 63) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1779)) := by
  have c106047 := (publicBatchWrapper86_copies3313 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106048 := (publicBatchWrapper86_copies3314 a h).1
  have c106049 := (publicBatchWrapper86_copies3314 a h).2.1
  have c106050 := (publicBatchWrapper86_copies3314 a h).2.2.1
  have c106051 := (publicBatchWrapper86_copies3314 a h).2.2.2.1
  have c106052 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1735_14 := arithEq_of_rows h (row := 1735) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_14
  simp only [← c106047, ← c106048, ← c106049] at e_1735_14
  have e_1735_15 := arithEq_of_rows h (row := 1735) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_15
  simp only [← c106050, ← c106051, k2, ← c106052] at e_1735_15
  have hr := e_1735_15
  simp only [e_1735_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16950 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1735 71) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1780)) := by
  have c106053 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.1
  have c106054 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.1
  have c106055 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.1
  have c106056 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.1
  have c106057 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.1
  have c106058 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1735_16 := arithEq_of_rows h (row := 1735) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_16
  simp only [← c106053, ← c106054, ← c106055] at e_1735_16
  have e_1735_17 := arithEq_of_rows h (row := 1735) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_17
  simp only [← c106056, ← c106057, k2, ← c106058] at e_1735_17
  have hr := e_1735_17
  simp only [e_1735_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16951 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1735 79) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1781)) := by
  have c106059 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106060 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106061 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106062 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106063 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106064 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1735_18 := arithEq_of_rows h (row := 1735) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_18
  simp only [← c106059, ← c106060, ← c106061] at e_1735_18
  have e_1735_19 := arithEq_of_rows h (row := 1735) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1735_19
  simp only [← c106062, ← c106063, k2, ← c106064] at e_1735_19
  have hr := e_1735_19
  simp only [e_1735_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16952 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1736 7) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1782)) := by
  have c106065 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106066 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106067 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106068 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106069 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106070 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1736_0 := arithEq_of_rows h (row := 1736) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_0
  simp only [← c106065, ← c106066, ← c106067] at e_1736_0
  have e_1736_1 := arithEq_of_rows h (row := 1736) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_1
  simp only [← c106068, ← c106069, k2, ← c106070] at e_1736_1
  have hr := e_1736_1
  simp only [e_1736_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1736 15) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1783)) := by
  have c106071 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106072 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106073 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106074 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106075 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106076 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1736_2 := arithEq_of_rows h (row := 1736) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_2
  simp only [← c106071, ← c106072, ← c106073] at e_1736_2
  have e_1736_3 := arithEq_of_rows h (row := 1736) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_3
  simp only [← c106074, ← c106075, k2, ← c106076] at e_1736_3
  have hr := e_1736_3
  simp only [e_1736_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16954 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1736 23) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1784)) := by
  have c106077 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106078 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106079 := (publicBatchWrapper86_copies3314 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106080 := (publicBatchWrapper86_copies3315 a h).1
  have c106081 := (publicBatchWrapper86_copies3315 a h).2.1
  have c106082 := (publicBatchWrapper86_copies3315 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1736_4 := arithEq_of_rows h (row := 1736) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_4
  simp only [← c106077, ← c106078, ← c106079] at e_1736_4
  have e_1736_5 := arithEq_of_rows h (row := 1736) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_5
  simp only [← c106080, ← c106081, k2, ← c106082] at e_1736_5
  have hr := e_1736_5
  simp only [e_1736_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16955 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1736 31) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1785)) := by
  have c106083 := (publicBatchWrapper86_copies3315 a h).2.2.2.1
  have c106084 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.1
  have c106085 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.1
  have c106086 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.1
  have c106087 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.1
  have c106088 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1736_6 := arithEq_of_rows h (row := 1736) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_6
  simp only [← c106083, ← c106084, ← c106085] at e_1736_6
  have e_1736_7 := arithEq_of_rows h (row := 1736) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_7
  simp only [← c106086, ← c106087, k2, ← c106088] at e_1736_7
  have hr := e_1736_7
  simp only [e_1736_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16956 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1736 39) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1786)) := by
  have c106089 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.1
  have c106090 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106091 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106092 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106093 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106094 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1736_8 := arithEq_of_rows h (row := 1736) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_8
  simp only [← c106089, ← c106090, ← c106091] at e_1736_8
  have e_1736_9 := arithEq_of_rows h (row := 1736) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_9
  simp only [← c106092, ← c106093, k2, ← c106094] at e_1736_9
  have hr := e_1736_9
  simp only [e_1736_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16957 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1736 47) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1787)) := by
  have c106095 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106096 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106097 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106098 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106099 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106100 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1736_10 := arithEq_of_rows h (row := 1736) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_10
  simp only [← c106095, ← c106096, ← c106097] at e_1736_10
  have e_1736_11 := arithEq_of_rows h (row := 1736) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_11
  simp only [← c106098, ← c106099, k2, ← c106100] at e_1736_11
  have hr := e_1736_11
  simp only [e_1736_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16958 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1736 55) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1788)) := by
  have c106101 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106102 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106103 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106104 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106105 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106106 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1736_12 := arithEq_of_rows h (row := 1736) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_12
  simp only [← c106101, ← c106102, ← c106103] at e_1736_12
  have e_1736_13 := arithEq_of_rows h (row := 1736) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_13
  simp only [← c106104, ← c106105, k2, ← c106106] at e_1736_13
  have hr := e_1736_13
  simp only [e_1736_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16959 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1736 63) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1789)) := by
  have c106107 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106108 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106109 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106110 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106111 := (publicBatchWrapper86_copies3315 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106112 := (publicBatchWrapper86_copies3316 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1736_14 := arithEq_of_rows h (row := 1736) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_14
  simp only [← c106107, ← c106108, ← c106109] at e_1736_14
  have e_1736_15 := arithEq_of_rows h (row := 1736) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_15
  simp only [← c106110, ← c106111, k2, ← c106112] at e_1736_15
  have hr := e_1736_15
  simp only [e_1736_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16960 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1736 71) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1790)) := by
  have c106113 := (publicBatchWrapper86_copies3316 a h).2.1
  have c106114 := (publicBatchWrapper86_copies3316 a h).2.2.1
  have c106115 := (publicBatchWrapper86_copies3316 a h).2.2.2.1
  have c106116 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.1
  have c106117 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.1
  have c106118 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1736_16 := arithEq_of_rows h (row := 1736) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_16
  simp only [← c106113, ← c106114, ← c106115] at e_1736_16
  have e_1736_17 := arithEq_of_rows h (row := 1736) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_17
  simp only [← c106116, ← c106117, k2, ← c106118] at e_1736_17
  have hr := e_1736_17
  simp only [e_1736_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16961 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1736 79) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1791)) := by
  have c106119 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.1
  have c106120 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.1
  have c106121 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.1
  have c106122 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106123 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106124 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1736_18 := arithEq_of_rows h (row := 1736) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_18
  simp only [← c106119, ← c106120, ← c106121] at e_1736_18
  have e_1736_19 := arithEq_of_rows h (row := 1736) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1736_19
  simp only [← c106122, ← c106123, k2, ← c106124] at e_1736_19
  have hr := e_1736_19
  simp only [e_1736_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16962 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1737 7) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1792)) := by
  have c106125 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106126 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106127 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106128 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106129 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106130 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1737_0 := arithEq_of_rows h (row := 1737) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_0
  simp only [← c106125, ← c106126, ← c106127] at e_1737_0
  have e_1737_1 := arithEq_of_rows h (row := 1737) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_1
  simp only [← c106128, ← c106129, k2, ← c106130] at e_1737_1
  have hr := e_1737_1
  simp only [e_1737_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16963 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1737 15) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1793)) := by
  have c106131 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106132 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106133 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106134 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106135 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106136 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1737_2 := arithEq_of_rows h (row := 1737) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_2
  simp only [← c106131, ← c106132, ← c106133] at e_1737_2
  have e_1737_3 := arithEq_of_rows h (row := 1737) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_3
  simp only [← c106134, ← c106135, k2, ← c106136] at e_1737_3
  have hr := e_1737_3
  simp only [e_1737_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16964 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1737 23) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1794)) := by
  have c106137 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106138 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106139 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106140 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106141 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106142 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1737_4 := arithEq_of_rows h (row := 1737) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_4
  simp only [← c106137, ← c106138, ← c106139] at e_1737_4
  have e_1737_5 := arithEq_of_rows h (row := 1737) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_5
  simp only [← c106140, ← c106141, k2, ← c106142] at e_1737_5
  have hr := e_1737_5
  simp only [e_1737_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16965 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1737 31) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1795)) := by
  have c106143 := (publicBatchWrapper86_copies3316 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106144 := (publicBatchWrapper86_copies3317 a h).1
  have c106145 := (publicBatchWrapper86_copies3317 a h).2.1
  have c106146 := (publicBatchWrapper86_copies3317 a h).2.2.1
  have c106147 := (publicBatchWrapper86_copies3317 a h).2.2.2.1
  have c106148 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1737_6 := arithEq_of_rows h (row := 1737) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_6
  simp only [← c106143, ← c106144, ← c106145] at e_1737_6
  have e_1737_7 := arithEq_of_rows h (row := 1737) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_7
  simp only [← c106146, ← c106147, k2, ← c106148] at e_1737_7
  have hr := e_1737_7
  simp only [e_1737_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16966 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1737 39) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1796)) := by
  have c106149 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.1
  have c106150 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.1
  have c106151 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.1
  have c106152 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.1
  have c106153 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.1
  have c106154 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1737_8 := arithEq_of_rows h (row := 1737) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_8
  simp only [← c106149, ← c106150, ← c106151] at e_1737_8
  have e_1737_9 := arithEq_of_rows h (row := 1737) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_9
  simp only [← c106152, ← c106153, k2, ← c106154] at e_1737_9
  have hr := e_1737_9
  simp only [e_1737_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16967 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1737 47) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1797)) := by
  have c106155 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106156 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106157 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106158 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106159 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106160 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1737_10 := arithEq_of_rows h (row := 1737) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_10
  simp only [← c106155, ← c106156, ← c106157] at e_1737_10
  have e_1737_11 := arithEq_of_rows h (row := 1737) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_11
  simp only [← c106158, ← c106159, k2, ← c106160] at e_1737_11
  have hr := e_1737_11
  simp only [e_1737_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16968 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1737 55) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1798)) := by
  have c106161 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106162 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106163 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106164 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106165 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106166 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1737_12 := arithEq_of_rows h (row := 1737) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_12
  simp only [← c106161, ← c106162, ← c106163] at e_1737_12
  have e_1737_13 := arithEq_of_rows h (row := 1737) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_13
  simp only [← c106164, ← c106165, k2, ← c106166] at e_1737_13
  have hr := e_1737_13
  simp only [e_1737_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16969 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1737 63) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1799)) := by
  have c106167 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106168 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106169 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106170 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106171 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106172 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1737_14 := arithEq_of_rows h (row := 1737) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_14
  simp only [← c106167, ← c106168, ← c106169] at e_1737_14
  have e_1737_15 := arithEq_of_rows h (row := 1737) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_15
  simp only [← c106170, ← c106171, k2, ← c106172] at e_1737_15
  have hr := e_1737_15
  simp only [e_1737_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16970 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1737 71) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1800)) := by
  have c106173 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106174 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106175 := (publicBatchWrapper86_copies3317 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106176 := (publicBatchWrapper86_copies3318 a h).1
  have c106177 := (publicBatchWrapper86_copies3318 a h).2.1
  have c106178 := (publicBatchWrapper86_copies3318 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1737_16 := arithEq_of_rows h (row := 1737) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_16
  simp only [← c106173, ← c106174, ← c106175] at e_1737_16
  have e_1737_17 := arithEq_of_rows h (row := 1737) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_17
  simp only [← c106176, ← c106177, k2, ← c106178] at e_1737_17
  have hr := e_1737_17
  simp only [e_1737_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16971 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1737 79) = bselect (a (.wire 10 59)) (a (.virt 200)) (a (.virt 1801)) := by
  have c106179 := (publicBatchWrapper86_copies3318 a h).2.2.2.1
  have c106180 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.1
  have c106181 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.1
  have c106182 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.1
  have c106183 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.1
  have c106184 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1737_18 := arithEq_of_rows h (row := 1737) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_18
  simp only [← c106179, ← c106180, ← c106181] at e_1737_18
  have e_1737_19 := arithEq_of_rows h (row := 1737) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1737_19
  simp only [← c106182, ← c106183, k2, ← c106184] at e_1737_19
  have hr := e_1737_19
  simp only [e_1737_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16972 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1738 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1970)) := by
  have c106185 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.1
  have c106186 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106187 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106188 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106189 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106190 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1738_0 := arithEq_of_rows h (row := 1738) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_0
  simp only [← c106185, ← c106186, ← c106187] at e_1738_0
  have e_1738_1 := arithEq_of_rows h (row := 1738) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_1
  simp only [← c106188, ← c106189, k2, ← c106190] at e_1738_1
  have hr := e_1738_1
  simp only [e_1738_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16973 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1738 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1971)) := by
  have c106191 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106192 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106193 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106194 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106195 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106196 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1738_2 := arithEq_of_rows h (row := 1738) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_2
  simp only [← c106191, ← c106192, ← c106193] at e_1738_2
  have e_1738_3 := arithEq_of_rows h (row := 1738) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_3
  simp only [← c106194, ← c106195, k2, ← c106196] at e_1738_3
  have hr := e_1738_3
  simp only [e_1738_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1738 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1972)) := by
  have c106197 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106198 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106199 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106200 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106201 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106202 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1738_4 := arithEq_of_rows h (row := 1738) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_4
  simp only [← c106197, ← c106198, ← c106199] at e_1738_4
  have e_1738_5 := arithEq_of_rows h (row := 1738) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_5
  simp only [← c106200, ← c106201, k2, ← c106202] at e_1738_5
  have hr := e_1738_5
  simp only [e_1738_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1738 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1973)) := by
  have c106203 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106204 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106205 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106206 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106207 := (publicBatchWrapper86_copies3318 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106208 := (publicBatchWrapper86_copies3319 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1738_6 := arithEq_of_rows h (row := 1738) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_6
  simp only [← c106203, ← c106204, ← c106205] at e_1738_6
  have e_1738_7 := arithEq_of_rows h (row := 1738) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_7
  simp only [← c106206, ← c106207, k2, ← c106208] at e_1738_7
  have hr := e_1738_7
  simp only [e_1738_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1738 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1974)) := by
  have c106209 := (publicBatchWrapper86_copies3319 a h).2.1
  have c106210 := (publicBatchWrapper86_copies3319 a h).2.2.1
  have c106211 := (publicBatchWrapper86_copies3319 a h).2.2.2.1
  have c106212 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.1
  have c106213 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.1
  have c106214 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1738_8 := arithEq_of_rows h (row := 1738) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_8
  simp only [← c106209, ← c106210, ← c106211] at e_1738_8
  have e_1738_9 := arithEq_of_rows h (row := 1738) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_9
  simp only [← c106212, ← c106213, k2, ← c106214] at e_1738_9
  have hr := e_1738_9
  simp only [e_1738_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1738 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1975)) := by
  have c106215 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.1
  have c106216 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.1
  have c106217 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.1
  have c106218 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106219 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106220 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1738_10 := arithEq_of_rows h (row := 1738) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_10
  simp only [← c106215, ← c106216, ← c106217] at e_1738_10
  have e_1738_11 := arithEq_of_rows h (row := 1738) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_11
  simp only [← c106218, ← c106219, k2, ← c106220] at e_1738_11
  have hr := e_1738_11
  simp only [e_1738_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1738 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1976)) := by
  have c106221 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106222 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106223 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106224 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106225 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106226 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1738_12 := arithEq_of_rows h (row := 1738) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_12
  simp only [← c106221, ← c106222, ← c106223] at e_1738_12
  have e_1738_13 := arithEq_of_rows h (row := 1738) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_13
  simp only [← c106224, ← c106225, k2, ← c106226] at e_1738_13
  have hr := e_1738_13
  simp only [e_1738_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1738 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1977)) := by
  have c106227 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106228 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106229 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106230 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106231 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106232 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1738_14 := arithEq_of_rows h (row := 1738) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_14
  simp only [← c106227, ← c106228, ← c106229] at e_1738_14
  have e_1738_15 := arithEq_of_rows h (row := 1738) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_15
  simp only [← c106230, ← c106231, k2, ← c106232] at e_1738_15
  have hr := e_1738_15
  simp only [e_1738_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1738 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1978)) := by
  have c106233 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106234 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106235 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106236 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106237 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106238 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1738_16 := arithEq_of_rows h (row := 1738) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_16
  simp only [← c106233, ← c106234, ← c106235] at e_1738_16
  have e_1738_17 := arithEq_of_rows h (row := 1738) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_17
  simp only [← c106236, ← c106237, k2, ← c106238] at e_1738_17
  have hr := e_1738_17
  simp only [e_1738_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1738 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1979)) := by
  have c106239 := (publicBatchWrapper86_copies3319 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106240 := (publicBatchWrapper86_copies3320 a h).1
  have c106241 := (publicBatchWrapper86_copies3320 a h).2.1
  have c106242 := (publicBatchWrapper86_copies3320 a h).2.2.1
  have c106243 := (publicBatchWrapper86_copies3320 a h).2.2.2.1
  have c106244 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1738_18 := arithEq_of_rows h (row := 1738) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_18
  simp only [← c106239, ← c106240, ← c106241] at e_1738_18
  have e_1738_19 := arithEq_of_rows h (row := 1738) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1738_19
  simp only [← c106242, ← c106243, k2, ← c106244] at e_1738_19
  have hr := e_1738_19
  simp only [e_1738_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1739 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1980)) := by
  have c106245 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.1
  have c106246 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.1
  have c106247 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.1
  have c106248 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.1
  have c106249 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.1
  have c106250 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1739_0 := arithEq_of_rows h (row := 1739) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_0
  simp only [← c106245, ← c106246, ← c106247] at e_1739_0
  have e_1739_1 := arithEq_of_rows h (row := 1739) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_1
  simp only [← c106248, ← c106249, k2, ← c106250] at e_1739_1
  have hr := e_1739_1
  simp only [e_1739_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1739 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1981)) := by
  have c106251 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106252 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106253 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106254 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106255 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106256 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1739_2 := arithEq_of_rows h (row := 1739) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_2
  simp only [← c106251, ← c106252, ← c106253] at e_1739_2
  have e_1739_3 := arithEq_of_rows h (row := 1739) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_3
  simp only [← c106254, ← c106255, k2, ← c106256] at e_1739_3
  have hr := e_1739_3
  simp only [e_1739_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1739 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1982)) := by
  have c106257 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106258 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106259 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106260 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106261 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106262 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1739_4 := arithEq_of_rows h (row := 1739) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_4
  simp only [← c106257, ← c106258, ← c106259] at e_1739_4
  have e_1739_5 := arithEq_of_rows h (row := 1739) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_5
  simp only [← c106260, ← c106261, k2, ← c106262] at e_1739_5
  have hr := e_1739_5
  simp only [e_1739_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1739 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1983)) := by
  have c106263 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106264 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106265 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106266 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106267 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106268 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1739_6 := arithEq_of_rows h (row := 1739) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_6
  simp only [← c106263, ← c106264, ← c106265] at e_1739_6
  have e_1739_7 := arithEq_of_rows h (row := 1739) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_7
  simp only [← c106266, ← c106267, k2, ← c106268] at e_1739_7
  have hr := e_1739_7
  simp only [e_1739_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1739 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1984)) := by
  have c106269 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106270 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106271 := (publicBatchWrapper86_copies3320 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106272 := (publicBatchWrapper86_copies3321 a h).1
  have c106273 := (publicBatchWrapper86_copies3321 a h).2.1
  have c106274 := (publicBatchWrapper86_copies3321 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1739_8 := arithEq_of_rows h (row := 1739) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_8
  simp only [← c106269, ← c106270, ← c106271] at e_1739_8
  have e_1739_9 := arithEq_of_rows h (row := 1739) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_9
  simp only [← c106272, ← c106273, k2, ← c106274] at e_1739_9
  have hr := e_1739_9
  simp only [e_1739_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1739 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1985)) := by
  have c106275 := (publicBatchWrapper86_copies3321 a h).2.2.2.1
  have c106276 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.1
  have c106277 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.1
  have c106278 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.1
  have c106279 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.1
  have c106280 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1739_10 := arithEq_of_rows h (row := 1739) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_10
  simp only [← c106275, ← c106276, ← c106277] at e_1739_10
  have e_1739_11 := arithEq_of_rows h (row := 1739) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_11
  simp only [← c106278, ← c106279, k2, ← c106280] at e_1739_11
  have hr := e_1739_11
  simp only [e_1739_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1739 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1986)) := by
  have c106281 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.1
  have c106282 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106283 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106284 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106285 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106286 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1739_12 := arithEq_of_rows h (row := 1739) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_12
  simp only [← c106281, ← c106282, ← c106283] at e_1739_12
  have e_1739_13 := arithEq_of_rows h (row := 1739) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_13
  simp only [← c106284, ← c106285, k2, ← c106286] at e_1739_13
  have hr := e_1739_13
  simp only [e_1739_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1739 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1987)) := by
  have c106287 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106288 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106289 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106290 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106291 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106292 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1739_14 := arithEq_of_rows h (row := 1739) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_14
  simp only [← c106287, ← c106288, ← c106289] at e_1739_14
  have e_1739_15 := arithEq_of_rows h (row := 1739) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_15
  simp only [← c106290, ← c106291, k2, ← c106292] at e_1739_15
  have hr := e_1739_15
  simp only [e_1739_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1739 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1988)) := by
  have c106293 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106294 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106295 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106296 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106297 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106298 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1739_16 := arithEq_of_rows h (row := 1739) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_16
  simp only [← c106293, ← c106294, ← c106295] at e_1739_16
  have e_1739_17 := arithEq_of_rows h (row := 1739) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_17
  simp only [← c106296, ← c106297, k2, ← c106298] at e_1739_17
  have hr := e_1739_17
  simp only [e_1739_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1739 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1989)) := by
  have c106299 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106300 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106301 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106302 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106303 := (publicBatchWrapper86_copies3321 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106304 := (publicBatchWrapper86_copies3322 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1739_18 := arithEq_of_rows h (row := 1739) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_18
  simp only [← c106299, ← c106300, ← c106301] at e_1739_18
  have e_1739_19 := arithEq_of_rows h (row := 1739) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1739_19
  simp only [← c106302, ← c106303, k2, ← c106304] at e_1739_19
  have hr := e_1739_19
  simp only [e_1739_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1740 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1990)) := by
  have c106305 := (publicBatchWrapper86_copies3322 a h).2.1
  have c106306 := (publicBatchWrapper86_copies3322 a h).2.2.1
  have c106307 := (publicBatchWrapper86_copies3322 a h).2.2.2.1
  have c106308 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.1
  have c106309 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.1
  have c106310 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1740_0 := arithEq_of_rows h (row := 1740) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_0
  simp only [← c106305, ← c106306, ← c106307] at e_1740_0
  have e_1740_1 := arithEq_of_rows h (row := 1740) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_1
  simp only [← c106308, ← c106309, k2, ← c106310] at e_1740_1
  have hr := e_1740_1
  simp only [e_1740_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1740 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1991)) := by
  have c106311 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.1
  have c106312 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.1
  have c106313 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.1
  have c106314 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.1
  have c106315 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106316 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1740_2 := arithEq_of_rows h (row := 1740) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_2
  simp only [← c106311, ← c106312, ← c106313] at e_1740_2
  have e_1740_3 := arithEq_of_rows h (row := 1740) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_3
  simp only [← c106314, ← c106315, k2, ← c106316] at e_1740_3
  have hr := e_1740_3
  simp only [e_1740_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1740 23) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1992)) := by
  have c106317 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106318 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106319 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106320 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106321 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106322 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1740_4 := arithEq_of_rows h (row := 1740) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_4
  simp only [← c106317, ← c106318, ← c106319] at e_1740_4
  have e_1740_5 := arithEq_of_rows h (row := 1740) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_5
  simp only [← c106320, ← c106321, k2, ← c106322] at e_1740_5
  have hr := e_1740_5
  simp only [e_1740_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1740 31) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1993)) := by
  have c106323 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106324 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106325 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106326 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106327 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106328 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1740_6 := arithEq_of_rows h (row := 1740) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_6
  simp only [← c106323, ← c106324, ← c106325] at e_1740_6
  have e_1740_7 := arithEq_of_rows h (row := 1740) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_7
  simp only [← c106326, ← c106327, k2, ← c106328] at e_1740_7
  have hr := e_1740_7
  simp only [e_1740_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1740 39) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1994)) := by
  have c106329 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106330 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106331 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106332 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106333 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106334 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1740_8 := arithEq_of_rows h (row := 1740) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_8
  simp only [← c106329, ← c106330, ← c106331] at e_1740_8
  have e_1740_9 := arithEq_of_rows h (row := 1740) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_9
  simp only [← c106332, ← c106333, k2, ← c106334] at e_1740_9
  have hr := e_1740_9
  simp only [e_1740_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1740 47) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1995)) := by
  have c106335 := (publicBatchWrapper86_copies3322 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106336 := (publicBatchWrapper86_copies3323 a h).1
  have c106337 := (publicBatchWrapper86_copies3323 a h).2.1
  have c106338 := (publicBatchWrapper86_copies3323 a h).2.2.1
  have c106339 := (publicBatchWrapper86_copies3323 a h).2.2.2.1
  have c106340 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1740_10 := arithEq_of_rows h (row := 1740) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_10
  simp only [← c106335, ← c106336, ← c106337] at e_1740_10
  have e_1740_11 := arithEq_of_rows h (row := 1740) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_11
  simp only [← c106338, ← c106339, k2, ← c106340] at e_1740_11
  have hr := e_1740_11
  simp only [e_1740_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1740 55) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1996)) := by
  have c106341 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.1
  have c106342 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.1
  have c106343 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.1
  have c106344 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.1
  have c106345 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.1
  have c106346 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1740_12 := arithEq_of_rows h (row := 1740) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_12
  simp only [← c106341, ← c106342, ← c106343] at e_1740_12
  have e_1740_13 := arithEq_of_rows h (row := 1740) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_13
  simp only [← c106344, ← c106345, k2, ← c106346] at e_1740_13
  have hr := e_1740_13
  simp only [e_1740_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f16999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1740 63) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1997)) := by
  have c106347 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c106348 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106349 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106350 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106351 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106352 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1740_14 := arithEq_of_rows h (row := 1740) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_14
  simp only [← c106347, ← c106348, ← c106349] at e_1740_14
  have e_1740_15 := arithEq_of_rows h (row := 1740) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_15
  simp only [← c106350, ← c106351, k2, ← c106352] at e_1740_15
  have hr := e_1740_15
  simp only [e_1740_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1740 71) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1998)) := by
  have c106353 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106354 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106355 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106356 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106357 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106358 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1740_16 := arithEq_of_rows h (row := 1740) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_16
  simp only [← c106353, ← c106354, ← c106355] at e_1740_16
  have e_1740_17 := arithEq_of_rows h (row := 1740) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_17
  simp only [← c106356, ← c106357, k2, ← c106358] at e_1740_17
  have hr := e_1740_17
  simp only [e_1740_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1740 79) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 1999)) := by
  have c106359 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106360 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106361 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106362 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106363 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106364 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1740_18 := arithEq_of_rows h (row := 1740) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_18
  simp only [← c106359, ← c106360, ← c106361] at e_1740_18
  have e_1740_19 := arithEq_of_rows h (row := 1740) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1740_19
  simp only [← c106362, ← c106363, k2, ← c106364] at e_1740_19
  have hr := e_1740_19
  simp only [e_1740_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1741 7) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 2000)) := by
  have c106365 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106366 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c106367 := (publicBatchWrapper86_copies3323 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c106368 := (publicBatchWrapper86_copies3324 a h).1
  have c106369 := (publicBatchWrapper86_copies3324 a h).2.1
  have c106370 := (publicBatchWrapper86_copies3324 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1741_0 := arithEq_of_rows h (row := 1741) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_0
  simp only [← c106365, ← c106366, ← c106367] at e_1741_0
  have e_1741_1 := arithEq_of_rows h (row := 1741) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_1
  simp only [← c106368, ← c106369, k2, ← c106370] at e_1741_1
  have hr := e_1741_1
  simp only [e_1741_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f17003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1741 15) = bselect (a (.wire 11 23)) (a (.virt 200)) (a (.virt 2001)) := by
  have c106371 := (publicBatchWrapper86_copies3324 a h).2.2.2.1
  have c106372 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.1
  have c106373 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.1
  have c106374 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.1
  have c106375 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.1
  have c106376 := (publicBatchWrapper86_copies3324 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1741_2 := arithEq_of_rows h (row := 1741) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_2
  simp only [← c106371, ← c106372, ← c106373] at e_1741_2
  have e_1741_3 := arithEq_of_rows h (row := 1741) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1741_3
  simp only [← c106374, ← c106375, k2, ← c106376] at e_1741_3
  have hr := e_1741_3
  simp only [e_1741_2] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
