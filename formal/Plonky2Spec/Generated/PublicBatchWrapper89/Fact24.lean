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

theorem publicBatchWrapper89_f18543 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1896 23) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 7999)) := by
  have c115764 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115765 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115766 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115767 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115768 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115769 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1896_4 := arithEq_of_rows h (row := 1896) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_4
  simp only [← c115764, ← c115765, ← c115766] at e_1896_4
  have e_1896_5 := arithEq_of_rows h (row := 1896) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_5
  simp only [← c115767, ← c115768, k2, ← c115769] at e_1896_5
  have hr := e_1896_5
  simp only [e_1896_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18544 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1896 31) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 8000)) := by
  have c115770 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115771 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115772 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115773 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115774 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115775 := (publicBatchWrapper89_copies3617 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1896_6 := arithEq_of_rows h (row := 1896) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_6
  simp only [← c115770, ← c115771, ← c115772] at e_1896_6
  have e_1896_7 := arithEq_of_rows h (row := 1896) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_7
  simp only [← c115773, ← c115774, k2, ← c115775] at e_1896_7
  have hr := e_1896_7
  simp only [e_1896_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18545 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1896 39) = bselect (a (.wire 39 15)) (a (.virt 200)) (a (.virt 8001)) := by
  have c115776 := (publicBatchWrapper89_copies3618 a h).1
  have c115777 := (publicBatchWrapper89_copies3618 a h).2.1
  have c115778 := (publicBatchWrapper89_copies3618 a h).2.2.1
  have c115779 := (publicBatchWrapper89_copies3618 a h).2.2.2.1
  have c115780 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.1
  have c115781 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1896_8 := arithEq_of_rows h (row := 1896) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_8
  simp only [← c115776, ← c115777, ← c115778] at e_1896_8
  have e_1896_9 := arithEq_of_rows h (row := 1896) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_9
  simp only [← c115779, ← c115780, k2, ← c115781] at e_1896_9
  have hr := e_1896_9
  simp only [e_1896_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18546 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1896 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8170)) := by
  have c115782 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.1
  have c115783 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.1
  have c115784 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.1
  have c115785 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.1
  have c115786 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115787 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1896_10 := arithEq_of_rows h (row := 1896) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_10
  simp only [← c115782, ← c115783, ← c115784] at e_1896_10
  have e_1896_11 := arithEq_of_rows h (row := 1896) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_11
  simp only [← c115785, ← c115786, k2, ← c115787] at e_1896_11
  have hr := e_1896_11
  simp only [e_1896_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18547 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1896 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8171)) := by
  have c115788 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115789 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115790 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115791 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115792 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115793 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1896_12 := arithEq_of_rows h (row := 1896) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_12
  simp only [← c115788, ← c115789, ← c115790] at e_1896_12
  have e_1896_13 := arithEq_of_rows h (row := 1896) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_13
  simp only [← c115791, ← c115792, k2, ← c115793] at e_1896_13
  have hr := e_1896_13
  simp only [e_1896_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18548 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1896 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8172)) := by
  have c115794 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115795 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115796 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115797 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115798 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115799 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1896_14 := arithEq_of_rows h (row := 1896) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_14
  simp only [← c115794, ← c115795, ← c115796] at e_1896_14
  have e_1896_15 := arithEq_of_rows h (row := 1896) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_15
  simp only [← c115797, ← c115798, k2, ← c115799] at e_1896_15
  have hr := e_1896_15
  simp only [e_1896_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18549 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1896 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8173)) := by
  have c115800 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115801 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115802 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115803 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115804 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115805 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1896_16 := arithEq_of_rows h (row := 1896) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_16
  simp only [← c115800, ← c115801, ← c115802] at e_1896_16
  have e_1896_17 := arithEq_of_rows h (row := 1896) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_17
  simp only [← c115803, ← c115804, k2, ← c115805] at e_1896_17
  have hr := e_1896_17
  simp only [e_1896_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18550 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1896 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8174)) := by
  have c115806 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115807 := (publicBatchWrapper89_copies3618 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115808 := (publicBatchWrapper89_copies3619 a h).1
  have c115809 := (publicBatchWrapper89_copies3619 a h).2.1
  have c115810 := (publicBatchWrapper89_copies3619 a h).2.2.1
  have c115811 := (publicBatchWrapper89_copies3619 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1896_18 := arithEq_of_rows h (row := 1896) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_18
  simp only [← c115806, ← c115807, ← c115808] at e_1896_18
  have e_1896_19 := arithEq_of_rows h (row := 1896) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1896_19
  simp only [← c115809, ← c115810, k2, ← c115811] at e_1896_19
  have hr := e_1896_19
  simp only [e_1896_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18551 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1897 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8175)) := by
  have c115812 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.1
  have c115813 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.1
  have c115814 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.1
  have c115815 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.1
  have c115816 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.1
  have c115817 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1897_0 := arithEq_of_rows h (row := 1897) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_0
  simp only [← c115812, ← c115813, ← c115814] at e_1897_0
  have e_1897_1 := arithEq_of_rows h (row := 1897) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_1
  simp only [← c115815, ← c115816, k2, ← c115817] at e_1897_1
  have hr := e_1897_1
  simp only [e_1897_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18552 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1897 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8176)) := by
  have c115818 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115819 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115820 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115821 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115822 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115823 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1897_2 := arithEq_of_rows h (row := 1897) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_2
  simp only [← c115818, ← c115819, ← c115820] at e_1897_2
  have e_1897_3 := arithEq_of_rows h (row := 1897) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_3
  simp only [← c115821, ← c115822, k2, ← c115823] at e_1897_3
  have hr := e_1897_3
  simp only [e_1897_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18553 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1897 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8177)) := by
  have c115824 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115825 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115826 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115827 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115828 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115829 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1897_4 := arithEq_of_rows h (row := 1897) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_4
  simp only [← c115824, ← c115825, ← c115826] at e_1897_4
  have e_1897_5 := arithEq_of_rows h (row := 1897) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_5
  simp only [← c115827, ← c115828, k2, ← c115829] at e_1897_5
  have hr := e_1897_5
  simp only [e_1897_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18554 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1897 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8178)) := by
  have c115830 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115831 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115832 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115833 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115834 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115835 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1897_6 := arithEq_of_rows h (row := 1897) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_6
  simp only [← c115830, ← c115831, ← c115832] at e_1897_6
  have e_1897_7 := arithEq_of_rows h (row := 1897) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_7
  simp only [← c115833, ← c115834, k2, ← c115835] at e_1897_7
  have hr := e_1897_7
  simp only [e_1897_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18555 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1897 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8179)) := by
  have c115836 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115837 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115838 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115839 := (publicBatchWrapper89_copies3619 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115840 := (publicBatchWrapper89_copies3620 a h).1
  have c115841 := (publicBatchWrapper89_copies3620 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1897_8 := arithEq_of_rows h (row := 1897) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_8
  simp only [← c115836, ← c115837, ← c115838] at e_1897_8
  have e_1897_9 := arithEq_of_rows h (row := 1897) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_9
  simp only [← c115839, ← c115840, k2, ← c115841] at e_1897_9
  have hr := e_1897_9
  simp only [e_1897_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18556 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1897 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8180)) := by
  have c115842 := (publicBatchWrapper89_copies3620 a h).2.2.1
  have c115843 := (publicBatchWrapper89_copies3620 a h).2.2.2.1
  have c115844 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.1
  have c115845 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.1
  have c115846 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.1
  have c115847 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1897_10 := arithEq_of_rows h (row := 1897) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_10
  simp only [← c115842, ← c115843, ← c115844] at e_1897_10
  have e_1897_11 := arithEq_of_rows h (row := 1897) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_11
  simp only [← c115845, ← c115846, k2, ← c115847] at e_1897_11
  have hr := e_1897_11
  simp only [e_1897_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18557 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1897 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8181)) := by
  have c115848 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.1
  have c115849 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.1
  have c115850 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115851 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115852 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115853 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1897_12 := arithEq_of_rows h (row := 1897) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_12
  simp only [← c115848, ← c115849, ← c115850] at e_1897_12
  have e_1897_13 := arithEq_of_rows h (row := 1897) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_13
  simp only [← c115851, ← c115852, k2, ← c115853] at e_1897_13
  have hr := e_1897_13
  simp only [e_1897_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18558 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1897 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8182)) := by
  have c115854 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115855 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115856 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115857 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115858 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115859 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1897_14 := arithEq_of_rows h (row := 1897) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_14
  simp only [← c115854, ← c115855, ← c115856] at e_1897_14
  have e_1897_15 := arithEq_of_rows h (row := 1897) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_15
  simp only [← c115857, ← c115858, k2, ← c115859] at e_1897_15
  have hr := e_1897_15
  simp only [e_1897_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18559 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1897 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8183)) := by
  have c115860 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115861 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115862 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115863 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115864 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115865 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1897_16 := arithEq_of_rows h (row := 1897) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_16
  simp only [← c115860, ← c115861, ← c115862] at e_1897_16
  have e_1897_17 := arithEq_of_rows h (row := 1897) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_17
  simp only [← c115863, ← c115864, k2, ← c115865] at e_1897_17
  have hr := e_1897_17
  simp only [e_1897_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18560 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1897 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8184)) := by
  have c115866 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115867 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115868 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115869 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115870 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115871 := (publicBatchWrapper89_copies3620 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1897_18 := arithEq_of_rows h (row := 1897) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_18
  simp only [← c115866, ← c115867, ← c115868] at e_1897_18
  have e_1897_19 := arithEq_of_rows h (row := 1897) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1897_19
  simp only [← c115869, ← c115870, k2, ← c115871] at e_1897_19
  have hr := e_1897_19
  simp only [e_1897_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18561 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1898 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8185)) := by
  have c115872 := (publicBatchWrapper89_copies3621 a h).1
  have c115873 := (publicBatchWrapper89_copies3621 a h).2.1
  have c115874 := (publicBatchWrapper89_copies3621 a h).2.2.1
  have c115875 := (publicBatchWrapper89_copies3621 a h).2.2.2.1
  have c115876 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.1
  have c115877 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1898_0 := arithEq_of_rows h (row := 1898) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_0
  simp only [← c115872, ← c115873, ← c115874] at e_1898_0
  have e_1898_1 := arithEq_of_rows h (row := 1898) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_1
  simp only [← c115875, ← c115876, k2, ← c115877] at e_1898_1
  have hr := e_1898_1
  simp only [e_1898_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18562 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1898 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8186)) := by
  have c115878 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.1
  have c115879 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.1
  have c115880 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.1
  have c115881 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.1
  have c115882 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115883 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1898_2 := arithEq_of_rows h (row := 1898) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_2
  simp only [← c115878, ← c115879, ← c115880] at e_1898_2
  have e_1898_3 := arithEq_of_rows h (row := 1898) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_3
  simp only [← c115881, ← c115882, k2, ← c115883] at e_1898_3
  have hr := e_1898_3
  simp only [e_1898_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18563 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1898 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8187)) := by
  have c115884 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115885 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115886 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115887 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115888 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115889 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1898_4 := arithEq_of_rows h (row := 1898) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_4
  simp only [← c115884, ← c115885, ← c115886] at e_1898_4
  have e_1898_5 := arithEq_of_rows h (row := 1898) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_5
  simp only [← c115887, ← c115888, k2, ← c115889] at e_1898_5
  have hr := e_1898_5
  simp only [e_1898_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18564 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1898 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8188)) := by
  have c115890 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115891 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115892 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115893 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115894 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115895 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1898_6 := arithEq_of_rows h (row := 1898) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_6
  simp only [← c115890, ← c115891, ← c115892] at e_1898_6
  have e_1898_7 := arithEq_of_rows h (row := 1898) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_7
  simp only [← c115893, ← c115894, k2, ← c115895] at e_1898_7
  have hr := e_1898_7
  simp only [e_1898_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18565 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1898 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8189)) := by
  have c115896 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115897 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115898 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115899 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115900 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115901 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1898_8 := arithEq_of_rows h (row := 1898) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_8
  simp only [← c115896, ← c115897, ← c115898] at e_1898_8
  have e_1898_9 := arithEq_of_rows h (row := 1898) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_9
  simp only [← c115899, ← c115900, k2, ← c115901] at e_1898_9
  have hr := e_1898_9
  simp only [e_1898_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18566 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1898 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8190)) := by
  have c115902 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115903 := (publicBatchWrapper89_copies3621 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115904 := (publicBatchWrapper89_copies3622 a h).1
  have c115905 := (publicBatchWrapper89_copies3622 a h).2.1
  have c115906 := (publicBatchWrapper89_copies3622 a h).2.2.1
  have c115907 := (publicBatchWrapper89_copies3622 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1898_10 := arithEq_of_rows h (row := 1898) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_10
  simp only [← c115902, ← c115903, ← c115904] at e_1898_10
  have e_1898_11 := arithEq_of_rows h (row := 1898) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_11
  simp only [← c115905, ← c115906, k2, ← c115907] at e_1898_11
  have hr := e_1898_11
  simp only [e_1898_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18567 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1898 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8191)) := by
  have c115908 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.1
  have c115909 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.1
  have c115910 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.1
  have c115911 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.1
  have c115912 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.1
  have c115913 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1898_12 := arithEq_of_rows h (row := 1898) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_12
  simp only [← c115908, ← c115909, ← c115910] at e_1898_12
  have e_1898_13 := arithEq_of_rows h (row := 1898) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_13
  simp only [← c115911, ← c115912, k2, ← c115913] at e_1898_13
  have hr := e_1898_13
  simp only [e_1898_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18568 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1898 63) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8192)) := by
  have c115914 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115915 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115916 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115917 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115918 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115919 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1898_14 := arithEq_of_rows h (row := 1898) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_14
  simp only [← c115914, ← c115915, ← c115916] at e_1898_14
  have e_1898_15 := arithEq_of_rows h (row := 1898) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_15
  simp only [← c115917, ← c115918, k2, ← c115919] at e_1898_15
  have hr := e_1898_15
  simp only [e_1898_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18569 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1898 71) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8193)) := by
  have c115920 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115921 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115922 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115923 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115924 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115925 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1898_16 := arithEq_of_rows h (row := 1898) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_16
  simp only [← c115920, ← c115921, ← c115922] at e_1898_16
  have e_1898_17 := arithEq_of_rows h (row := 1898) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_17
  simp only [← c115923, ← c115924, k2, ← c115925] at e_1898_17
  have hr := e_1898_17
  simp only [e_1898_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18570 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1898 79) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8194)) := by
  have c115926 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115927 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115928 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115929 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115930 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115931 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1898_18 := arithEq_of_rows h (row := 1898) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_18
  simp only [← c115926, ← c115927, ← c115928] at e_1898_18
  have e_1898_19 := arithEq_of_rows h (row := 1898) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1898_19
  simp only [← c115929, ← c115930, k2, ← c115931] at e_1898_19
  have hr := e_1898_19
  simp only [e_1898_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18571 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1899 7) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8195)) := by
  have c115932 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115933 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115934 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115935 := (publicBatchWrapper89_copies3622 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c115936 := (publicBatchWrapper89_copies3623 a h).1
  have c115937 := (publicBatchWrapper89_copies3623 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1899_0 := arithEq_of_rows h (row := 1899) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_0
  simp only [← c115932, ← c115933, ← c115934] at e_1899_0
  have e_1899_1 := arithEq_of_rows h (row := 1899) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_1
  simp only [← c115935, ← c115936, k2, ← c115937] at e_1899_1
  have hr := e_1899_1
  simp only [e_1899_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18572 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1899 15) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8196)) := by
  have c115938 := (publicBatchWrapper89_copies3623 a h).2.2.1
  have c115939 := (publicBatchWrapper89_copies3623 a h).2.2.2.1
  have c115940 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.1
  have c115941 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.1
  have c115942 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.1
  have c115943 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1899_2 := arithEq_of_rows h (row := 1899) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_2
  simp only [← c115938, ← c115939, ← c115940] at e_1899_2
  have e_1899_3 := arithEq_of_rows h (row := 1899) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_3
  simp only [← c115941, ← c115942, k2, ← c115943] at e_1899_3
  have hr := e_1899_3
  simp only [e_1899_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18573 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1899 23) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8197)) := by
  have c115944 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.1
  have c115945 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.1
  have c115946 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115947 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c115948 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115949 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1899_4 := arithEq_of_rows h (row := 1899) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_4
  simp only [← c115944, ← c115945, ← c115946] at e_1899_4
  have e_1899_5 := arithEq_of_rows h (row := 1899) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_5
  simp only [← c115947, ← c115948, k2, ← c115949] at e_1899_5
  have hr := e_1899_5
  simp only [e_1899_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18574 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1899 31) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8198)) := by
  have c115950 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115951 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115952 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115953 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115954 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115955 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1899_6 := arithEq_of_rows h (row := 1899) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_6
  simp only [← c115950, ← c115951, ← c115952] at e_1899_6
  have e_1899_7 := arithEq_of_rows h (row := 1899) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_7
  simp only [← c115953, ← c115954, k2, ← c115955] at e_1899_7
  have hr := e_1899_7
  simp only [e_1899_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18575 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1899 39) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8199)) := by
  have c115956 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115957 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115958 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115959 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115960 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115961 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1899_8 := arithEq_of_rows h (row := 1899) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_8
  simp only [← c115956, ← c115957, ← c115958] at e_1899_8
  have e_1899_9 := arithEq_of_rows h (row := 1899) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_9
  simp only [← c115959, ← c115960, k2, ← c115961] at e_1899_9
  have hr := e_1899_9
  simp only [e_1899_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18576 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1899 47) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8200)) := by
  have c115962 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115963 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115964 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115965 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115966 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115967 := (publicBatchWrapper89_copies3623 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1899_10 := arithEq_of_rows h (row := 1899) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_10
  simp only [← c115962, ← c115963, ← c115964] at e_1899_10
  have e_1899_11 := arithEq_of_rows h (row := 1899) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_11
  simp only [← c115965, ← c115966, k2, ← c115967] at e_1899_11
  have hr := e_1899_11
  simp only [e_1899_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18577 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1899 55) = bselect (a (.wire 39 59)) (a (.virt 200)) (a (.virt 8201)) := by
  have c115968 := (publicBatchWrapper89_copies3624 a h).1
  have c115969 := (publicBatchWrapper89_copies3624 a h).2.1
  have c115970 := (publicBatchWrapper89_copies3624 a h).2.2.1
  have c115971 := (publicBatchWrapper89_copies3624 a h).2.2.2.1
  have c115972 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.1
  have c115973 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1899_12 := arithEq_of_rows h (row := 1899) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_12
  simp only [← c115968, ← c115969, ← c115970] at e_1899_12
  have e_1899_13 := arithEq_of_rows h (row := 1899) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_13
  simp only [← c115971, ← c115972, k2, ← c115973] at e_1899_13
  have hr := e_1899_13
  simp only [e_1899_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18578 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1899 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8370)) := by
  have c115974 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.1
  have c115975 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.1
  have c115976 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.1
  have c115977 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.1
  have c115978 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.1
  have c115979 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1899_14 := arithEq_of_rows h (row := 1899) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_14
  simp only [← c115974, ← c115975, ← c115976] at e_1899_14
  have e_1899_15 := arithEq_of_rows h (row := 1899) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_15
  simp only [← c115977, ← c115978, k2, ← c115979] at e_1899_15
  have hr := e_1899_15
  simp only [e_1899_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18579 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1899 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8371)) := by
  have c115980 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115981 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115982 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115983 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115984 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115985 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1899_16 := arithEq_of_rows h (row := 1899) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_16
  simp only [← c115980, ← c115981, ← c115982] at e_1899_16
  have e_1899_17 := arithEq_of_rows h (row := 1899) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_17
  simp only [← c115983, ← c115984, k2, ← c115985] at e_1899_17
  have hr := e_1899_17
  simp only [e_1899_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18580 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1899 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8372)) := by
  have c115986 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115987 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115988 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115989 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115990 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115991 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1899_18 := arithEq_of_rows h (row := 1899) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_18
  simp only [← c115986, ← c115987, ← c115988] at e_1899_18
  have e_1899_19 := arithEq_of_rows h (row := 1899) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1899_19
  simp only [← c115989, ← c115990, k2, ← c115991] at e_1899_19
  have hr := e_1899_19
  simp only [e_1899_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18581 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1900 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8373)) := by
  have c115992 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115993 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115994 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115995 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115996 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115997 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1900_0 := arithEq_of_rows h (row := 1900) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_0
  simp only [← c115992, ← c115993, ← c115994] at e_1900_0
  have e_1900_1 := arithEq_of_rows h (row := 1900) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_1
  simp only [← c115995, ← c115996, k2, ← c115997] at e_1900_1
  have hr := e_1900_1
  simp only [e_1900_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18582 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1900 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8374)) := by
  have c115998 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c115999 := (publicBatchWrapper89_copies3624 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116000 := (publicBatchWrapper89_copies3625 a h).1
  have c116001 := (publicBatchWrapper89_copies3625 a h).2.1
  have c116002 := (publicBatchWrapper89_copies3625 a h).2.2.1
  have c116003 := (publicBatchWrapper89_copies3625 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1900_2 := arithEq_of_rows h (row := 1900) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_2
  simp only [← c115998, ← c115999, ← c116000] at e_1900_2
  have e_1900_3 := arithEq_of_rows h (row := 1900) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_3
  simp only [← c116001, ← c116002, k2, ← c116003] at e_1900_3
  have hr := e_1900_3
  simp only [e_1900_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18583 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1900 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8375)) := by
  have c116004 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.1
  have c116005 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.1
  have c116006 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.1
  have c116007 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.1
  have c116008 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.1
  have c116009 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1900_4 := arithEq_of_rows h (row := 1900) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_4
  simp only [← c116004, ← c116005, ← c116006] at e_1900_4
  have e_1900_5 := arithEq_of_rows h (row := 1900) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_5
  simp only [← c116007, ← c116008, k2, ← c116009] at e_1900_5
  have hr := e_1900_5
  simp only [e_1900_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18584 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1900 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8376)) := by
  have c116010 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116011 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116012 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116013 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116014 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116015 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1900_6 := arithEq_of_rows h (row := 1900) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_6
  simp only [← c116010, ← c116011, ← c116012] at e_1900_6
  have e_1900_7 := arithEq_of_rows h (row := 1900) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_7
  simp only [← c116013, ← c116014, k2, ← c116015] at e_1900_7
  have hr := e_1900_7
  simp only [e_1900_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18585 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1900 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8377)) := by
  have c116016 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116017 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116018 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116019 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116020 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116021 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1900_8 := arithEq_of_rows h (row := 1900) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_8
  simp only [← c116016, ← c116017, ← c116018] at e_1900_8
  have e_1900_9 := arithEq_of_rows h (row := 1900) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_9
  simp only [← c116019, ← c116020, k2, ← c116021] at e_1900_9
  have hr := e_1900_9
  simp only [e_1900_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18586 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1900 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8378)) := by
  have c116022 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116023 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116024 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116025 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116026 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116027 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1900_10 := arithEq_of_rows h (row := 1900) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_10
  simp only [← c116022, ← c116023, ← c116024] at e_1900_10
  have e_1900_11 := arithEq_of_rows h (row := 1900) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_11
  simp only [← c116025, ← c116026, k2, ← c116027] at e_1900_11
  have hr := e_1900_11
  simp only [e_1900_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18587 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1900 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8379)) := by
  have c116028 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116029 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116030 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116031 := (publicBatchWrapper89_copies3625 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116032 := (publicBatchWrapper89_copies3626 a h).1
  have c116033 := (publicBatchWrapper89_copies3626 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1900_12 := arithEq_of_rows h (row := 1900) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_12
  simp only [← c116028, ← c116029, ← c116030] at e_1900_12
  have e_1900_13 := arithEq_of_rows h (row := 1900) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_13
  simp only [← c116031, ← c116032, k2, ← c116033] at e_1900_13
  have hr := e_1900_13
  simp only [e_1900_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18588 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1900 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8380)) := by
  have c116034 := (publicBatchWrapper89_copies3626 a h).2.2.1
  have c116035 := (publicBatchWrapper89_copies3626 a h).2.2.2.1
  have c116036 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.1
  have c116037 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.1
  have c116038 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.1
  have c116039 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1900_14 := arithEq_of_rows h (row := 1900) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_14
  simp only [← c116034, ← c116035, ← c116036] at e_1900_14
  have e_1900_15 := arithEq_of_rows h (row := 1900) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_15
  simp only [← c116037, ← c116038, k2, ← c116039] at e_1900_15
  have hr := e_1900_15
  simp only [e_1900_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18589 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1900 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8381)) := by
  have c116040 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.1
  have c116041 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.1
  have c116042 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116043 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116044 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116045 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1900_16 := arithEq_of_rows h (row := 1900) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_16
  simp only [← c116040, ← c116041, ← c116042] at e_1900_16
  have e_1900_17 := arithEq_of_rows h (row := 1900) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_17
  simp only [← c116043, ← c116044, k2, ← c116045] at e_1900_17
  have hr := e_1900_17
  simp only [e_1900_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18590 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1900 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8382)) := by
  have c116046 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116047 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116048 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116049 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116050 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116051 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1900_18 := arithEq_of_rows h (row := 1900) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_18
  simp only [← c116046, ← c116047, ← c116048] at e_1900_18
  have e_1900_19 := arithEq_of_rows h (row := 1900) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1900_19
  simp only [← c116049, ← c116050, k2, ← c116051] at e_1900_19
  have hr := e_1900_19
  simp only [e_1900_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18591 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1901 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8383)) := by
  have c116052 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116053 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116054 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116055 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116056 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116057 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1901_0 := arithEq_of_rows h (row := 1901) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_0
  simp only [← c116052, ← c116053, ← c116054] at e_1901_0
  have e_1901_1 := arithEq_of_rows h (row := 1901) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_1
  simp only [← c116055, ← c116056, k2, ← c116057] at e_1901_1
  have hr := e_1901_1
  simp only [e_1901_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18592 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1901 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8384)) := by
  have c116058 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116059 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116060 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116061 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116062 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116063 := (publicBatchWrapper89_copies3626 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1901_2 := arithEq_of_rows h (row := 1901) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_2
  simp only [← c116058, ← c116059, ← c116060] at e_1901_2
  have e_1901_3 := arithEq_of_rows h (row := 1901) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_3
  simp only [← c116061, ← c116062, k2, ← c116063] at e_1901_3
  have hr := e_1901_3
  simp only [e_1901_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18593 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1901 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8385)) := by
  have c116064 := (publicBatchWrapper89_copies3627 a h).1
  have c116065 := (publicBatchWrapper89_copies3627 a h).2.1
  have c116066 := (publicBatchWrapper89_copies3627 a h).2.2.1
  have c116067 := (publicBatchWrapper89_copies3627 a h).2.2.2.1
  have c116068 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.1
  have c116069 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1901_4 := arithEq_of_rows h (row := 1901) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_4
  simp only [← c116064, ← c116065, ← c116066] at e_1901_4
  have e_1901_5 := arithEq_of_rows h (row := 1901) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_5
  simp only [← c116067, ← c116068, k2, ← c116069] at e_1901_5
  have hr := e_1901_5
  simp only [e_1901_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18594 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1901 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8386)) := by
  have c116070 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.1
  have c116071 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.1
  have c116072 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.1
  have c116073 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.1
  have c116074 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116075 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1901_6 := arithEq_of_rows h (row := 1901) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_6
  simp only [← c116070, ← c116071, ← c116072] at e_1901_6
  have e_1901_7 := arithEq_of_rows h (row := 1901) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_7
  simp only [← c116073, ← c116074, k2, ← c116075] at e_1901_7
  have hr := e_1901_7
  simp only [e_1901_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18595 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1901 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8387)) := by
  have c116076 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116077 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116078 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116079 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116080 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116081 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1901_8 := arithEq_of_rows h (row := 1901) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_8
  simp only [← c116076, ← c116077, ← c116078] at e_1901_8
  have e_1901_9 := arithEq_of_rows h (row := 1901) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_9
  simp only [← c116079, ← c116080, k2, ← c116081] at e_1901_9
  have hr := e_1901_9
  simp only [e_1901_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18596 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1901 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8388)) := by
  have c116082 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116083 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116084 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116085 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116086 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116087 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1901_10 := arithEq_of_rows h (row := 1901) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_10
  simp only [← c116082, ← c116083, ← c116084] at e_1901_10
  have e_1901_11 := arithEq_of_rows h (row := 1901) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_11
  simp only [← c116085, ← c116086, k2, ← c116087] at e_1901_11
  have hr := e_1901_11
  simp only [e_1901_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18597 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1901 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8389)) := by
  have c116088 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116089 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116090 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116091 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116092 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116093 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1901_12 := arithEq_of_rows h (row := 1901) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_12
  simp only [← c116088, ← c116089, ← c116090] at e_1901_12
  have e_1901_13 := arithEq_of_rows h (row := 1901) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_13
  simp only [← c116091, ← c116092, k2, ← c116093] at e_1901_13
  have hr := e_1901_13
  simp only [e_1901_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18598 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1901 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8390)) := by
  have c116094 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116095 := (publicBatchWrapper89_copies3627 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116096 := (publicBatchWrapper89_copies3628 a h).1
  have c116097 := (publicBatchWrapper89_copies3628 a h).2.1
  have c116098 := (publicBatchWrapper89_copies3628 a h).2.2.1
  have c116099 := (publicBatchWrapper89_copies3628 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1901_14 := arithEq_of_rows h (row := 1901) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_14
  simp only [← c116094, ← c116095, ← c116096] at e_1901_14
  have e_1901_15 := arithEq_of_rows h (row := 1901) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_15
  simp only [← c116097, ← c116098, k2, ← c116099] at e_1901_15
  have hr := e_1901_15
  simp only [e_1901_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18599 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1901 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8391)) := by
  have c116100 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.1
  have c116101 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.1
  have c116102 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.1
  have c116103 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.1
  have c116104 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.1
  have c116105 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1901_16 := arithEq_of_rows h (row := 1901) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_16
  simp only [← c116100, ← c116101, ← c116102] at e_1901_16
  have e_1901_17 := arithEq_of_rows h (row := 1901) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_17
  simp only [← c116103, ← c116104, k2, ← c116105] at e_1901_17
  have hr := e_1901_17
  simp only [e_1901_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18600 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1901 79) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8392)) := by
  have c116106 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116107 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116108 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116109 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116110 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116111 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1901_18 := arithEq_of_rows h (row := 1901) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_18
  simp only [← c116106, ← c116107, ← c116108] at e_1901_18
  have e_1901_19 := arithEq_of_rows h (row := 1901) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1901_19
  simp only [← c116109, ← c116110, k2, ← c116111] at e_1901_19
  have hr := e_1901_19
  simp only [e_1901_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18601 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1902 7) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8393)) := by
  have c116112 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116113 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116114 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116115 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116116 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116117 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1902_0 := arithEq_of_rows h (row := 1902) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_0
  simp only [← c116112, ← c116113, ← c116114] at e_1902_0
  have e_1902_1 := arithEq_of_rows h (row := 1902) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_1
  simp only [← c116115, ← c116116, k2, ← c116117] at e_1902_1
  have hr := e_1902_1
  simp only [e_1902_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18602 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1902 15) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8394)) := by
  have c116118 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116119 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116120 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116121 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116122 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116123 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1902_2 := arithEq_of_rows h (row := 1902) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_2
  simp only [← c116118, ← c116119, ← c116120] at e_1902_2
  have e_1902_3 := arithEq_of_rows h (row := 1902) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_3
  simp only [← c116121, ← c116122, k2, ← c116123] at e_1902_3
  have hr := e_1902_3
  simp only [e_1902_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18603 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1902 23) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8395)) := by
  have c116124 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116125 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116126 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116127 := (publicBatchWrapper89_copies3628 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116128 := (publicBatchWrapper89_copies3629 a h).1
  have c116129 := (publicBatchWrapper89_copies3629 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1902_4 := arithEq_of_rows h (row := 1902) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_4
  simp only [← c116124, ← c116125, ← c116126] at e_1902_4
  have e_1902_5 := arithEq_of_rows h (row := 1902) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_5
  simp only [← c116127, ← c116128, k2, ← c116129] at e_1902_5
  have hr := e_1902_5
  simp only [e_1902_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18604 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1902 31) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8396)) := by
  have c116130 := (publicBatchWrapper89_copies3629 a h).2.2.1
  have c116131 := (publicBatchWrapper89_copies3629 a h).2.2.2.1
  have c116132 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.1
  have c116133 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.1
  have c116134 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.1
  have c116135 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1902_6 := arithEq_of_rows h (row := 1902) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_6
  simp only [← c116130, ← c116131, ← c116132] at e_1902_6
  have e_1902_7 := arithEq_of_rows h (row := 1902) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_7
  simp only [← c116133, ← c116134, k2, ← c116135] at e_1902_7
  have hr := e_1902_7
  simp only [e_1902_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18605 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1902 39) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8397)) := by
  have c116136 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.1
  have c116137 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.1
  have c116138 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116139 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116140 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116141 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1902_8 := arithEq_of_rows h (row := 1902) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_8
  simp only [← c116136, ← c116137, ← c116138] at e_1902_8
  have e_1902_9 := arithEq_of_rows h (row := 1902) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_9
  simp only [← c116139, ← c116140, k2, ← c116141] at e_1902_9
  have hr := e_1902_9
  simp only [e_1902_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18606 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1902 47) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8398)) := by
  have c116142 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116143 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116144 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116145 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116146 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116147 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1902_10 := arithEq_of_rows h (row := 1902) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_10
  simp only [← c116142, ← c116143, ← c116144] at e_1902_10
  have e_1902_11 := arithEq_of_rows h (row := 1902) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_11
  simp only [← c116145, ← c116146, k2, ← c116147] at e_1902_11
  have hr := e_1902_11
  simp only [e_1902_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18607 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1902 55) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8399)) := by
  have c116148 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116149 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116150 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116151 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116152 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116153 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1902_12 := arithEq_of_rows h (row := 1902) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_12
  simp only [← c116148, ← c116149, ← c116150] at e_1902_12
  have e_1902_13 := arithEq_of_rows h (row := 1902) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_13
  simp only [← c116151, ← c116152, k2, ← c116153] at e_1902_13
  have hr := e_1902_13
  simp only [e_1902_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18608 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1902 63) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8400)) := by
  have c116154 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116155 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116156 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116157 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116158 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116159 := (publicBatchWrapper89_copies3629 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1902_14 := arithEq_of_rows h (row := 1902) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_14
  simp only [← c116154, ← c116155, ← c116156] at e_1902_14
  have e_1902_15 := arithEq_of_rows h (row := 1902) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_15
  simp only [← c116157, ← c116158, k2, ← c116159] at e_1902_15
  have hr := e_1902_15
  simp only [e_1902_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18609 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1902 71) = bselect (a (.wire 41 23)) (a (.virt 200)) (a (.virt 8401)) := by
  have c116160 := (publicBatchWrapper89_copies3630 a h).1
  have c116161 := (publicBatchWrapper89_copies3630 a h).2.1
  have c116162 := (publicBatchWrapper89_copies3630 a h).2.2.1
  have c116163 := (publicBatchWrapper89_copies3630 a h).2.2.2.1
  have c116164 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.1
  have c116165 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1902_16 := arithEq_of_rows h (row := 1902) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_16
  simp only [← c116160, ← c116161, ← c116162] at e_1902_16
  have e_1902_17 := arithEq_of_rows h (row := 1902) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_17
  simp only [← c116163, ← c116164, k2, ← c116165] at e_1902_17
  have hr := e_1902_17
  simp only [e_1902_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18610 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1902 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8570)) := by
  have c116166 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.1
  have c116167 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.1
  have c116168 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.1
  have c116169 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.1
  have c116170 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116171 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1902_18 := arithEq_of_rows h (row := 1902) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_18
  simp only [← c116166, ← c116167, ← c116168] at e_1902_18
  have e_1902_19 := arithEq_of_rows h (row := 1902) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1902_19
  simp only [← c116169, ← c116170, k2, ← c116171] at e_1902_19
  have hr := e_1902_19
  simp only [e_1902_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18611 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1903 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8571)) := by
  have c116172 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116173 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116174 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116175 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116176 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116177 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1903_0 := arithEq_of_rows h (row := 1903) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_0
  simp only [← c116172, ← c116173, ← c116174] at e_1903_0
  have e_1903_1 := arithEq_of_rows h (row := 1903) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_1
  simp only [← c116175, ← c116176, k2, ← c116177] at e_1903_1
  have hr := e_1903_1
  simp only [e_1903_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18612 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1903 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8572)) := by
  have c116178 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116179 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116180 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116181 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116182 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116183 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1903_2 := arithEq_of_rows h (row := 1903) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_2
  simp only [← c116178, ← c116179, ← c116180] at e_1903_2
  have e_1903_3 := arithEq_of_rows h (row := 1903) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_3
  simp only [← c116181, ← c116182, k2, ← c116183] at e_1903_3
  have hr := e_1903_3
  simp only [e_1903_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18613 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1903 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8573)) := by
  have c116184 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116185 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116186 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116187 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116188 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116189 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1903_4 := arithEq_of_rows h (row := 1903) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_4
  simp only [← c116184, ← c116185, ← c116186] at e_1903_4
  have e_1903_5 := arithEq_of_rows h (row := 1903) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_5
  simp only [← c116187, ← c116188, k2, ← c116189] at e_1903_5
  have hr := e_1903_5
  simp only [e_1903_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18614 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1903 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8574)) := by
  have c116190 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116191 := (publicBatchWrapper89_copies3630 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116192 := (publicBatchWrapper89_copies3631 a h).1
  have c116193 := (publicBatchWrapper89_copies3631 a h).2.1
  have c116194 := (publicBatchWrapper89_copies3631 a h).2.2.1
  have c116195 := (publicBatchWrapper89_copies3631 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1903_6 := arithEq_of_rows h (row := 1903) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_6
  simp only [← c116190, ← c116191, ← c116192] at e_1903_6
  have e_1903_7 := arithEq_of_rows h (row := 1903) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_7
  simp only [← c116193, ← c116194, k2, ← c116195] at e_1903_7
  have hr := e_1903_7
  simp only [e_1903_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18615 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1903 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8575)) := by
  have c116196 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.1
  have c116197 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.1
  have c116198 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.1
  have c116199 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.1
  have c116200 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.1
  have c116201 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1903_8 := arithEq_of_rows h (row := 1903) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_8
  simp only [← c116196, ← c116197, ← c116198] at e_1903_8
  have e_1903_9 := arithEq_of_rows h (row := 1903) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_9
  simp only [← c116199, ← c116200, k2, ← c116201] at e_1903_9
  have hr := e_1903_9
  simp only [e_1903_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18616 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1903 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8576)) := by
  have c116202 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116203 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116204 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116205 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116206 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116207 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1903_10 := arithEq_of_rows h (row := 1903) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_10
  simp only [← c116202, ← c116203, ← c116204] at e_1903_10
  have e_1903_11 := arithEq_of_rows h (row := 1903) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_11
  simp only [← c116205, ← c116206, k2, ← c116207] at e_1903_11
  have hr := e_1903_11
  simp only [e_1903_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18617 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1903 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8577)) := by
  have c116208 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116209 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116210 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116211 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116212 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116213 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1903_12 := arithEq_of_rows h (row := 1903) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_12
  simp only [← c116208, ← c116209, ← c116210] at e_1903_12
  have e_1903_13 := arithEq_of_rows h (row := 1903) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_13
  simp only [← c116211, ← c116212, k2, ← c116213] at e_1903_13
  have hr := e_1903_13
  simp only [e_1903_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18618 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1903 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8578)) := by
  have c116214 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116215 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116216 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116217 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116218 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116219 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1903_14 := arithEq_of_rows h (row := 1903) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_14
  simp only [← c116214, ← c116215, ← c116216] at e_1903_14
  have e_1903_15 := arithEq_of_rows h (row := 1903) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_15
  simp only [← c116217, ← c116218, k2, ← c116219] at e_1903_15
  have hr := e_1903_15
  simp only [e_1903_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18619 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1903 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8579)) := by
  have c116220 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116221 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116222 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116223 := (publicBatchWrapper89_copies3631 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116224 := (publicBatchWrapper89_copies3632 a h).1
  have c116225 := (publicBatchWrapper89_copies3632 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1903_16 := arithEq_of_rows h (row := 1903) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_16
  simp only [← c116220, ← c116221, ← c116222] at e_1903_16
  have e_1903_17 := arithEq_of_rows h (row := 1903) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_17
  simp only [← c116223, ← c116224, k2, ← c116225] at e_1903_17
  have hr := e_1903_17
  simp only [e_1903_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18620 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1903 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8580)) := by
  have c116226 := (publicBatchWrapper89_copies3632 a h).2.2.1
  have c116227 := (publicBatchWrapper89_copies3632 a h).2.2.2.1
  have c116228 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.1
  have c116229 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.1
  have c116230 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.1
  have c116231 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1903_18 := arithEq_of_rows h (row := 1903) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_18
  simp only [← c116226, ← c116227, ← c116228] at e_1903_18
  have e_1903_19 := arithEq_of_rows h (row := 1903) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1903_19
  simp only [← c116229, ← c116230, k2, ← c116231] at e_1903_19
  have hr := e_1903_19
  simp only [e_1903_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18621 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1904 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8581)) := by
  have c116232 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.1
  have c116233 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.1
  have c116234 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116235 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116236 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116237 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1904_0 := arithEq_of_rows h (row := 1904) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_0
  simp only [← c116232, ← c116233, ← c116234] at e_1904_0
  have e_1904_1 := arithEq_of_rows h (row := 1904) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_1
  simp only [← c116235, ← c116236, k2, ← c116237] at e_1904_1
  have hr := e_1904_1
  simp only [e_1904_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18622 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1904 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8582)) := by
  have c116238 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116239 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116240 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116241 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116242 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116243 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1904_2 := arithEq_of_rows h (row := 1904) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_2
  simp only [← c116238, ← c116239, ← c116240] at e_1904_2
  have e_1904_3 := arithEq_of_rows h (row := 1904) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_3
  simp only [← c116241, ← c116242, k2, ← c116243] at e_1904_3
  have hr := e_1904_3
  simp only [e_1904_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18623 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1904 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8583)) := by
  have c116244 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116245 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116246 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116247 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116248 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116249 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1904_4 := arithEq_of_rows h (row := 1904) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_4
  simp only [← c116244, ← c116245, ← c116246] at e_1904_4
  have e_1904_5 := arithEq_of_rows h (row := 1904) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_5
  simp only [← c116247, ← c116248, k2, ← c116249] at e_1904_5
  have hr := e_1904_5
  simp only [e_1904_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18624 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1904 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8584)) := by
  have c116250 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116251 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116252 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116253 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116254 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116255 := (publicBatchWrapper89_copies3632 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1904_6 := arithEq_of_rows h (row := 1904) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_6
  simp only [← c116250, ← c116251, ← c116252] at e_1904_6
  have e_1904_7 := arithEq_of_rows h (row := 1904) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_7
  simp only [← c116253, ← c116254, k2, ← c116255] at e_1904_7
  have hr := e_1904_7
  simp only [e_1904_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18625 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1904 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8585)) := by
  have c116256 := (publicBatchWrapper89_copies3633 a h).1
  have c116257 := (publicBatchWrapper89_copies3633 a h).2.1
  have c116258 := (publicBatchWrapper89_copies3633 a h).2.2.1
  have c116259 := (publicBatchWrapper89_copies3633 a h).2.2.2.1
  have c116260 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.1
  have c116261 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1904_8 := arithEq_of_rows h (row := 1904) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_8
  simp only [← c116256, ← c116257, ← c116258] at e_1904_8
  have e_1904_9 := arithEq_of_rows h (row := 1904) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_9
  simp only [← c116259, ← c116260, k2, ← c116261] at e_1904_9
  have hr := e_1904_9
  simp only [e_1904_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18626 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1904 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8586)) := by
  have c116262 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.1
  have c116263 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.1
  have c116264 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.1
  have c116265 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.1
  have c116266 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116267 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1904_10 := arithEq_of_rows h (row := 1904) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_10
  simp only [← c116262, ← c116263, ← c116264] at e_1904_10
  have e_1904_11 := arithEq_of_rows h (row := 1904) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_11
  simp only [← c116265, ← c116266, k2, ← c116267] at e_1904_11
  have hr := e_1904_11
  simp only [e_1904_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18627 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1904 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8587)) := by
  have c116268 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116269 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116270 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116271 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116272 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116273 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1904_12 := arithEq_of_rows h (row := 1904) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_12
  simp only [← c116268, ← c116269, ← c116270] at e_1904_12
  have e_1904_13 := arithEq_of_rows h (row := 1904) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_13
  simp only [← c116271, ← c116272, k2, ← c116273] at e_1904_13
  have hr := e_1904_13
  simp only [e_1904_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18628 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1904 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8588)) := by
  have c116274 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116275 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116276 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116277 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116278 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116279 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1904_14 := arithEq_of_rows h (row := 1904) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_14
  simp only [← c116274, ← c116275, ← c116276] at e_1904_14
  have e_1904_15 := arithEq_of_rows h (row := 1904) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_15
  simp only [← c116277, ← c116278, k2, ← c116279] at e_1904_15
  have hr := e_1904_15
  simp only [e_1904_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18629 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1904 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8589)) := by
  have c116280 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116281 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116282 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116283 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116284 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116285 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1904_16 := arithEq_of_rows h (row := 1904) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_16
  simp only [← c116280, ← c116281, ← c116282] at e_1904_16
  have e_1904_17 := arithEq_of_rows h (row := 1904) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_17
  simp only [← c116283, ← c116284, k2, ← c116285] at e_1904_17
  have hr := e_1904_17
  simp only [e_1904_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18630 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1904 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8590)) := by
  have c116286 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116287 := (publicBatchWrapper89_copies3633 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116288 := (publicBatchWrapper89_copies3634 a h).1
  have c116289 := (publicBatchWrapper89_copies3634 a h).2.1
  have c116290 := (publicBatchWrapper89_copies3634 a h).2.2.1
  have c116291 := (publicBatchWrapper89_copies3634 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1904_18 := arithEq_of_rows h (row := 1904) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_18
  simp only [← c116286, ← c116287, ← c116288] at e_1904_18
  have e_1904_19 := arithEq_of_rows h (row := 1904) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1904_19
  simp only [← c116289, ← c116290, k2, ← c116291] at e_1904_19
  have hr := e_1904_19
  simp only [e_1904_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18631 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1905 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8591)) := by
  have c116292 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.1
  have c116293 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.1
  have c116294 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.1
  have c116295 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.1
  have c116296 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.1
  have c116297 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1905_0 := arithEq_of_rows h (row := 1905) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_0
  simp only [← c116292, ← c116293, ← c116294] at e_1905_0
  have e_1905_1 := arithEq_of_rows h (row := 1905) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_1
  simp only [← c116295, ← c116296, k2, ← c116297] at e_1905_1
  have hr := e_1905_1
  simp only [e_1905_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18632 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1905 15) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8592)) := by
  have c116298 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116299 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116300 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116301 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116302 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116303 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1905_2 := arithEq_of_rows h (row := 1905) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_2
  simp only [← c116298, ← c116299, ← c116300] at e_1905_2
  have e_1905_3 := arithEq_of_rows h (row := 1905) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_3
  simp only [← c116301, ← c116302, k2, ← c116303] at e_1905_3
  have hr := e_1905_3
  simp only [e_1905_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18633 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1905 23) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8593)) := by
  have c116304 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116305 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116306 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116307 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116308 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116309 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1905_4 := arithEq_of_rows h (row := 1905) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_4
  simp only [← c116304, ← c116305, ← c116306] at e_1905_4
  have e_1905_5 := arithEq_of_rows h (row := 1905) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_5
  simp only [← c116307, ← c116308, k2, ← c116309] at e_1905_5
  have hr := e_1905_5
  simp only [e_1905_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18634 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1905 31) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8594)) := by
  have c116310 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116311 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116312 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116313 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116314 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116315 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1905_6 := arithEq_of_rows h (row := 1905) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_6
  simp only [← c116310, ← c116311, ← c116312] at e_1905_6
  have e_1905_7 := arithEq_of_rows h (row := 1905) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_7
  simp only [← c116313, ← c116314, k2, ← c116315] at e_1905_7
  have hr := e_1905_7
  simp only [e_1905_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18635 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1905 39) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8595)) := by
  have c116316 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116317 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116318 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116319 := (publicBatchWrapper89_copies3634 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116320 := (publicBatchWrapper89_copies3635 a h).1
  have c116321 := (publicBatchWrapper89_copies3635 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1905_8 := arithEq_of_rows h (row := 1905) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_8
  simp only [← c116316, ← c116317, ← c116318] at e_1905_8
  have e_1905_9 := arithEq_of_rows h (row := 1905) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_9
  simp only [← c116319, ← c116320, k2, ← c116321] at e_1905_9
  have hr := e_1905_9
  simp only [e_1905_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18636 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1905 47) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8596)) := by
  have c116322 := (publicBatchWrapper89_copies3635 a h).2.2.1
  have c116323 := (publicBatchWrapper89_copies3635 a h).2.2.2.1
  have c116324 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.1
  have c116325 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.1
  have c116326 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.1
  have c116327 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1905_10 := arithEq_of_rows h (row := 1905) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_10
  simp only [← c116322, ← c116323, ← c116324] at e_1905_10
  have e_1905_11 := arithEq_of_rows h (row := 1905) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_11
  simp only [← c116325, ← c116326, k2, ← c116327] at e_1905_11
  have hr := e_1905_11
  simp only [e_1905_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18637 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1905 55) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8597)) := by
  have c116328 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.1
  have c116329 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.1
  have c116330 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116331 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116332 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116333 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1905_12 := arithEq_of_rows h (row := 1905) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_12
  simp only [← c116328, ← c116329, ← c116330] at e_1905_12
  have e_1905_13 := arithEq_of_rows h (row := 1905) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_13
  simp only [← c116331, ← c116332, k2, ← c116333] at e_1905_13
  have hr := e_1905_13
  simp only [e_1905_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18638 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1905 63) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8598)) := by
  have c116334 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116335 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116336 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116337 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116338 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116339 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1905_14 := arithEq_of_rows h (row := 1905) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_14
  simp only [← c116334, ← c116335, ← c116336] at e_1905_14
  have e_1905_15 := arithEq_of_rows h (row := 1905) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_15
  simp only [← c116337, ← c116338, k2, ← c116339] at e_1905_15
  have hr := e_1905_15
  simp only [e_1905_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18639 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1905 71) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8599)) := by
  have c116340 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116341 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116342 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116343 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116344 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116345 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1905_16 := arithEq_of_rows h (row := 1905) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_16
  simp only [← c116340, ← c116341, ← c116342] at e_1905_16
  have e_1905_17 := arithEq_of_rows h (row := 1905) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_17
  simp only [← c116343, ← c116344, k2, ← c116345] at e_1905_17
  have hr := e_1905_17
  simp only [e_1905_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18640 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1905 79) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8600)) := by
  have c116346 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116347 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116348 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116349 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116350 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116351 := (publicBatchWrapper89_copies3635 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1905_18 := arithEq_of_rows h (row := 1905) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_18
  simp only [← c116346, ← c116347, ← c116348] at e_1905_18
  have e_1905_19 := arithEq_of_rows h (row := 1905) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1905_19
  simp only [← c116349, ← c116350, k2, ← c116351] at e_1905_19
  have hr := e_1905_19
  simp only [e_1905_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18641 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1906 7) = bselect (a (.wire 41 67)) (a (.virt 200)) (a (.virt 8601)) := by
  have c116352 := (publicBatchWrapper89_copies3636 a h).1
  have c116353 := (publicBatchWrapper89_copies3636 a h).2.1
  have c116354 := (publicBatchWrapper89_copies3636 a h).2.2.1
  have c116355 := (publicBatchWrapper89_copies3636 a h).2.2.2.1
  have c116356 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.1
  have c116357 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1906_0 := arithEq_of_rows h (row := 1906) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_0
  simp only [← c116352, ← c116353, ← c116354] at e_1906_0
  have e_1906_1 := arithEq_of_rows h (row := 1906) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_1
  simp only [← c116355, ← c116356, k2, ← c116357] at e_1906_1
  have hr := e_1906_1
  simp only [e_1906_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18642 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1906 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8770)) := by
  have c116358 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.1
  have c116359 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.1
  have c116360 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.1
  have c116361 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.1
  have c116362 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116363 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1906_2 := arithEq_of_rows h (row := 1906) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_2
  simp only [← c116358, ← c116359, ← c116360] at e_1906_2
  have e_1906_3 := arithEq_of_rows h (row := 1906) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_3
  simp only [← c116361, ← c116362, k2, ← c116363] at e_1906_3
  have hr := e_1906_3
  simp only [e_1906_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18643 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1906 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8771)) := by
  have c116364 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116365 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116366 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116367 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116368 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116369 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1906_4 := arithEq_of_rows h (row := 1906) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_4
  simp only [← c116364, ← c116365, ← c116366] at e_1906_4
  have e_1906_5 := arithEq_of_rows h (row := 1906) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_5
  simp only [← c116367, ← c116368, k2, ← c116369] at e_1906_5
  have hr := e_1906_5
  simp only [e_1906_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18644 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1906 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8772)) := by
  have c116370 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116371 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116372 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116373 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116374 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116375 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1906_6 := arithEq_of_rows h (row := 1906) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_6
  simp only [← c116370, ← c116371, ← c116372] at e_1906_6
  have e_1906_7 := arithEq_of_rows h (row := 1906) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_7
  simp only [← c116373, ← c116374, k2, ← c116375] at e_1906_7
  have hr := e_1906_7
  simp only [e_1906_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18645 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1906 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8773)) := by
  have c116376 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116377 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116378 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116379 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116380 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116381 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1906_8 := arithEq_of_rows h (row := 1906) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_8
  simp only [← c116376, ← c116377, ← c116378] at e_1906_8
  have e_1906_9 := arithEq_of_rows h (row := 1906) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_9
  simp only [← c116379, ← c116380, k2, ← c116381] at e_1906_9
  have hr := e_1906_9
  simp only [e_1906_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18646 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1906 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8774)) := by
  have c116382 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116383 := (publicBatchWrapper89_copies3636 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116384 := (publicBatchWrapper89_copies3637 a h).1
  have c116385 := (publicBatchWrapper89_copies3637 a h).2.1
  have c116386 := (publicBatchWrapper89_copies3637 a h).2.2.1
  have c116387 := (publicBatchWrapper89_copies3637 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1906_10 := arithEq_of_rows h (row := 1906) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_10
  simp only [← c116382, ← c116383, ← c116384] at e_1906_10
  have e_1906_11 := arithEq_of_rows h (row := 1906) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_11
  simp only [← c116385, ← c116386, k2, ← c116387] at e_1906_11
  have hr := e_1906_11
  simp only [e_1906_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18647 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1906 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8775)) := by
  have c116388 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.1
  have c116389 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.1
  have c116390 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.1
  have c116391 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.1
  have c116392 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.1
  have c116393 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1906_12 := arithEq_of_rows h (row := 1906) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_12
  simp only [← c116388, ← c116389, ← c116390] at e_1906_12
  have e_1906_13 := arithEq_of_rows h (row := 1906) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_13
  simp only [← c116391, ← c116392, k2, ← c116393] at e_1906_13
  have hr := e_1906_13
  simp only [e_1906_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18648 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1906 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8776)) := by
  have c116394 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116395 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116396 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116397 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116398 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116399 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1906_14 := arithEq_of_rows h (row := 1906) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_14
  simp only [← c116394, ← c116395, ← c116396] at e_1906_14
  have e_1906_15 := arithEq_of_rows h (row := 1906) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_15
  simp only [← c116397, ← c116398, k2, ← c116399] at e_1906_15
  have hr := e_1906_15
  simp only [e_1906_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18649 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1906 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8777)) := by
  have c116400 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116401 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116402 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116403 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116404 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116405 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1906_16 := arithEq_of_rows h (row := 1906) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_16
  simp only [← c116400, ← c116401, ← c116402] at e_1906_16
  have e_1906_17 := arithEq_of_rows h (row := 1906) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_17
  simp only [← c116403, ← c116404, k2, ← c116405] at e_1906_17
  have hr := e_1906_17
  simp only [e_1906_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18650 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1906 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8778)) := by
  have c116406 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116407 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116408 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116409 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116410 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116411 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1906_18 := arithEq_of_rows h (row := 1906) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_18
  simp only [← c116406, ← c116407, ← c116408] at e_1906_18
  have e_1906_19 := arithEq_of_rows h (row := 1906) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1906_19
  simp only [← c116409, ← c116410, k2, ← c116411] at e_1906_19
  have hr := e_1906_19
  simp only [e_1906_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18651 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1907 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8779)) := by
  have c116412 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116413 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116414 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116415 := (publicBatchWrapper89_copies3637 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116416 := (publicBatchWrapper89_copies3638 a h).1
  have c116417 := (publicBatchWrapper89_copies3638 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1907_0 := arithEq_of_rows h (row := 1907) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_0
  simp only [← c116412, ← c116413, ← c116414] at e_1907_0
  have e_1907_1 := arithEq_of_rows h (row := 1907) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_1
  simp only [← c116415, ← c116416, k2, ← c116417] at e_1907_1
  have hr := e_1907_1
  simp only [e_1907_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18652 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1907 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8780)) := by
  have c116418 := (publicBatchWrapper89_copies3638 a h).2.2.1
  have c116419 := (publicBatchWrapper89_copies3638 a h).2.2.2.1
  have c116420 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.1
  have c116421 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.1
  have c116422 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.1
  have c116423 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1907_2 := arithEq_of_rows h (row := 1907) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_2
  simp only [← c116418, ← c116419, ← c116420] at e_1907_2
  have e_1907_3 := arithEq_of_rows h (row := 1907) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_3
  simp only [← c116421, ← c116422, k2, ← c116423] at e_1907_3
  have hr := e_1907_3
  simp only [e_1907_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18653 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1907 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8781)) := by
  have c116424 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.1
  have c116425 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.1
  have c116426 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116427 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116428 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116429 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1907_4 := arithEq_of_rows h (row := 1907) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_4
  simp only [← c116424, ← c116425, ← c116426] at e_1907_4
  have e_1907_5 := arithEq_of_rows h (row := 1907) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_5
  simp only [← c116427, ← c116428, k2, ← c116429] at e_1907_5
  have hr := e_1907_5
  simp only [e_1907_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18654 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1907 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8782)) := by
  have c116430 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116431 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116432 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116433 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116434 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116435 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1907_6 := arithEq_of_rows h (row := 1907) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_6
  simp only [← c116430, ← c116431, ← c116432] at e_1907_6
  have e_1907_7 := arithEq_of_rows h (row := 1907) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_7
  simp only [← c116433, ← c116434, k2, ← c116435] at e_1907_7
  have hr := e_1907_7
  simp only [e_1907_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18655 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1907 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8783)) := by
  have c116436 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116437 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116438 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116439 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116440 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116441 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1907_8 := arithEq_of_rows h (row := 1907) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_8
  simp only [← c116436, ← c116437, ← c116438] at e_1907_8
  have e_1907_9 := arithEq_of_rows h (row := 1907) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_9
  simp only [← c116439, ← c116440, k2, ← c116441] at e_1907_9
  have hr := e_1907_9
  simp only [e_1907_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18656 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1907 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8784)) := by
  have c116442 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116443 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116444 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116445 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116446 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116447 := (publicBatchWrapper89_copies3638 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1907_10 := arithEq_of_rows h (row := 1907) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_10
  simp only [← c116442, ← c116443, ← c116444] at e_1907_10
  have e_1907_11 := arithEq_of_rows h (row := 1907) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_11
  simp only [← c116445, ← c116446, k2, ← c116447] at e_1907_11
  have hr := e_1907_11
  simp only [e_1907_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18657 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1907 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8785)) := by
  have c116448 := (publicBatchWrapper89_copies3639 a h).1
  have c116449 := (publicBatchWrapper89_copies3639 a h).2.1
  have c116450 := (publicBatchWrapper89_copies3639 a h).2.2.1
  have c116451 := (publicBatchWrapper89_copies3639 a h).2.2.2.1
  have c116452 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.1
  have c116453 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1907_12 := arithEq_of_rows h (row := 1907) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_12
  simp only [← c116448, ← c116449, ← c116450] at e_1907_12
  have e_1907_13 := arithEq_of_rows h (row := 1907) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_13
  simp only [← c116451, ← c116452, k2, ← c116453] at e_1907_13
  have hr := e_1907_13
  simp only [e_1907_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18658 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1907 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8786)) := by
  have c116454 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.1
  have c116455 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.1
  have c116456 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.1
  have c116457 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.1
  have c116458 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116459 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1907_14 := arithEq_of_rows h (row := 1907) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_14
  simp only [← c116454, ← c116455, ← c116456] at e_1907_14
  have e_1907_15 := arithEq_of_rows h (row := 1907) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_15
  simp only [← c116457, ← c116458, k2, ← c116459] at e_1907_15
  have hr := e_1907_15
  simp only [e_1907_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18659 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1907 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8787)) := by
  have c116460 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116461 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116462 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116463 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116464 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116465 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1907_16 := arithEq_of_rows h (row := 1907) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_16
  simp only [← c116460, ← c116461, ← c116462] at e_1907_16
  have e_1907_17 := arithEq_of_rows h (row := 1907) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_17
  simp only [← c116463, ← c116464, k2, ← c116465] at e_1907_17
  have hr := e_1907_17
  simp only [e_1907_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18660 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1907 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8788)) := by
  have c116466 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116467 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116468 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116469 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116470 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116471 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1907_18 := arithEq_of_rows h (row := 1907) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_18
  simp only [← c116466, ← c116467, ← c116468] at e_1907_18
  have e_1907_19 := arithEq_of_rows h (row := 1907) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1907_19
  simp only [← c116469, ← c116470, k2, ← c116471] at e_1907_19
  have hr := e_1907_19
  simp only [e_1907_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18661 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1908 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8789)) := by
  have c116472 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116473 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116474 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116475 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116476 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116477 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1908_0 := arithEq_of_rows h (row := 1908) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_0
  simp only [← c116472, ← c116473, ← c116474] at e_1908_0
  have e_1908_1 := arithEq_of_rows h (row := 1908) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_1
  simp only [← c116475, ← c116476, k2, ← c116477] at e_1908_1
  have hr := e_1908_1
  simp only [e_1908_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18662 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1908 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8790)) := by
  have c116478 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116479 := (publicBatchWrapper89_copies3639 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116480 := (publicBatchWrapper89_copies3640 a h).1
  have c116481 := (publicBatchWrapper89_copies3640 a h).2.1
  have c116482 := (publicBatchWrapper89_copies3640 a h).2.2.1
  have c116483 := (publicBatchWrapper89_copies3640 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1908_2 := arithEq_of_rows h (row := 1908) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_2
  simp only [← c116478, ← c116479, ← c116480] at e_1908_2
  have e_1908_3 := arithEq_of_rows h (row := 1908) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_3
  simp only [← c116481, ← c116482, k2, ← c116483] at e_1908_3
  have hr := e_1908_3
  simp only [e_1908_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18663 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1908 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8791)) := by
  have c116484 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.1
  have c116485 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.1
  have c116486 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.1
  have c116487 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.1
  have c116488 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.1
  have c116489 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1908_4 := arithEq_of_rows h (row := 1908) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_4
  simp only [← c116484, ← c116485, ← c116486] at e_1908_4
  have e_1908_5 := arithEq_of_rows h (row := 1908) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_5
  simp only [← c116487, ← c116488, k2, ← c116489] at e_1908_5
  have hr := e_1908_5
  simp only [e_1908_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18664 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1908 31) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8792)) := by
  have c116490 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116491 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116492 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116493 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116494 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116495 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1908_6 := arithEq_of_rows h (row := 1908) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_6
  simp only [← c116490, ← c116491, ← c116492] at e_1908_6
  have e_1908_7 := arithEq_of_rows h (row := 1908) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_7
  simp only [← c116493, ← c116494, k2, ← c116495] at e_1908_7
  have hr := e_1908_7
  simp only [e_1908_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18665 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1908 39) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8793)) := by
  have c116496 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116497 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116498 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116499 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116500 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116501 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1908_8 := arithEq_of_rows h (row := 1908) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_8
  simp only [← c116496, ← c116497, ← c116498] at e_1908_8
  have e_1908_9 := arithEq_of_rows h (row := 1908) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_9
  simp only [← c116499, ← c116500, k2, ← c116501] at e_1908_9
  have hr := e_1908_9
  simp only [e_1908_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18666 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1908 47) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8794)) := by
  have c116502 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116503 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116504 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116505 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116506 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116507 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1908_10 := arithEq_of_rows h (row := 1908) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_10
  simp only [← c116502, ← c116503, ← c116504] at e_1908_10
  have e_1908_11 := arithEq_of_rows h (row := 1908) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_11
  simp only [← c116505, ← c116506, k2, ← c116507] at e_1908_11
  have hr := e_1908_11
  simp only [e_1908_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18667 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1908 55) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8795)) := by
  have c116508 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116509 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116510 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116511 := (publicBatchWrapper89_copies3640 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116512 := (publicBatchWrapper89_copies3641 a h).1
  have c116513 := (publicBatchWrapper89_copies3641 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1908_12 := arithEq_of_rows h (row := 1908) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_12
  simp only [← c116508, ← c116509, ← c116510] at e_1908_12
  have e_1908_13 := arithEq_of_rows h (row := 1908) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_13
  simp only [← c116511, ← c116512, k2, ← c116513] at e_1908_13
  have hr := e_1908_13
  simp only [e_1908_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18668 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1908 63) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8796)) := by
  have c116514 := (publicBatchWrapper89_copies3641 a h).2.2.1
  have c116515 := (publicBatchWrapper89_copies3641 a h).2.2.2.1
  have c116516 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.1
  have c116517 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.1
  have c116518 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.1
  have c116519 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1908_14 := arithEq_of_rows h (row := 1908) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_14
  simp only [← c116514, ← c116515, ← c116516] at e_1908_14
  have e_1908_15 := arithEq_of_rows h (row := 1908) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_15
  simp only [← c116517, ← c116518, k2, ← c116519] at e_1908_15
  have hr := e_1908_15
  simp only [e_1908_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18669 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1908 71) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8797)) := by
  have c116520 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.1
  have c116521 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.1
  have c116522 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116523 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116524 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116525 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1908_16 := arithEq_of_rows h (row := 1908) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_16
  simp only [← c116520, ← c116521, ← c116522] at e_1908_16
  have e_1908_17 := arithEq_of_rows h (row := 1908) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_17
  simp only [← c116523, ← c116524, k2, ← c116525] at e_1908_17
  have hr := e_1908_17
  simp only [e_1908_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18670 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1908 79) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8798)) := by
  have c116526 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116527 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116528 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116529 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116530 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116531 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1908_18 := arithEq_of_rows h (row := 1908) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_18
  simp only [← c116526, ← c116527, ← c116528] at e_1908_18
  have e_1908_19 := arithEq_of_rows h (row := 1908) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1908_19
  simp only [← c116529, ← c116530, k2, ← c116531] at e_1908_19
  have hr := e_1908_19
  simp only [e_1908_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18671 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1909 7) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8799)) := by
  have c116532 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116533 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116534 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116535 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116536 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116537 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1909_0 := arithEq_of_rows h (row := 1909) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_0
  simp only [← c116532, ← c116533, ← c116534] at e_1909_0
  have e_1909_1 := arithEq_of_rows h (row := 1909) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_1
  simp only [← c116535, ← c116536, k2, ← c116537] at e_1909_1
  have hr := e_1909_1
  simp only [e_1909_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18672 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1909 15) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8800)) := by
  have c116538 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116539 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116540 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116541 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116542 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116543 := (publicBatchWrapper89_copies3641 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1909_2 := arithEq_of_rows h (row := 1909) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_2
  simp only [← c116538, ← c116539, ← c116540] at e_1909_2
  have e_1909_3 := arithEq_of_rows h (row := 1909) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_3
  simp only [← c116541, ← c116542, k2, ← c116543] at e_1909_3
  have hr := e_1909_3
  simp only [e_1909_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18673 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1909 23) = bselect (a (.wire 43 31)) (a (.virt 200)) (a (.virt 8801)) := by
  have c116544 := (publicBatchWrapper89_copies3642 a h).1
  have c116545 := (publicBatchWrapper89_copies3642 a h).2.1
  have c116546 := (publicBatchWrapper89_copies3642 a h).2.2.1
  have c116547 := (publicBatchWrapper89_copies3642 a h).2.2.2.1
  have c116548 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.1
  have c116549 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1909_4 := arithEq_of_rows h (row := 1909) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_4
  simp only [← c116544, ← c116545, ← c116546] at e_1909_4
  have e_1909_5 := arithEq_of_rows h (row := 1909) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_5
  simp only [← c116547, ← c116548, k2, ← c116549] at e_1909_5
  have hr := e_1909_5
  simp only [e_1909_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18674 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1909 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8970)) := by
  have c116550 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.1
  have c116551 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.1
  have c116552 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.1
  have c116553 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.1
  have c116554 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116555 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1909_6 := arithEq_of_rows h (row := 1909) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_6
  simp only [← c116550, ← c116551, ← c116552] at e_1909_6
  have e_1909_7 := arithEq_of_rows h (row := 1909) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_7
  simp only [← c116553, ← c116554, k2, ← c116555] at e_1909_7
  have hr := e_1909_7
  simp only [e_1909_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18675 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1909 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8971)) := by
  have c116556 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116557 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116558 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116559 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116560 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116561 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1909_8 := arithEq_of_rows h (row := 1909) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_8
  simp only [← c116556, ← c116557, ← c116558] at e_1909_8
  have e_1909_9 := arithEq_of_rows h (row := 1909) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_9
  simp only [← c116559, ← c116560, k2, ← c116561] at e_1909_9
  have hr := e_1909_9
  simp only [e_1909_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18676 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1909 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8972)) := by
  have c116562 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116563 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116564 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116565 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116566 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116567 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1909_10 := arithEq_of_rows h (row := 1909) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_10
  simp only [← c116562, ← c116563, ← c116564] at e_1909_10
  have e_1909_11 := arithEq_of_rows h (row := 1909) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_11
  simp only [← c116565, ← c116566, k2, ← c116567] at e_1909_11
  have hr := e_1909_11
  simp only [e_1909_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18677 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1909 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8973)) := by
  have c116568 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116569 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116570 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116571 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116572 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116573 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1909_12 := arithEq_of_rows h (row := 1909) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_12
  simp only [← c116568, ← c116569, ← c116570] at e_1909_12
  have e_1909_13 := arithEq_of_rows h (row := 1909) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_13
  simp only [← c116571, ← c116572, k2, ← c116573] at e_1909_13
  have hr := e_1909_13
  simp only [e_1909_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18678 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1909 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8974)) := by
  have c116574 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116575 := (publicBatchWrapper89_copies3642 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116576 := (publicBatchWrapper89_copies3643 a h).1
  have c116577 := (publicBatchWrapper89_copies3643 a h).2.1
  have c116578 := (publicBatchWrapper89_copies3643 a h).2.2.1
  have c116579 := (publicBatchWrapper89_copies3643 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1909_14 := arithEq_of_rows h (row := 1909) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_14
  simp only [← c116574, ← c116575, ← c116576] at e_1909_14
  have e_1909_15 := arithEq_of_rows h (row := 1909) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_15
  simp only [← c116577, ← c116578, k2, ← c116579] at e_1909_15
  have hr := e_1909_15
  simp only [e_1909_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18679 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1909 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8975)) := by
  have c116580 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.1
  have c116581 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.1
  have c116582 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.1
  have c116583 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.1
  have c116584 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.1
  have c116585 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1909_16 := arithEq_of_rows h (row := 1909) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_16
  simp only [← c116580, ← c116581, ← c116582] at e_1909_16
  have e_1909_17 := arithEq_of_rows h (row := 1909) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_17
  simp only [← c116583, ← c116584, k2, ← c116585] at e_1909_17
  have hr := e_1909_17
  simp only [e_1909_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18680 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1909 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8976)) := by
  have c116586 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116587 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116588 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116589 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116590 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116591 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1909_18 := arithEq_of_rows h (row := 1909) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_18
  simp only [← c116586, ← c116587, ← c116588] at e_1909_18
  have e_1909_19 := arithEq_of_rows h (row := 1909) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1909_19
  simp only [← c116589, ← c116590, k2, ← c116591] at e_1909_19
  have hr := e_1909_19
  simp only [e_1909_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18681 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1910 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8977)) := by
  have c116592 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116593 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116594 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116595 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116596 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116597 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1910_0 := arithEq_of_rows h (row := 1910) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_0
  simp only [← c116592, ← c116593, ← c116594] at e_1910_0
  have e_1910_1 := arithEq_of_rows h (row := 1910) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_1
  simp only [← c116595, ← c116596, k2, ← c116597] at e_1910_1
  have hr := e_1910_1
  simp only [e_1910_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18682 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1910 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8978)) := by
  have c116598 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116599 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116600 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116601 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116602 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116603 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1910_2 := arithEq_of_rows h (row := 1910) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_2
  simp only [← c116598, ← c116599, ← c116600] at e_1910_2
  have e_1910_3 := arithEq_of_rows h (row := 1910) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_3
  simp only [← c116601, ← c116602, k2, ← c116603] at e_1910_3
  have hr := e_1910_3
  simp only [e_1910_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18683 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1910 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8979)) := by
  have c116604 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116605 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116606 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116607 := (publicBatchWrapper89_copies3643 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116608 := (publicBatchWrapper89_copies3644 a h).1
  have c116609 := (publicBatchWrapper89_copies3644 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1910_4 := arithEq_of_rows h (row := 1910) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_4
  simp only [← c116604, ← c116605, ← c116606] at e_1910_4
  have e_1910_5 := arithEq_of_rows h (row := 1910) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_5
  simp only [← c116607, ← c116608, k2, ← c116609] at e_1910_5
  have hr := e_1910_5
  simp only [e_1910_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18684 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1910 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8980)) := by
  have c116610 := (publicBatchWrapper89_copies3644 a h).2.2.1
  have c116611 := (publicBatchWrapper89_copies3644 a h).2.2.2.1
  have c116612 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.1
  have c116613 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.1
  have c116614 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.1
  have c116615 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1910_6 := arithEq_of_rows h (row := 1910) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_6
  simp only [← c116610, ← c116611, ← c116612] at e_1910_6
  have e_1910_7 := arithEq_of_rows h (row := 1910) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_7
  simp only [← c116613, ← c116614, k2, ← c116615] at e_1910_7
  have hr := e_1910_7
  simp only [e_1910_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18685 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1910 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8981)) := by
  have c116616 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.1
  have c116617 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.1
  have c116618 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116619 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116620 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116621 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1910_8 := arithEq_of_rows h (row := 1910) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_8
  simp only [← c116616, ← c116617, ← c116618] at e_1910_8
  have e_1910_9 := arithEq_of_rows h (row := 1910) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_9
  simp only [← c116619, ← c116620, k2, ← c116621] at e_1910_9
  have hr := e_1910_9
  simp only [e_1910_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18686 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1910 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8982)) := by
  have c116622 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116623 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116624 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116625 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116626 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116627 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1910_10 := arithEq_of_rows h (row := 1910) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_10
  simp only [← c116622, ← c116623, ← c116624] at e_1910_10
  have e_1910_11 := arithEq_of_rows h (row := 1910) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_11
  simp only [← c116625, ← c116626, k2, ← c116627] at e_1910_11
  have hr := e_1910_11
  simp only [e_1910_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18687 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1910 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8983)) := by
  have c116628 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116629 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116630 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116631 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116632 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116633 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1910_12 := arithEq_of_rows h (row := 1910) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_12
  simp only [← c116628, ← c116629, ← c116630] at e_1910_12
  have e_1910_13 := arithEq_of_rows h (row := 1910) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_13
  simp only [← c116631, ← c116632, k2, ← c116633] at e_1910_13
  have hr := e_1910_13
  simp only [e_1910_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18688 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1910 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8984)) := by
  have c116634 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116635 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116636 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116637 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116638 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116639 := (publicBatchWrapper89_copies3644 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1910_14 := arithEq_of_rows h (row := 1910) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_14
  simp only [← c116634, ← c116635, ← c116636] at e_1910_14
  have e_1910_15 := arithEq_of_rows h (row := 1910) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_15
  simp only [← c116637, ← c116638, k2, ← c116639] at e_1910_15
  have hr := e_1910_15
  simp only [e_1910_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18689 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1910 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8985)) := by
  have c116640 := (publicBatchWrapper89_copies3645 a h).1
  have c116641 := (publicBatchWrapper89_copies3645 a h).2.1
  have c116642 := (publicBatchWrapper89_copies3645 a h).2.2.1
  have c116643 := (publicBatchWrapper89_copies3645 a h).2.2.2.1
  have c116644 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.1
  have c116645 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1910_16 := arithEq_of_rows h (row := 1910) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_16
  simp only [← c116640, ← c116641, ← c116642] at e_1910_16
  have e_1910_17 := arithEq_of_rows h (row := 1910) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_17
  simp only [← c116643, ← c116644, k2, ← c116645] at e_1910_17
  have hr := e_1910_17
  simp only [e_1910_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18690 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1910 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8986)) := by
  have c116646 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.1
  have c116647 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.1
  have c116648 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.1
  have c116649 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.1
  have c116650 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116651 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1910_18 := arithEq_of_rows h (row := 1910) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_18
  simp only [← c116646, ← c116647, ← c116648] at e_1910_18
  have e_1910_19 := arithEq_of_rows h (row := 1910) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1910_19
  simp only [← c116649, ← c116650, k2, ← c116651] at e_1910_19
  have hr := e_1910_19
  simp only [e_1910_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18691 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1911 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8987)) := by
  have c116652 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116653 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116654 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116655 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116656 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116657 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1911_0 := arithEq_of_rows h (row := 1911) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_0
  simp only [← c116652, ← c116653, ← c116654] at e_1911_0
  have e_1911_1 := arithEq_of_rows h (row := 1911) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_1
  simp only [← c116655, ← c116656, k2, ← c116657] at e_1911_1
  have hr := e_1911_1
  simp only [e_1911_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18692 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1911 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8988)) := by
  have c116658 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116659 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116660 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116661 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116662 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116663 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1911_2 := arithEq_of_rows h (row := 1911) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_2
  simp only [← c116658, ← c116659, ← c116660] at e_1911_2
  have e_1911_3 := arithEq_of_rows h (row := 1911) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_3
  simp only [← c116661, ← c116662, k2, ← c116663] at e_1911_3
  have hr := e_1911_3
  simp only [e_1911_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18693 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1911 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8989)) := by
  have c116664 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116665 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116666 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116667 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116668 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116669 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1911_4 := arithEq_of_rows h (row := 1911) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_4
  simp only [← c116664, ← c116665, ← c116666] at e_1911_4
  have e_1911_5 := arithEq_of_rows h (row := 1911) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_5
  simp only [← c116667, ← c116668, k2, ← c116669] at e_1911_5
  have hr := e_1911_5
  simp only [e_1911_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18694 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1911 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8990)) := by
  have c116670 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116671 := (publicBatchWrapper89_copies3645 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116672 := (publicBatchWrapper89_copies3646 a h).1
  have c116673 := (publicBatchWrapper89_copies3646 a h).2.1
  have c116674 := (publicBatchWrapper89_copies3646 a h).2.2.1
  have c116675 := (publicBatchWrapper89_copies3646 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1911_6 := arithEq_of_rows h (row := 1911) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_6
  simp only [← c116670, ← c116671, ← c116672] at e_1911_6
  have e_1911_7 := arithEq_of_rows h (row := 1911) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_7
  simp only [← c116673, ← c116674, k2, ← c116675] at e_1911_7
  have hr := e_1911_7
  simp only [e_1911_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18695 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1911 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8991)) := by
  have c116676 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.1
  have c116677 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.1
  have c116678 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.1
  have c116679 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.1
  have c116680 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.1
  have c116681 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1911_8 := arithEq_of_rows h (row := 1911) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_8
  simp only [← c116676, ← c116677, ← c116678] at e_1911_8
  have e_1911_9 := arithEq_of_rows h (row := 1911) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_9
  simp only [← c116679, ← c116680, k2, ← c116681] at e_1911_9
  have hr := e_1911_9
  simp only [e_1911_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18696 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1911 47) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8992)) := by
  have c116682 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116683 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116684 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116685 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116686 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116687 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1911_10 := arithEq_of_rows h (row := 1911) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_10
  simp only [← c116682, ← c116683, ← c116684] at e_1911_10
  have e_1911_11 := arithEq_of_rows h (row := 1911) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_11
  simp only [← c116685, ← c116686, k2, ← c116687] at e_1911_11
  have hr := e_1911_11
  simp only [e_1911_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18697 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1911 55) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8993)) := by
  have c116688 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116689 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116690 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116691 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116692 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116693 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1911_12 := arithEq_of_rows h (row := 1911) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_12
  simp only [← c116688, ← c116689, ← c116690] at e_1911_12
  have e_1911_13 := arithEq_of_rows h (row := 1911) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_13
  simp only [← c116691, ← c116692, k2, ← c116693] at e_1911_13
  have hr := e_1911_13
  simp only [e_1911_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18698 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1911 63) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8994)) := by
  have c116694 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116695 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116696 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116697 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116698 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116699 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1911_14 := arithEq_of_rows h (row := 1911) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_14
  simp only [← c116694, ← c116695, ← c116696] at e_1911_14
  have e_1911_15 := arithEq_of_rows h (row := 1911) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_15
  simp only [← c116697, ← c116698, k2, ← c116699] at e_1911_15
  have hr := e_1911_15
  simp only [e_1911_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18699 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1911 71) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8995)) := by
  have c116700 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116701 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116702 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116703 := (publicBatchWrapper89_copies3646 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116704 := (publicBatchWrapper89_copies3647 a h).1
  have c116705 := (publicBatchWrapper89_copies3647 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1911_16 := arithEq_of_rows h (row := 1911) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_16
  simp only [← c116700, ← c116701, ← c116702] at e_1911_16
  have e_1911_17 := arithEq_of_rows h (row := 1911) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_17
  simp only [← c116703, ← c116704, k2, ← c116705] at e_1911_17
  have hr := e_1911_17
  simp only [e_1911_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18700 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1911 79) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8996)) := by
  have c116706 := (publicBatchWrapper89_copies3647 a h).2.2.1
  have c116707 := (publicBatchWrapper89_copies3647 a h).2.2.2.1
  have c116708 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.1
  have c116709 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.1
  have c116710 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.1
  have c116711 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1911_18 := arithEq_of_rows h (row := 1911) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_18
  simp only [← c116706, ← c116707, ← c116708] at e_1911_18
  have e_1911_19 := arithEq_of_rows h (row := 1911) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1911_19
  simp only [← c116709, ← c116710, k2, ← c116711] at e_1911_19
  have hr := e_1911_19
  simp only [e_1911_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18701 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1912 7) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8997)) := by
  have c116712 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.1
  have c116713 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.1
  have c116714 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116715 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116716 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116717 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1912_0 := arithEq_of_rows h (row := 1912) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_0
  simp only [← c116712, ← c116713, ← c116714] at e_1912_0
  have e_1912_1 := arithEq_of_rows h (row := 1912) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_1
  simp only [← c116715, ← c116716, k2, ← c116717] at e_1912_1
  have hr := e_1912_1
  simp only [e_1912_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18702 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1912 15) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8998)) := by
  have c116718 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116719 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116720 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116721 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116722 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116723 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1912_2 := arithEq_of_rows h (row := 1912) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_2
  simp only [← c116718, ← c116719, ← c116720] at e_1912_2
  have e_1912_3 := arithEq_of_rows h (row := 1912) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_3
  simp only [← c116721, ← c116722, k2, ← c116723] at e_1912_3
  have hr := e_1912_3
  simp only [e_1912_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18703 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1912 23) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 8999)) := by
  have c116724 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116725 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116726 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116727 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116728 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116729 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1912_4 := arithEq_of_rows h (row := 1912) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_4
  simp only [← c116724, ← c116725, ← c116726] at e_1912_4
  have e_1912_5 := arithEq_of_rows h (row := 1912) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_5
  simp only [← c116727, ← c116728, k2, ← c116729] at e_1912_5
  have hr := e_1912_5
  simp only [e_1912_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18704 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1912 31) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 9000)) := by
  have c116730 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116731 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116732 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116733 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116734 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116735 := (publicBatchWrapper89_copies3647 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1912_6 := arithEq_of_rows h (row := 1912) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_6
  simp only [← c116730, ← c116731, ← c116732] at e_1912_6
  have e_1912_7 := arithEq_of_rows h (row := 1912) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_7
  simp only [← c116733, ← c116734, k2, ← c116735] at e_1912_7
  have hr := e_1912_7
  simp only [e_1912_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18705 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1912 39) = bselect (a (.wire 43 75)) (a (.virt 200)) (a (.virt 9001)) := by
  have c116736 := (publicBatchWrapper89_copies3648 a h).1
  have c116737 := (publicBatchWrapper89_copies3648 a h).2.1
  have c116738 := (publicBatchWrapper89_copies3648 a h).2.2.1
  have c116739 := (publicBatchWrapper89_copies3648 a h).2.2.2.1
  have c116740 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.1
  have c116741 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1912_8 := arithEq_of_rows h (row := 1912) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_8
  simp only [← c116736, ← c116737, ← c116738] at e_1912_8
  have e_1912_9 := arithEq_of_rows h (row := 1912) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_9
  simp only [← c116739, ← c116740, k2, ← c116741] at e_1912_9
  have hr := e_1912_9
  simp only [e_1912_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18706 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1912 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9170)) := by
  have c116742 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.1
  have c116743 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.1
  have c116744 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.1
  have c116745 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.1
  have c116746 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116747 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1912_10 := arithEq_of_rows h (row := 1912) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_10
  simp only [← c116742, ← c116743, ← c116744] at e_1912_10
  have e_1912_11 := arithEq_of_rows h (row := 1912) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_11
  simp only [← c116745, ← c116746, k2, ← c116747] at e_1912_11
  have hr := e_1912_11
  simp only [e_1912_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18707 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1912 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9171)) := by
  have c116748 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116749 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116750 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116751 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116752 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116753 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1912_12 := arithEq_of_rows h (row := 1912) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_12
  simp only [← c116748, ← c116749, ← c116750] at e_1912_12
  have e_1912_13 := arithEq_of_rows h (row := 1912) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_13
  simp only [← c116751, ← c116752, k2, ← c116753] at e_1912_13
  have hr := e_1912_13
  simp only [e_1912_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18708 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1912 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9172)) := by
  have c116754 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116755 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116756 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116757 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116758 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116759 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1912_14 := arithEq_of_rows h (row := 1912) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_14
  simp only [← c116754, ← c116755, ← c116756] at e_1912_14
  have e_1912_15 := arithEq_of_rows h (row := 1912) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_15
  simp only [← c116757, ← c116758, k2, ← c116759] at e_1912_15
  have hr := e_1912_15
  simp only [e_1912_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18709 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1912 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9173)) := by
  have c116760 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116761 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116762 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116763 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116764 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116765 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1912_16 := arithEq_of_rows h (row := 1912) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_16
  simp only [← c116760, ← c116761, ← c116762] at e_1912_16
  have e_1912_17 := arithEq_of_rows h (row := 1912) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_17
  simp only [← c116763, ← c116764, k2, ← c116765] at e_1912_17
  have hr := e_1912_17
  simp only [e_1912_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18710 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1912 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9174)) := by
  have c116766 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116767 := (publicBatchWrapper89_copies3648 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116768 := (publicBatchWrapper89_copies3649 a h).1
  have c116769 := (publicBatchWrapper89_copies3649 a h).2.1
  have c116770 := (publicBatchWrapper89_copies3649 a h).2.2.1
  have c116771 := (publicBatchWrapper89_copies3649 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1912_18 := arithEq_of_rows h (row := 1912) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_18
  simp only [← c116766, ← c116767, ← c116768] at e_1912_18
  have e_1912_19 := arithEq_of_rows h (row := 1912) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1912_19
  simp only [← c116769, ← c116770, k2, ← c116771] at e_1912_19
  have hr := e_1912_19
  simp only [e_1912_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18711 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1913 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9175)) := by
  have c116772 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.1
  have c116773 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.1
  have c116774 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.1
  have c116775 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.1
  have c116776 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.1
  have c116777 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1913_0 := arithEq_of_rows h (row := 1913) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_0
  simp only [← c116772, ← c116773, ← c116774] at e_1913_0
  have e_1913_1 := arithEq_of_rows h (row := 1913) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_1
  simp only [← c116775, ← c116776, k2, ← c116777] at e_1913_1
  have hr := e_1913_1
  simp only [e_1913_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18712 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1913 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9176)) := by
  have c116778 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116779 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116780 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116781 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116782 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116783 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1913_2 := arithEq_of_rows h (row := 1913) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_2
  simp only [← c116778, ← c116779, ← c116780] at e_1913_2
  have e_1913_3 := arithEq_of_rows h (row := 1913) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_3
  simp only [← c116781, ← c116782, k2, ← c116783] at e_1913_3
  have hr := e_1913_3
  simp only [e_1913_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18713 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1913 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9177)) := by
  have c116784 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116785 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116786 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116787 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116788 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116789 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1913_4 := arithEq_of_rows h (row := 1913) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_4
  simp only [← c116784, ← c116785, ← c116786] at e_1913_4
  have e_1913_5 := arithEq_of_rows h (row := 1913) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_5
  simp only [← c116787, ← c116788, k2, ← c116789] at e_1913_5
  have hr := e_1913_5
  simp only [e_1913_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18714 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1913 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9178)) := by
  have c116790 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116791 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116792 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116793 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116794 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116795 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1913_6 := arithEq_of_rows h (row := 1913) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_6
  simp only [← c116790, ← c116791, ← c116792] at e_1913_6
  have e_1913_7 := arithEq_of_rows h (row := 1913) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_7
  simp only [← c116793, ← c116794, k2, ← c116795] at e_1913_7
  have hr := e_1913_7
  simp only [e_1913_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18715 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1913 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9179)) := by
  have c116796 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116797 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116798 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116799 := (publicBatchWrapper89_copies3649 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116800 := (publicBatchWrapper89_copies3650 a h).1
  have c116801 := (publicBatchWrapper89_copies3650 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1913_8 := arithEq_of_rows h (row := 1913) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_8
  simp only [← c116796, ← c116797, ← c116798] at e_1913_8
  have e_1913_9 := arithEq_of_rows h (row := 1913) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_9
  simp only [← c116799, ← c116800, k2, ← c116801] at e_1913_9
  have hr := e_1913_9
  simp only [e_1913_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18716 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1913 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9180)) := by
  have c116802 := (publicBatchWrapper89_copies3650 a h).2.2.1
  have c116803 := (publicBatchWrapper89_copies3650 a h).2.2.2.1
  have c116804 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.1
  have c116805 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.1
  have c116806 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.1
  have c116807 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1913_10 := arithEq_of_rows h (row := 1913) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_10
  simp only [← c116802, ← c116803, ← c116804] at e_1913_10
  have e_1913_11 := arithEq_of_rows h (row := 1913) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_11
  simp only [← c116805, ← c116806, k2, ← c116807] at e_1913_11
  have hr := e_1913_11
  simp only [e_1913_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18717 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1913 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9181)) := by
  have c116808 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.1
  have c116809 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.1
  have c116810 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116811 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116812 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116813 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1913_12 := arithEq_of_rows h (row := 1913) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_12
  simp only [← c116808, ← c116809, ← c116810] at e_1913_12
  have e_1913_13 := arithEq_of_rows h (row := 1913) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_13
  simp only [← c116811, ← c116812, k2, ← c116813] at e_1913_13
  have hr := e_1913_13
  simp only [e_1913_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18718 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1913 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9182)) := by
  have c116814 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116815 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116816 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116817 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116818 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116819 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1913_14 := arithEq_of_rows h (row := 1913) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_14
  simp only [← c116814, ← c116815, ← c116816] at e_1913_14
  have e_1913_15 := arithEq_of_rows h (row := 1913) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_15
  simp only [← c116817, ← c116818, k2, ← c116819] at e_1913_15
  have hr := e_1913_15
  simp only [e_1913_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18719 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1913 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9183)) := by
  have c116820 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116821 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116822 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116823 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116824 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116825 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1913_16 := arithEq_of_rows h (row := 1913) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_16
  simp only [← c116820, ← c116821, ← c116822] at e_1913_16
  have e_1913_17 := arithEq_of_rows h (row := 1913) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_17
  simp only [← c116823, ← c116824, k2, ← c116825] at e_1913_17
  have hr := e_1913_17
  simp only [e_1913_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18720 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1913 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9184)) := by
  have c116826 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116827 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116828 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116829 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116830 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116831 := (publicBatchWrapper89_copies3650 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1913_18 := arithEq_of_rows h (row := 1913) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_18
  simp only [← c116826, ← c116827, ← c116828] at e_1913_18
  have e_1913_19 := arithEq_of_rows h (row := 1913) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1913_19
  simp only [← c116829, ← c116830, k2, ← c116831] at e_1913_19
  have hr := e_1913_19
  simp only [e_1913_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18721 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1914 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9185)) := by
  have c116832 := (publicBatchWrapper89_copies3651 a h).1
  have c116833 := (publicBatchWrapper89_copies3651 a h).2.1
  have c116834 := (publicBatchWrapper89_copies3651 a h).2.2.1
  have c116835 := (publicBatchWrapper89_copies3651 a h).2.2.2.1
  have c116836 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.1
  have c116837 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1914_0 := arithEq_of_rows h (row := 1914) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_0
  simp only [← c116832, ← c116833, ← c116834] at e_1914_0
  have e_1914_1 := arithEq_of_rows h (row := 1914) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_1
  simp only [← c116835, ← c116836, k2, ← c116837] at e_1914_1
  have hr := e_1914_1
  simp only [e_1914_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18722 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1914 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9186)) := by
  have c116838 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.1
  have c116839 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.1
  have c116840 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.1
  have c116841 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.1
  have c116842 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116843 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1914_2 := arithEq_of_rows h (row := 1914) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_2
  simp only [← c116838, ← c116839, ← c116840] at e_1914_2
  have e_1914_3 := arithEq_of_rows h (row := 1914) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_3
  simp only [← c116841, ← c116842, k2, ← c116843] at e_1914_3
  have hr := e_1914_3
  simp only [e_1914_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18723 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1914 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9187)) := by
  have c116844 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116845 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116846 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116847 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116848 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116849 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1914_4 := arithEq_of_rows h (row := 1914) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_4
  simp only [← c116844, ← c116845, ← c116846] at e_1914_4
  have e_1914_5 := arithEq_of_rows h (row := 1914) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_5
  simp only [← c116847, ← c116848, k2, ← c116849] at e_1914_5
  have hr := e_1914_5
  simp only [e_1914_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18724 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1914 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9188)) := by
  have c116850 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116851 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116852 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116853 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116854 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116855 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1914_6 := arithEq_of_rows h (row := 1914) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_6
  simp only [← c116850, ← c116851, ← c116852] at e_1914_6
  have e_1914_7 := arithEq_of_rows h (row := 1914) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_7
  simp only [← c116853, ← c116854, k2, ← c116855] at e_1914_7
  have hr := e_1914_7
  simp only [e_1914_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18725 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1914 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9189)) := by
  have c116856 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116857 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116858 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116859 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116860 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116861 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1914_8 := arithEq_of_rows h (row := 1914) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_8
  simp only [← c116856, ← c116857, ← c116858] at e_1914_8
  have e_1914_9 := arithEq_of_rows h (row := 1914) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_9
  simp only [← c116859, ← c116860, k2, ← c116861] at e_1914_9
  have hr := e_1914_9
  simp only [e_1914_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18726 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1914 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9190)) := by
  have c116862 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116863 := (publicBatchWrapper89_copies3651 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116864 := (publicBatchWrapper89_copies3652 a h).1
  have c116865 := (publicBatchWrapper89_copies3652 a h).2.1
  have c116866 := (publicBatchWrapper89_copies3652 a h).2.2.1
  have c116867 := (publicBatchWrapper89_copies3652 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1914_10 := arithEq_of_rows h (row := 1914) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_10
  simp only [← c116862, ← c116863, ← c116864] at e_1914_10
  have e_1914_11 := arithEq_of_rows h (row := 1914) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_11
  simp only [← c116865, ← c116866, k2, ← c116867] at e_1914_11
  have hr := e_1914_11
  simp only [e_1914_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18727 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1914 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9191)) := by
  have c116868 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.1
  have c116869 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.1
  have c116870 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.1
  have c116871 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.1
  have c116872 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.1
  have c116873 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1914_12 := arithEq_of_rows h (row := 1914) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_12
  simp only [← c116868, ← c116869, ← c116870] at e_1914_12
  have e_1914_13 := arithEq_of_rows h (row := 1914) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_13
  simp only [← c116871, ← c116872, k2, ← c116873] at e_1914_13
  have hr := e_1914_13
  simp only [e_1914_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18728 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1914 63) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9192)) := by
  have c116874 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116875 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116876 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116877 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116878 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116879 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1914_14 := arithEq_of_rows h (row := 1914) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_14
  simp only [← c116874, ← c116875, ← c116876] at e_1914_14
  have e_1914_15 := arithEq_of_rows h (row := 1914) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_15
  simp only [← c116877, ← c116878, k2, ← c116879] at e_1914_15
  have hr := e_1914_15
  simp only [e_1914_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18729 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1914 71) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9193)) := by
  have c116880 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116881 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116882 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116883 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116884 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116885 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1914_16 := arithEq_of_rows h (row := 1914) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_16
  simp only [← c116880, ← c116881, ← c116882] at e_1914_16
  have e_1914_17 := arithEq_of_rows h (row := 1914) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_17
  simp only [← c116883, ← c116884, k2, ← c116885] at e_1914_17
  have hr := e_1914_17
  simp only [e_1914_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18730 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1914 79) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9194)) := by
  have c116886 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116887 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116888 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116889 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116890 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116891 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1914_18 := arithEq_of_rows h (row := 1914) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_18
  simp only [← c116886, ← c116887, ← c116888] at e_1914_18
  have e_1914_19 := arithEq_of_rows h (row := 1914) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1914_19
  simp only [← c116889, ← c116890, k2, ← c116891] at e_1914_19
  have hr := e_1914_19
  simp only [e_1914_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18731 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1915 7) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9195)) := by
  have c116892 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116893 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116894 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116895 := (publicBatchWrapper89_copies3652 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116896 := (publicBatchWrapper89_copies3653 a h).1
  have c116897 := (publicBatchWrapper89_copies3653 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1915_0 := arithEq_of_rows h (row := 1915) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_0
  simp only [← c116892, ← c116893, ← c116894] at e_1915_0
  have e_1915_1 := arithEq_of_rows h (row := 1915) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_1
  simp only [← c116895, ← c116896, k2, ← c116897] at e_1915_1
  have hr := e_1915_1
  simp only [e_1915_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18732 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1915 15) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9196)) := by
  have c116898 := (publicBatchWrapper89_copies3653 a h).2.2.1
  have c116899 := (publicBatchWrapper89_copies3653 a h).2.2.2.1
  have c116900 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.1
  have c116901 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.1
  have c116902 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.1
  have c116903 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1915_2 := arithEq_of_rows h (row := 1915) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_2
  simp only [← c116898, ← c116899, ← c116900] at e_1915_2
  have e_1915_3 := arithEq_of_rows h (row := 1915) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_3
  simp only [← c116901, ← c116902, k2, ← c116903] at e_1915_3
  have hr := e_1915_3
  simp only [e_1915_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18733 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1915 23) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9197)) := by
  have c116904 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.1
  have c116905 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.1
  have c116906 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116907 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116908 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116909 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1915_4 := arithEq_of_rows h (row := 1915) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_4
  simp only [← c116904, ← c116905, ← c116906] at e_1915_4
  have e_1915_5 := arithEq_of_rows h (row := 1915) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_5
  simp only [← c116907, ← c116908, k2, ← c116909] at e_1915_5
  have hr := e_1915_5
  simp only [e_1915_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18734 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1915 31) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9198)) := by
  have c116910 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116911 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116912 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116913 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116914 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116915 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1915_6 := arithEq_of_rows h (row := 1915) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_6
  simp only [← c116910, ← c116911, ← c116912] at e_1915_6
  have e_1915_7 := arithEq_of_rows h (row := 1915) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_7
  simp only [← c116913, ← c116914, k2, ← c116915] at e_1915_7
  have hr := e_1915_7
  simp only [e_1915_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18735 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1915 39) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9199)) := by
  have c116916 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116917 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116918 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116919 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116920 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116921 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1915_8 := arithEq_of_rows h (row := 1915) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_8
  simp only [← c116916, ← c116917, ← c116918] at e_1915_8
  have e_1915_9 := arithEq_of_rows h (row := 1915) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_9
  simp only [← c116919, ← c116920, k2, ← c116921] at e_1915_9
  have hr := e_1915_9
  simp only [e_1915_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18736 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1915 47) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9200)) := by
  have c116922 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116923 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116924 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116925 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116926 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116927 := (publicBatchWrapper89_copies3653 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1915_10 := arithEq_of_rows h (row := 1915) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_10
  simp only [← c116922, ← c116923, ← c116924] at e_1915_10
  have e_1915_11 := arithEq_of_rows h (row := 1915) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_11
  simp only [← c116925, ← c116926, k2, ← c116927] at e_1915_11
  have hr := e_1915_11
  simp only [e_1915_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18737 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1915 55) = bselect (a (.wire 45 39)) (a (.virt 200)) (a (.virt 9201)) := by
  have c116928 := (publicBatchWrapper89_copies3654 a h).1
  have c116929 := (publicBatchWrapper89_copies3654 a h).2.1
  have c116930 := (publicBatchWrapper89_copies3654 a h).2.2.1
  have c116931 := (publicBatchWrapper89_copies3654 a h).2.2.2.1
  have c116932 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.1
  have c116933 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1915_12 := arithEq_of_rows h (row := 1915) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_12
  simp only [← c116928, ← c116929, ← c116930] at e_1915_12
  have e_1915_13 := arithEq_of_rows h (row := 1915) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_13
  simp only [← c116931, ← c116932, k2, ← c116933] at e_1915_13
  have hr := e_1915_13
  simp only [e_1915_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18738 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1915 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9370)) := by
  have c116934 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.1
  have c116935 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.1
  have c116936 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.1
  have c116937 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.1
  have c116938 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116939 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1915_14 := arithEq_of_rows h (row := 1915) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_14
  simp only [← c116934, ← c116935, ← c116936] at e_1915_14
  have e_1915_15 := arithEq_of_rows h (row := 1915) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_15
  simp only [← c116937, ← c116938, k2, ← c116939] at e_1915_15
  have hr := e_1915_15
  simp only [e_1915_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18739 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1915 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9371)) := by
  have c116940 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116941 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116942 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116943 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116944 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116945 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1915_16 := arithEq_of_rows h (row := 1915) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_16
  simp only [← c116940, ← c116941, ← c116942] at e_1915_16
  have e_1915_17 := arithEq_of_rows h (row := 1915) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_17
  simp only [← c116943, ← c116944, k2, ← c116945] at e_1915_17
  have hr := e_1915_17
  simp only [e_1915_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18740 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1915 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9372)) := by
  have c116946 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116947 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116948 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116949 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116950 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116951 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1915_18 := arithEq_of_rows h (row := 1915) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_18
  simp only [← c116946, ← c116947, ← c116948] at e_1915_18
  have e_1915_19 := arithEq_of_rows h (row := 1915) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1915_19
  simp only [← c116949, ← c116950, k2, ← c116951] at e_1915_19
  have hr := e_1915_19
  simp only [e_1915_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18741 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1916 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9373)) := by
  have c116952 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116953 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116954 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116955 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116956 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116957 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1916_0 := arithEq_of_rows h (row := 1916) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_0
  simp only [← c116952, ← c116953, ← c116954] at e_1916_0
  have e_1916_1 := arithEq_of_rows h (row := 1916) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_1
  simp only [← c116955, ← c116956, k2, ← c116957] at e_1916_1
  have hr := e_1916_1
  simp only [e_1916_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18742 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1916 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9374)) := by
  have c116958 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116959 := (publicBatchWrapper89_copies3654 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116960 := (publicBatchWrapper89_copies3655 a h).1
  have c116961 := (publicBatchWrapper89_copies3655 a h).2.1
  have c116962 := (publicBatchWrapper89_copies3655 a h).2.2.1
  have c116963 := (publicBatchWrapper89_copies3655 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1916_2 := arithEq_of_rows h (row := 1916) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_2
  simp only [← c116958, ← c116959, ← c116960] at e_1916_2
  have e_1916_3 := arithEq_of_rows h (row := 1916) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_3
  simp only [← c116961, ← c116962, k2, ← c116963] at e_1916_3
  have hr := e_1916_3
  simp only [e_1916_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18743 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1916 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9375)) := by
  have c116964 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.1
  have c116965 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.1
  have c116966 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.1
  have c116967 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.1
  have c116968 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.1
  have c116969 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1916_4 := arithEq_of_rows h (row := 1916) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_4
  simp only [← c116964, ← c116965, ← c116966] at e_1916_4
  have e_1916_5 := arithEq_of_rows h (row := 1916) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_5
  simp only [← c116967, ← c116968, k2, ← c116969] at e_1916_5
  have hr := e_1916_5
  simp only [e_1916_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18744 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1916 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9376)) := by
  have c116970 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.1
  have c116971 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c116972 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116973 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116974 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116975 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1916_6 := arithEq_of_rows h (row := 1916) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_6
  simp only [← c116970, ← c116971, ← c116972] at e_1916_6
  have e_1916_7 := arithEq_of_rows h (row := 1916) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_7
  simp only [← c116973, ← c116974, k2, ← c116975] at e_1916_7
  have hr := e_1916_7
  simp only [e_1916_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18745 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1916 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9377)) := by
  have c116976 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116977 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116978 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116979 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116980 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116981 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1916_8 := arithEq_of_rows h (row := 1916) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_8
  simp only [← c116976, ← c116977, ← c116978] at e_1916_8
  have e_1916_9 := arithEq_of_rows h (row := 1916) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_9
  simp only [← c116979, ← c116980, k2, ← c116981] at e_1916_9
  have hr := e_1916_9
  simp only [e_1916_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18746 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1916 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9378)) := by
  have c116982 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116983 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116984 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116985 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116986 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116987 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1916_10 := arithEq_of_rows h (row := 1916) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_10
  simp only [← c116982, ← c116983, ← c116984] at e_1916_10
  have e_1916_11 := arithEq_of_rows h (row := 1916) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_11
  simp only [← c116985, ← c116986, k2, ← c116987] at e_1916_11
  have hr := e_1916_11
  simp only [e_1916_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18747 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1916 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9379)) := by
  have c116988 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116989 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116990 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c116991 := (publicBatchWrapper89_copies3655 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c116992 := (publicBatchWrapper89_copies3656 a h).1
  have c116993 := (publicBatchWrapper89_copies3656 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1916_12 := arithEq_of_rows h (row := 1916) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_12
  simp only [← c116988, ← c116989, ← c116990] at e_1916_12
  have e_1916_13 := arithEq_of_rows h (row := 1916) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_13
  simp only [← c116991, ← c116992, k2, ← c116993] at e_1916_13
  have hr := e_1916_13
  simp only [e_1916_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18748 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1916 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9380)) := by
  have c116994 := (publicBatchWrapper89_copies3656 a h).2.2.1
  have c116995 := (publicBatchWrapper89_copies3656 a h).2.2.2.1
  have c116996 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.1
  have c116997 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.1
  have c116998 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.1
  have c116999 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1916_14 := arithEq_of_rows h (row := 1916) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_14
  simp only [← c116994, ← c116995, ← c116996] at e_1916_14
  have e_1916_15 := arithEq_of_rows h (row := 1916) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_15
  simp only [← c116997, ← c116998, k2, ← c116999] at e_1916_15
  have hr := e_1916_15
  simp only [e_1916_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18749 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1916 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9381)) := by
  have c117000 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.1
  have c117001 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.1
  have c117002 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117003 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117004 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117005 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1916_16 := arithEq_of_rows h (row := 1916) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_16
  simp only [← c117000, ← c117001, ← c117002] at e_1916_16
  have e_1916_17 := arithEq_of_rows h (row := 1916) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_17
  simp only [← c117003, ← c117004, k2, ← c117005] at e_1916_17
  have hr := e_1916_17
  simp only [e_1916_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18750 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1916 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9382)) := by
  have c117006 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117007 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117008 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117009 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117010 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117011 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1916_18 := arithEq_of_rows h (row := 1916) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_18
  simp only [← c117006, ← c117007, ← c117008] at e_1916_18
  have e_1916_19 := arithEq_of_rows h (row := 1916) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1916_19
  simp only [← c117009, ← c117010, k2, ← c117011] at e_1916_19
  have hr := e_1916_19
  simp only [e_1916_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18751 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1917 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9383)) := by
  have c117012 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117013 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117014 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117015 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117016 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117017 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1917_0 := arithEq_of_rows h (row := 1917) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_0
  simp only [← c117012, ← c117013, ← c117014] at e_1917_0
  have e_1917_1 := arithEq_of_rows h (row := 1917) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_1
  simp only [← c117015, ← c117016, k2, ← c117017] at e_1917_1
  have hr := e_1917_1
  simp only [e_1917_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18752 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1917 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9384)) := by
  have c117018 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117019 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117020 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117021 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117022 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117023 := (publicBatchWrapper89_copies3656 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1917_2 := arithEq_of_rows h (row := 1917) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_2
  simp only [← c117018, ← c117019, ← c117020] at e_1917_2
  have e_1917_3 := arithEq_of_rows h (row := 1917) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_3
  simp only [← c117021, ← c117022, k2, ← c117023] at e_1917_3
  have hr := e_1917_3
  simp only [e_1917_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18753 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1917 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9385)) := by
  have c117024 := (publicBatchWrapper89_copies3657 a h).1
  have c117025 := (publicBatchWrapper89_copies3657 a h).2.1
  have c117026 := (publicBatchWrapper89_copies3657 a h).2.2.1
  have c117027 := (publicBatchWrapper89_copies3657 a h).2.2.2.1
  have c117028 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.1
  have c117029 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1917_4 := arithEq_of_rows h (row := 1917) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_4
  simp only [← c117024, ← c117025, ← c117026] at e_1917_4
  have e_1917_5 := arithEq_of_rows h (row := 1917) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_5
  simp only [← c117027, ← c117028, k2, ← c117029] at e_1917_5
  have hr := e_1917_5
  simp only [e_1917_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18754 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1917 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9386)) := by
  have c117030 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.1
  have c117031 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.1
  have c117032 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.1
  have c117033 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.1
  have c117034 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117035 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1917_6 := arithEq_of_rows h (row := 1917) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_6
  simp only [← c117030, ← c117031, ← c117032] at e_1917_6
  have e_1917_7 := arithEq_of_rows h (row := 1917) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_7
  simp only [← c117033, ← c117034, k2, ← c117035] at e_1917_7
  have hr := e_1917_7
  simp only [e_1917_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18755 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1917 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9387)) := by
  have c117036 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117037 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117038 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117039 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117040 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117041 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1917_8 := arithEq_of_rows h (row := 1917) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_8
  simp only [← c117036, ← c117037, ← c117038] at e_1917_8
  have e_1917_9 := arithEq_of_rows h (row := 1917) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_9
  simp only [← c117039, ← c117040, k2, ← c117041] at e_1917_9
  have hr := e_1917_9
  simp only [e_1917_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18756 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1917 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9388)) := by
  have c117042 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117043 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117044 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117045 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117046 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117047 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1917_10 := arithEq_of_rows h (row := 1917) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_10
  simp only [← c117042, ← c117043, ← c117044] at e_1917_10
  have e_1917_11 := arithEq_of_rows h (row := 1917) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_11
  simp only [← c117045, ← c117046, k2, ← c117047] at e_1917_11
  have hr := e_1917_11
  simp only [e_1917_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18757 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1917 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9389)) := by
  have c117048 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117049 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117050 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117051 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117052 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117053 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1917_12 := arithEq_of_rows h (row := 1917) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_12
  simp only [← c117048, ← c117049, ← c117050] at e_1917_12
  have e_1917_13 := arithEq_of_rows h (row := 1917) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_13
  simp only [← c117051, ← c117052, k2, ← c117053] at e_1917_13
  have hr := e_1917_13
  simp only [e_1917_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18758 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1917 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9390)) := by
  have c117054 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117055 := (publicBatchWrapper89_copies3657 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117056 := (publicBatchWrapper89_copies3658 a h).1
  have c117057 := (publicBatchWrapper89_copies3658 a h).2.1
  have c117058 := (publicBatchWrapper89_copies3658 a h).2.2.1
  have c117059 := (publicBatchWrapper89_copies3658 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1917_14 := arithEq_of_rows h (row := 1917) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_14
  simp only [← c117054, ← c117055, ← c117056] at e_1917_14
  have e_1917_15 := arithEq_of_rows h (row := 1917) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_15
  simp only [← c117057, ← c117058, k2, ← c117059] at e_1917_15
  have hr := e_1917_15
  simp only [e_1917_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18759 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1917 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9391)) := by
  have c117060 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.1
  have c117061 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.1
  have c117062 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.1
  have c117063 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.1
  have c117064 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.1
  have c117065 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1917_16 := arithEq_of_rows h (row := 1917) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_16
  simp only [← c117060, ← c117061, ← c117062] at e_1917_16
  have e_1917_17 := arithEq_of_rows h (row := 1917) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_17
  simp only [← c117063, ← c117064, k2, ← c117065] at e_1917_17
  have hr := e_1917_17
  simp only [e_1917_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18760 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1917 79) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9392)) := by
  have c117066 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117067 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117068 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117069 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117070 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117071 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1917_18 := arithEq_of_rows h (row := 1917) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_18
  simp only [← c117066, ← c117067, ← c117068] at e_1917_18
  have e_1917_19 := arithEq_of_rows h (row := 1917) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1917_19
  simp only [← c117069, ← c117070, k2, ← c117071] at e_1917_19
  have hr := e_1917_19
  simp only [e_1917_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18761 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1918 7) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9393)) := by
  have c117072 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117073 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117074 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117075 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117076 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117077 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1918_0 := arithEq_of_rows h (row := 1918) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_0
  simp only [← c117072, ← c117073, ← c117074] at e_1918_0
  have e_1918_1 := arithEq_of_rows h (row := 1918) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_1
  simp only [← c117075, ← c117076, k2, ← c117077] at e_1918_1
  have hr := e_1918_1
  simp only [e_1918_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18762 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1918 15) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9394)) := by
  have c117078 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117079 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117080 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117081 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117082 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117083 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1918_2 := arithEq_of_rows h (row := 1918) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_2
  simp only [← c117078, ← c117079, ← c117080] at e_1918_2
  have e_1918_3 := arithEq_of_rows h (row := 1918) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_3
  simp only [← c117081, ← c117082, k2, ← c117083] at e_1918_3
  have hr := e_1918_3
  simp only [e_1918_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18763 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1918 23) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9395)) := by
  have c117084 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117085 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117086 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117087 := (publicBatchWrapper89_copies3658 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117088 := (publicBatchWrapper89_copies3659 a h).1
  have c117089 := (publicBatchWrapper89_copies3659 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1918_4 := arithEq_of_rows h (row := 1918) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_4
  simp only [← c117084, ← c117085, ← c117086] at e_1918_4
  have e_1918_5 := arithEq_of_rows h (row := 1918) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_5
  simp only [← c117087, ← c117088, k2, ← c117089] at e_1918_5
  have hr := e_1918_5
  simp only [e_1918_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18764 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1918 31) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9396)) := by
  have c117090 := (publicBatchWrapper89_copies3659 a h).2.2.1
  have c117091 := (publicBatchWrapper89_copies3659 a h).2.2.2.1
  have c117092 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.1
  have c117093 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.1
  have c117094 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.1
  have c117095 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1918_6 := arithEq_of_rows h (row := 1918) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_6
  simp only [← c117090, ← c117091, ← c117092] at e_1918_6
  have e_1918_7 := arithEq_of_rows h (row := 1918) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_7
  simp only [← c117093, ← c117094, k2, ← c117095] at e_1918_7
  have hr := e_1918_7
  simp only [e_1918_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18765 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1918 39) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9397)) := by
  have c117096 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.1
  have c117097 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.1
  have c117098 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117099 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117100 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117101 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1918_8 := arithEq_of_rows h (row := 1918) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_8
  simp only [← c117096, ← c117097, ← c117098] at e_1918_8
  have e_1918_9 := arithEq_of_rows h (row := 1918) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_9
  simp only [← c117099, ← c117100, k2, ← c117101] at e_1918_9
  have hr := e_1918_9
  simp only [e_1918_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18766 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1918 47) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9398)) := by
  have c117102 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117103 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117104 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117105 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117106 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117107 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1918_10 := arithEq_of_rows h (row := 1918) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_10
  simp only [← c117102, ← c117103, ← c117104] at e_1918_10
  have e_1918_11 := arithEq_of_rows h (row := 1918) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_11
  simp only [← c117105, ← c117106, k2, ← c117107] at e_1918_11
  have hr := e_1918_11
  simp only [e_1918_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18767 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1918 55) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9399)) := by
  have c117108 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117109 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117110 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117111 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117112 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117113 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1918_12 := arithEq_of_rows h (row := 1918) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_12
  simp only [← c117108, ← c117109, ← c117110] at e_1918_12
  have e_1918_13 := arithEq_of_rows h (row := 1918) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_13
  simp only [← c117111, ← c117112, k2, ← c117113] at e_1918_13
  have hr := e_1918_13
  simp only [e_1918_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18768 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1918 63) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9400)) := by
  have c117114 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117115 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117116 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117117 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117118 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117119 := (publicBatchWrapper89_copies3659 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1918_14 := arithEq_of_rows h (row := 1918) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_14
  simp only [← c117114, ← c117115, ← c117116] at e_1918_14
  have e_1918_15 := arithEq_of_rows h (row := 1918) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_15
  simp only [← c117117, ← c117118, k2, ← c117119] at e_1918_15
  have hr := e_1918_15
  simp only [e_1918_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18769 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1918 71) = bselect (a (.wire 46 3)) (a (.virt 200)) (a (.virt 9401)) := by
  have c117120 := (publicBatchWrapper89_copies3660 a h).1
  have c117121 := (publicBatchWrapper89_copies3660 a h).2.1
  have c117122 := (publicBatchWrapper89_copies3660 a h).2.2.1
  have c117123 := (publicBatchWrapper89_copies3660 a h).2.2.2.1
  have c117124 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.1
  have c117125 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1918_16 := arithEq_of_rows h (row := 1918) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_16
  simp only [← c117120, ← c117121, ← c117122] at e_1918_16
  have e_1918_17 := arithEq_of_rows h (row := 1918) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_17
  simp only [← c117123, ← c117124, k2, ← c117125] at e_1918_17
  have hr := e_1918_17
  simp only [e_1918_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18770 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1918 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9570)) := by
  have c117126 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.1
  have c117127 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.1
  have c117128 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.1
  have c117129 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.1
  have c117130 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117131 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1918_18 := arithEq_of_rows h (row := 1918) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_18
  simp only [← c117126, ← c117127, ← c117128] at e_1918_18
  have e_1918_19 := arithEq_of_rows h (row := 1918) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1918_19
  simp only [← c117129, ← c117130, k2, ← c117131] at e_1918_19
  have hr := e_1918_19
  simp only [e_1918_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18771 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1919 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9571)) := by
  have c117132 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117133 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117134 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117135 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117136 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117137 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1919_0 := arithEq_of_rows h (row := 1919) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_0
  simp only [← c117132, ← c117133, ← c117134] at e_1919_0
  have e_1919_1 := arithEq_of_rows h (row := 1919) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_1
  simp only [← c117135, ← c117136, k2, ← c117137] at e_1919_1
  have hr := e_1919_1
  simp only [e_1919_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18772 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1919 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9572)) := by
  have c117138 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117139 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117140 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117141 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117142 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117143 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1919_2 := arithEq_of_rows h (row := 1919) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_2
  simp only [← c117138, ← c117139, ← c117140] at e_1919_2
  have e_1919_3 := arithEq_of_rows h (row := 1919) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_3
  simp only [← c117141, ← c117142, k2, ← c117143] at e_1919_3
  have hr := e_1919_3
  simp only [e_1919_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18773 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1919 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9573)) := by
  have c117144 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117145 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117146 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117147 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117148 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117149 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1919_4 := arithEq_of_rows h (row := 1919) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_4
  simp only [← c117144, ← c117145, ← c117146] at e_1919_4
  have e_1919_5 := arithEq_of_rows h (row := 1919) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_5
  simp only [← c117147, ← c117148, k2, ← c117149] at e_1919_5
  have hr := e_1919_5
  simp only [e_1919_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18774 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1919 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9574)) := by
  have c117150 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117151 := (publicBatchWrapper89_copies3660 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117152 := (publicBatchWrapper89_copies3661 a h).1
  have c117153 := (publicBatchWrapper89_copies3661 a h).2.1
  have c117154 := (publicBatchWrapper89_copies3661 a h).2.2.1
  have c117155 := (publicBatchWrapper89_copies3661 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1919_6 := arithEq_of_rows h (row := 1919) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_6
  simp only [← c117150, ← c117151, ← c117152] at e_1919_6
  have e_1919_7 := arithEq_of_rows h (row := 1919) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_7
  simp only [← c117153, ← c117154, k2, ← c117155] at e_1919_7
  have hr := e_1919_7
  simp only [e_1919_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18775 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1919 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9575)) := by
  have c117156 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.1
  have c117157 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.1
  have c117158 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.1
  have c117159 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.1
  have c117160 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.1
  have c117161 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1919_8 := arithEq_of_rows h (row := 1919) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_8
  simp only [← c117156, ← c117157, ← c117158] at e_1919_8
  have e_1919_9 := arithEq_of_rows h (row := 1919) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_9
  simp only [← c117159, ← c117160, k2, ← c117161] at e_1919_9
  have hr := e_1919_9
  simp only [e_1919_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18776 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1919 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9576)) := by
  have c117162 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117163 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117164 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117165 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117166 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117167 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1919_10 := arithEq_of_rows h (row := 1919) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_10
  simp only [← c117162, ← c117163, ← c117164] at e_1919_10
  have e_1919_11 := arithEq_of_rows h (row := 1919) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_11
  simp only [← c117165, ← c117166, k2, ← c117167] at e_1919_11
  have hr := e_1919_11
  simp only [e_1919_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18777 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1919 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9577)) := by
  have c117168 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117169 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117170 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117171 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117172 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117173 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1919_12 := arithEq_of_rows h (row := 1919) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_12
  simp only [← c117168, ← c117169, ← c117170] at e_1919_12
  have e_1919_13 := arithEq_of_rows h (row := 1919) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_13
  simp only [← c117171, ← c117172, k2, ← c117173] at e_1919_13
  have hr := e_1919_13
  simp only [e_1919_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18778 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1919 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9578)) := by
  have c117174 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117175 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117176 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117177 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117178 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117179 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1919_14 := arithEq_of_rows h (row := 1919) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_14
  simp only [← c117174, ← c117175, ← c117176] at e_1919_14
  have e_1919_15 := arithEq_of_rows h (row := 1919) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_15
  simp only [← c117177, ← c117178, k2, ← c117179] at e_1919_15
  have hr := e_1919_15
  simp only [e_1919_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18779 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1919 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9579)) := by
  have c117180 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117181 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117182 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117183 := (publicBatchWrapper89_copies3661 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117184 := (publicBatchWrapper89_copies3662 a h).1
  have c117185 := (publicBatchWrapper89_copies3662 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1919_16 := arithEq_of_rows h (row := 1919) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_16
  simp only [← c117180, ← c117181, ← c117182] at e_1919_16
  have e_1919_17 := arithEq_of_rows h (row := 1919) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_17
  simp only [← c117183, ← c117184, k2, ← c117185] at e_1919_17
  have hr := e_1919_17
  simp only [e_1919_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18780 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1919 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9580)) := by
  have c117186 := (publicBatchWrapper89_copies3662 a h).2.2.1
  have c117187 := (publicBatchWrapper89_copies3662 a h).2.2.2.1
  have c117188 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.1
  have c117189 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.1
  have c117190 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.1
  have c117191 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1919_18 := arithEq_of_rows h (row := 1919) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_18
  simp only [← c117186, ← c117187, ← c117188] at e_1919_18
  have e_1919_19 := arithEq_of_rows h (row := 1919) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1919_19
  simp only [← c117189, ← c117190, k2, ← c117191] at e_1919_19
  have hr := e_1919_19
  simp only [e_1919_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18781 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1920 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9581)) := by
  have c117192 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.1
  have c117193 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.1
  have c117194 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117195 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117196 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117197 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1920_0 := arithEq_of_rows h (row := 1920) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_0
  simp only [← c117192, ← c117193, ← c117194] at e_1920_0
  have e_1920_1 := arithEq_of_rows h (row := 1920) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_1
  simp only [← c117195, ← c117196, k2, ← c117197] at e_1920_1
  have hr := e_1920_1
  simp only [e_1920_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18782 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1920 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9582)) := by
  have c117198 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117199 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117200 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117201 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117202 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117203 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1920_2 := arithEq_of_rows h (row := 1920) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_2
  simp only [← c117198, ← c117199, ← c117200] at e_1920_2
  have e_1920_3 := arithEq_of_rows h (row := 1920) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_3
  simp only [← c117201, ← c117202, k2, ← c117203] at e_1920_3
  have hr := e_1920_3
  simp only [e_1920_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18783 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1920 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9583)) := by
  have c117204 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117205 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117206 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117207 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117208 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117209 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1920_4 := arithEq_of_rows h (row := 1920) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_4
  simp only [← c117204, ← c117205, ← c117206] at e_1920_4
  have e_1920_5 := arithEq_of_rows h (row := 1920) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_5
  simp only [← c117207, ← c117208, k2, ← c117209] at e_1920_5
  have hr := e_1920_5
  simp only [e_1920_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18784 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1920 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9584)) := by
  have c117210 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117211 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117212 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117213 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117214 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117215 := (publicBatchWrapper89_copies3662 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1920_6 := arithEq_of_rows h (row := 1920) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_6
  simp only [← c117210, ← c117211, ← c117212] at e_1920_6
  have e_1920_7 := arithEq_of_rows h (row := 1920) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_7
  simp only [← c117213, ← c117214, k2, ← c117215] at e_1920_7
  have hr := e_1920_7
  simp only [e_1920_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18785 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1920 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9585)) := by
  have c117216 := (publicBatchWrapper89_copies3663 a h).1
  have c117217 := (publicBatchWrapper89_copies3663 a h).2.1
  have c117218 := (publicBatchWrapper89_copies3663 a h).2.2.1
  have c117219 := (publicBatchWrapper89_copies3663 a h).2.2.2.1
  have c117220 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.1
  have c117221 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1920_8 := arithEq_of_rows h (row := 1920) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_8
  simp only [← c117216, ← c117217, ← c117218] at e_1920_8
  have e_1920_9 := arithEq_of_rows h (row := 1920) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_9
  simp only [← c117219, ← c117220, k2, ← c117221] at e_1920_9
  have hr := e_1920_9
  simp only [e_1920_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18786 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1920 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9586)) := by
  have c117222 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.1
  have c117223 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.1
  have c117224 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.1
  have c117225 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.1
  have c117226 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117227 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1920_10 := arithEq_of_rows h (row := 1920) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_10
  simp only [← c117222, ← c117223, ← c117224] at e_1920_10
  have e_1920_11 := arithEq_of_rows h (row := 1920) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_11
  simp only [← c117225, ← c117226, k2, ← c117227] at e_1920_11
  have hr := e_1920_11
  simp only [e_1920_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18787 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1920 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9587)) := by
  have c117228 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117229 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117230 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117231 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117232 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117233 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1920_12 := arithEq_of_rows h (row := 1920) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_12
  simp only [← c117228, ← c117229, ← c117230] at e_1920_12
  have e_1920_13 := arithEq_of_rows h (row := 1920) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_13
  simp only [← c117231, ← c117232, k2, ← c117233] at e_1920_13
  have hr := e_1920_13
  simp only [e_1920_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18788 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1920 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9588)) := by
  have c117234 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117235 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117236 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117237 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117238 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117239 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1920_14 := arithEq_of_rows h (row := 1920) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_14
  simp only [← c117234, ← c117235, ← c117236] at e_1920_14
  have e_1920_15 := arithEq_of_rows h (row := 1920) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_15
  simp only [← c117237, ← c117238, k2, ← c117239] at e_1920_15
  have hr := e_1920_15
  simp only [e_1920_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18789 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1920 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9589)) := by
  have c117240 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117241 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117242 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117243 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117244 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117245 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1920_16 := arithEq_of_rows h (row := 1920) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_16
  simp only [← c117240, ← c117241, ← c117242] at e_1920_16
  have e_1920_17 := arithEq_of_rows h (row := 1920) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_17
  simp only [← c117243, ← c117244, k2, ← c117245] at e_1920_17
  have hr := e_1920_17
  simp only [e_1920_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18790 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1920 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9590)) := by
  have c117246 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117247 := (publicBatchWrapper89_copies3663 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117248 := (publicBatchWrapper89_copies3664 a h).1
  have c117249 := (publicBatchWrapper89_copies3664 a h).2.1
  have c117250 := (publicBatchWrapper89_copies3664 a h).2.2.1
  have c117251 := (publicBatchWrapper89_copies3664 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1920_18 := arithEq_of_rows h (row := 1920) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_18
  simp only [← c117246, ← c117247, ← c117248] at e_1920_18
  have e_1920_19 := arithEq_of_rows h (row := 1920) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1920_19
  simp only [← c117249, ← c117250, k2, ← c117251] at e_1920_19
  have hr := e_1920_19
  simp only [e_1920_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18791 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1921 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9591)) := by
  have c117252 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.1
  have c117253 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.1
  have c117254 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.1
  have c117255 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.1
  have c117256 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.1
  have c117257 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1921_0 := arithEq_of_rows h (row := 1921) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_0
  simp only [← c117252, ← c117253, ← c117254] at e_1921_0
  have e_1921_1 := arithEq_of_rows h (row := 1921) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_1
  simp only [← c117255, ← c117256, k2, ← c117257] at e_1921_1
  have hr := e_1921_1
  simp only [e_1921_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18792 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1921 15) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9592)) := by
  have c117258 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117259 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117260 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117261 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117262 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117263 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1921_2 := arithEq_of_rows h (row := 1921) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_2
  simp only [← c117258, ← c117259, ← c117260] at e_1921_2
  have e_1921_3 := arithEq_of_rows h (row := 1921) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_3
  simp only [← c117261, ← c117262, k2, ← c117263] at e_1921_3
  have hr := e_1921_3
  simp only [e_1921_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18793 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1921 23) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9593)) := by
  have c117264 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117265 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117266 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117267 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117268 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117269 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1921_4 := arithEq_of_rows h (row := 1921) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_4
  simp only [← c117264, ← c117265, ← c117266] at e_1921_4
  have e_1921_5 := arithEq_of_rows h (row := 1921) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_5
  simp only [← c117267, ← c117268, k2, ← c117269] at e_1921_5
  have hr := e_1921_5
  simp only [e_1921_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18794 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1921 31) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9594)) := by
  have c117270 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117271 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117272 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117273 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117274 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117275 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1921_6 := arithEq_of_rows h (row := 1921) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_6
  simp only [← c117270, ← c117271, ← c117272] at e_1921_6
  have e_1921_7 := arithEq_of_rows h (row := 1921) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_7
  simp only [← c117273, ← c117274, k2, ← c117275] at e_1921_7
  have hr := e_1921_7
  simp only [e_1921_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18795 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1921 39) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9595)) := by
  have c117276 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117277 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117278 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117279 := (publicBatchWrapper89_copies3664 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117280 := (publicBatchWrapper89_copies3665 a h).1
  have c117281 := (publicBatchWrapper89_copies3665 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1921_8 := arithEq_of_rows h (row := 1921) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_8
  simp only [← c117276, ← c117277, ← c117278] at e_1921_8
  have e_1921_9 := arithEq_of_rows h (row := 1921) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_9
  simp only [← c117279, ← c117280, k2, ← c117281] at e_1921_9
  have hr := e_1921_9
  simp only [e_1921_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18796 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1921 47) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9596)) := by
  have c117282 := (publicBatchWrapper89_copies3665 a h).2.2.1
  have c117283 := (publicBatchWrapper89_copies3665 a h).2.2.2.1
  have c117284 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.1
  have c117285 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.1
  have c117286 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.1
  have c117287 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1921_10 := arithEq_of_rows h (row := 1921) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_10
  simp only [← c117282, ← c117283, ← c117284] at e_1921_10
  have e_1921_11 := arithEq_of_rows h (row := 1921) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_11
  simp only [← c117285, ← c117286, k2, ← c117287] at e_1921_11
  have hr := e_1921_11
  simp only [e_1921_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18797 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1921 55) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9597)) := by
  have c117288 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.1
  have c117289 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.1
  have c117290 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117291 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117292 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117293 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1921_12 := arithEq_of_rows h (row := 1921) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_12
  simp only [← c117288, ← c117289, ← c117290] at e_1921_12
  have e_1921_13 := arithEq_of_rows h (row := 1921) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_13
  simp only [← c117291, ← c117292, k2, ← c117293] at e_1921_13
  have hr := e_1921_13
  simp only [e_1921_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18798 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1921 63) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9598)) := by
  have c117294 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117295 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117296 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117297 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117298 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117299 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1921_14 := arithEq_of_rows h (row := 1921) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_14
  simp only [← c117294, ← c117295, ← c117296] at e_1921_14
  have e_1921_15 := arithEq_of_rows h (row := 1921) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_15
  simp only [← c117297, ← c117298, k2, ← c117299] at e_1921_15
  have hr := e_1921_15
  simp only [e_1921_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18799 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1921 71) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9599)) := by
  have c117300 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117301 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117302 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117303 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117304 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117305 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1921_16 := arithEq_of_rows h (row := 1921) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_16
  simp only [← c117300, ← c117301, ← c117302] at e_1921_16
  have e_1921_17 := arithEq_of_rows h (row := 1921) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_17
  simp only [← c117303, ← c117304, k2, ← c117305] at e_1921_17
  have hr := e_1921_17
  simp only [e_1921_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18800 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1921 79) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9600)) := by
  have c117306 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117307 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117308 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117309 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117310 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117311 := (publicBatchWrapper89_copies3665 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1921_18 := arithEq_of_rows h (row := 1921) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_18
  simp only [← c117306, ← c117307, ← c117308] at e_1921_18
  have e_1921_19 := arithEq_of_rows h (row := 1921) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1921_19
  simp only [← c117309, ← c117310, k2, ← c117311] at e_1921_19
  have hr := e_1921_19
  simp only [e_1921_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18801 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1922 7) = bselect (a (.wire 46 47)) (a (.virt 200)) (a (.virt 9601)) := by
  have c117312 := (publicBatchWrapper89_copies3666 a h).1
  have c117313 := (publicBatchWrapper89_copies3666 a h).2.1
  have c117314 := (publicBatchWrapper89_copies3666 a h).2.2.1
  have c117315 := (publicBatchWrapper89_copies3666 a h).2.2.2.1
  have c117316 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.1
  have c117317 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1922_0 := arithEq_of_rows h (row := 1922) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_0
  simp only [← c117312, ← c117313, ← c117314] at e_1922_0
  have e_1922_1 := arithEq_of_rows h (row := 1922) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_1
  simp only [← c117315, ← c117316, k2, ← c117317] at e_1922_1
  have hr := e_1922_1
  simp only [e_1922_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18802 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1922 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9770)) := by
  have c117318 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.1
  have c117319 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.1
  have c117320 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.1
  have c117321 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.1
  have c117322 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117323 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1922_2 := arithEq_of_rows h (row := 1922) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_2
  simp only [← c117318, ← c117319, ← c117320] at e_1922_2
  have e_1922_3 := arithEq_of_rows h (row := 1922) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_3
  simp only [← c117321, ← c117322, k2, ← c117323] at e_1922_3
  have hr := e_1922_3
  simp only [e_1922_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18803 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1922 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9771)) := by
  have c117324 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117325 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117326 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117327 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117328 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117329 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1922_4 := arithEq_of_rows h (row := 1922) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_4
  simp only [← c117324, ← c117325, ← c117326] at e_1922_4
  have e_1922_5 := arithEq_of_rows h (row := 1922) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_5
  simp only [← c117327, ← c117328, k2, ← c117329] at e_1922_5
  have hr := e_1922_5
  simp only [e_1922_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18804 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1922 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9772)) := by
  have c117330 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117331 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117332 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117333 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117334 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117335 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1922_6 := arithEq_of_rows h (row := 1922) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_6
  simp only [← c117330, ← c117331, ← c117332] at e_1922_6
  have e_1922_7 := arithEq_of_rows h (row := 1922) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_7
  simp only [← c117333, ← c117334, k2, ← c117335] at e_1922_7
  have hr := e_1922_7
  simp only [e_1922_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18805 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1922 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9773)) := by
  have c117336 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117337 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117338 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117339 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117340 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117341 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1922_8 := arithEq_of_rows h (row := 1922) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_8
  simp only [← c117336, ← c117337, ← c117338] at e_1922_8
  have e_1922_9 := arithEq_of_rows h (row := 1922) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_9
  simp only [← c117339, ← c117340, k2, ← c117341] at e_1922_9
  have hr := e_1922_9
  simp only [e_1922_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18806 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1922 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9774)) := by
  have c117342 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117343 := (publicBatchWrapper89_copies3666 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117344 := (publicBatchWrapper89_copies3667 a h).1
  have c117345 := (publicBatchWrapper89_copies3667 a h).2.1
  have c117346 := (publicBatchWrapper89_copies3667 a h).2.2.1
  have c117347 := (publicBatchWrapper89_copies3667 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1922_10 := arithEq_of_rows h (row := 1922) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_10
  simp only [← c117342, ← c117343, ← c117344] at e_1922_10
  have e_1922_11 := arithEq_of_rows h (row := 1922) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_11
  simp only [← c117345, ← c117346, k2, ← c117347] at e_1922_11
  have hr := e_1922_11
  simp only [e_1922_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18807 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1922 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9775)) := by
  have c117348 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.1
  have c117349 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.1
  have c117350 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.1
  have c117351 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.1
  have c117352 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.1
  have c117353 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1922_12 := arithEq_of_rows h (row := 1922) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_12
  simp only [← c117348, ← c117349, ← c117350] at e_1922_12
  have e_1922_13 := arithEq_of_rows h (row := 1922) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_13
  simp only [← c117351, ← c117352, k2, ← c117353] at e_1922_13
  have hr := e_1922_13
  simp only [e_1922_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18808 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1922 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9776)) := by
  have c117354 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117355 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117356 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117357 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117358 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117359 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1922_14 := arithEq_of_rows h (row := 1922) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_14
  simp only [← c117354, ← c117355, ← c117356] at e_1922_14
  have e_1922_15 := arithEq_of_rows h (row := 1922) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_15
  simp only [← c117357, ← c117358, k2, ← c117359] at e_1922_15
  have hr := e_1922_15
  simp only [e_1922_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18809 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1922 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9777)) := by
  have c117360 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117361 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117362 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117363 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117364 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117365 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1922_16 := arithEq_of_rows h (row := 1922) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_16
  simp only [← c117360, ← c117361, ← c117362] at e_1922_16
  have e_1922_17 := arithEq_of_rows h (row := 1922) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_17
  simp only [← c117363, ← c117364, k2, ← c117365] at e_1922_17
  have hr := e_1922_17
  simp only [e_1922_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18810 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1922 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9778)) := by
  have c117366 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117367 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117368 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117369 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117370 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117371 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1922_18 := arithEq_of_rows h (row := 1922) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_18
  simp only [← c117366, ← c117367, ← c117368] at e_1922_18
  have e_1922_19 := arithEq_of_rows h (row := 1922) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1922_19
  simp only [← c117369, ← c117370, k2, ← c117371] at e_1922_19
  have hr := e_1922_19
  simp only [e_1922_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18811 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1923 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9779)) := by
  have c117372 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117373 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117374 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117375 := (publicBatchWrapper89_copies3667 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117376 := (publicBatchWrapper89_copies3668 a h).1
  have c117377 := (publicBatchWrapper89_copies3668 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1923_0 := arithEq_of_rows h (row := 1923) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_0
  simp only [← c117372, ← c117373, ← c117374] at e_1923_0
  have e_1923_1 := arithEq_of_rows h (row := 1923) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_1
  simp only [← c117375, ← c117376, k2, ← c117377] at e_1923_1
  have hr := e_1923_1
  simp only [e_1923_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18812 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1923 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9780)) := by
  have c117378 := (publicBatchWrapper89_copies3668 a h).2.2.1
  have c117379 := (publicBatchWrapper89_copies3668 a h).2.2.2.1
  have c117380 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.1
  have c117381 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.1
  have c117382 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.1
  have c117383 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1923_2 := arithEq_of_rows h (row := 1923) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_2
  simp only [← c117378, ← c117379, ← c117380] at e_1923_2
  have e_1923_3 := arithEq_of_rows h (row := 1923) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_3
  simp only [← c117381, ← c117382, k2, ← c117383] at e_1923_3
  have hr := e_1923_3
  simp only [e_1923_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18813 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1923 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9781)) := by
  have c117384 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.1
  have c117385 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.1
  have c117386 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117387 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117388 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117389 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1923_4 := arithEq_of_rows h (row := 1923) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_4
  simp only [← c117384, ← c117385, ← c117386] at e_1923_4
  have e_1923_5 := arithEq_of_rows h (row := 1923) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_5
  simp only [← c117387, ← c117388, k2, ← c117389] at e_1923_5
  have hr := e_1923_5
  simp only [e_1923_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18814 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1923 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9782)) := by
  have c117390 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117391 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117392 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117393 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117394 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117395 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1923_6 := arithEq_of_rows h (row := 1923) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_6
  simp only [← c117390, ← c117391, ← c117392] at e_1923_6
  have e_1923_7 := arithEq_of_rows h (row := 1923) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_7
  simp only [← c117393, ← c117394, k2, ← c117395] at e_1923_7
  have hr := e_1923_7
  simp only [e_1923_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18815 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1923 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9783)) := by
  have c117396 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117397 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117398 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117399 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117400 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117401 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1923_8 := arithEq_of_rows h (row := 1923) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_8
  simp only [← c117396, ← c117397, ← c117398] at e_1923_8
  have e_1923_9 := arithEq_of_rows h (row := 1923) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_9
  simp only [← c117399, ← c117400, k2, ← c117401] at e_1923_9
  have hr := e_1923_9
  simp only [e_1923_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18816 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1923 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9784)) := by
  have c117402 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117403 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117404 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117405 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117406 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117407 := (publicBatchWrapper89_copies3668 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1923_10 := arithEq_of_rows h (row := 1923) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_10
  simp only [← c117402, ← c117403, ← c117404] at e_1923_10
  have e_1923_11 := arithEq_of_rows h (row := 1923) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_11
  simp only [← c117405, ← c117406, k2, ← c117407] at e_1923_11
  have hr := e_1923_11
  simp only [e_1923_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18817 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1923 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9785)) := by
  have c117408 := (publicBatchWrapper89_copies3669 a h).1
  have c117409 := (publicBatchWrapper89_copies3669 a h).2.1
  have c117410 := (publicBatchWrapper89_copies3669 a h).2.2.1
  have c117411 := (publicBatchWrapper89_copies3669 a h).2.2.2.1
  have c117412 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.1
  have c117413 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1923_12 := arithEq_of_rows h (row := 1923) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_12
  simp only [← c117408, ← c117409, ← c117410] at e_1923_12
  have e_1923_13 := arithEq_of_rows h (row := 1923) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_13
  simp only [← c117411, ← c117412, k2, ← c117413] at e_1923_13
  have hr := e_1923_13
  simp only [e_1923_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18818 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1923 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9786)) := by
  have c117414 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.1
  have c117415 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.1
  have c117416 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.1
  have c117417 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.1
  have c117418 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117419 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1923_14 := arithEq_of_rows h (row := 1923) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_14
  simp only [← c117414, ← c117415, ← c117416] at e_1923_14
  have e_1923_15 := arithEq_of_rows h (row := 1923) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_15
  simp only [← c117417, ← c117418, k2, ← c117419] at e_1923_15
  have hr := e_1923_15
  simp only [e_1923_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18819 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1923 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9787)) := by
  have c117420 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117421 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117422 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117423 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117424 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117425 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1923_16 := arithEq_of_rows h (row := 1923) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_16
  simp only [← c117420, ← c117421, ← c117422] at e_1923_16
  have e_1923_17 := arithEq_of_rows h (row := 1923) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_17
  simp only [← c117423, ← c117424, k2, ← c117425] at e_1923_17
  have hr := e_1923_17
  simp only [e_1923_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18820 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1923 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9788)) := by
  have c117426 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117427 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117428 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117429 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117430 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117431 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1923_18 := arithEq_of_rows h (row := 1923) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_18
  simp only [← c117426, ← c117427, ← c117428] at e_1923_18
  have e_1923_19 := arithEq_of_rows h (row := 1923) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1923_19
  simp only [← c117429, ← c117430, k2, ← c117431] at e_1923_19
  have hr := e_1923_19
  simp only [e_1923_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18821 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1924 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9789)) := by
  have c117432 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117433 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117434 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117435 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117436 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117437 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1924_0 := arithEq_of_rows h (row := 1924) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_0
  simp only [← c117432, ← c117433, ← c117434] at e_1924_0
  have e_1924_1 := arithEq_of_rows h (row := 1924) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_1
  simp only [← c117435, ← c117436, k2, ← c117437] at e_1924_1
  have hr := e_1924_1
  simp only [e_1924_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18822 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1924 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9790)) := by
  have c117438 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117439 := (publicBatchWrapper89_copies3669 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117440 := (publicBatchWrapper89_copies3670 a h).1
  have c117441 := (publicBatchWrapper89_copies3670 a h).2.1
  have c117442 := (publicBatchWrapper89_copies3670 a h).2.2.1
  have c117443 := (publicBatchWrapper89_copies3670 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1924_2 := arithEq_of_rows h (row := 1924) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_2
  simp only [← c117438, ← c117439, ← c117440] at e_1924_2
  have e_1924_3 := arithEq_of_rows h (row := 1924) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_3
  simp only [← c117441, ← c117442, k2, ← c117443] at e_1924_3
  have hr := e_1924_3
  simp only [e_1924_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18823 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1924 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9791)) := by
  have c117444 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.1
  have c117445 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.1
  have c117446 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.1
  have c117447 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.1
  have c117448 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.1
  have c117449 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1924_4 := arithEq_of_rows h (row := 1924) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_4
  simp only [← c117444, ← c117445, ← c117446] at e_1924_4
  have e_1924_5 := arithEq_of_rows h (row := 1924) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_5
  simp only [← c117447, ← c117448, k2, ← c117449] at e_1924_5
  have hr := e_1924_5
  simp only [e_1924_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18824 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1924 31) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9792)) := by
  have c117450 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117451 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117452 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117453 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117454 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117455 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1924_6 := arithEq_of_rows h (row := 1924) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_6
  simp only [← c117450, ← c117451, ← c117452] at e_1924_6
  have e_1924_7 := arithEq_of_rows h (row := 1924) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_7
  simp only [← c117453, ← c117454, k2, ← c117455] at e_1924_7
  have hr := e_1924_7
  simp only [e_1924_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18825 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1924 39) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9793)) := by
  have c117456 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117457 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117458 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117459 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117460 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117461 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1924_8 := arithEq_of_rows h (row := 1924) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_8
  simp only [← c117456, ← c117457, ← c117458] at e_1924_8
  have e_1924_9 := arithEq_of_rows h (row := 1924) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_9
  simp only [← c117459, ← c117460, k2, ← c117461] at e_1924_9
  have hr := e_1924_9
  simp only [e_1924_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18826 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1924 47) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9794)) := by
  have c117462 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117463 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117464 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117465 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117466 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117467 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1924_10 := arithEq_of_rows h (row := 1924) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_10
  simp only [← c117462, ← c117463, ← c117464] at e_1924_10
  have e_1924_11 := arithEq_of_rows h (row := 1924) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_11
  simp only [← c117465, ← c117466, k2, ← c117467] at e_1924_11
  have hr := e_1924_11
  simp only [e_1924_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18827 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1924 55) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9795)) := by
  have c117468 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117469 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117470 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117471 := (publicBatchWrapper89_copies3670 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117472 := (publicBatchWrapper89_copies3671 a h).1
  have c117473 := (publicBatchWrapper89_copies3671 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1924_12 := arithEq_of_rows h (row := 1924) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_12
  simp only [← c117468, ← c117469, ← c117470] at e_1924_12
  have e_1924_13 := arithEq_of_rows h (row := 1924) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_13
  simp only [← c117471, ← c117472, k2, ← c117473] at e_1924_13
  have hr := e_1924_13
  simp only [e_1924_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18828 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1924 63) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9796)) := by
  have c117474 := (publicBatchWrapper89_copies3671 a h).2.2.1
  have c117475 := (publicBatchWrapper89_copies3671 a h).2.2.2.1
  have c117476 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.1
  have c117477 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.1
  have c117478 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.1
  have c117479 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1924_14 := arithEq_of_rows h (row := 1924) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_14
  simp only [← c117474, ← c117475, ← c117476] at e_1924_14
  have e_1924_15 := arithEq_of_rows h (row := 1924) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_15
  simp only [← c117477, ← c117478, k2, ← c117479] at e_1924_15
  have hr := e_1924_15
  simp only [e_1924_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18829 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1924 71) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9797)) := by
  have c117480 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.1
  have c117481 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.1
  have c117482 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117483 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117484 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117485 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1924_16 := arithEq_of_rows h (row := 1924) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_16
  simp only [← c117480, ← c117481, ← c117482] at e_1924_16
  have e_1924_17 := arithEq_of_rows h (row := 1924) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_17
  simp only [← c117483, ← c117484, k2, ← c117485] at e_1924_17
  have hr := e_1924_17
  simp only [e_1924_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18830 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1924 79) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9798)) := by
  have c117486 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117487 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117488 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117489 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117490 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117491 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1924_18 := arithEq_of_rows h (row := 1924) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_18
  simp only [← c117486, ← c117487, ← c117488] at e_1924_18
  have e_1924_19 := arithEq_of_rows h (row := 1924) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1924_19
  simp only [← c117489, ← c117490, k2, ← c117491] at e_1924_19
  have hr := e_1924_19
  simp only [e_1924_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18831 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1925 7) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9799)) := by
  have c117492 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117493 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117494 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117495 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117496 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117497 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1925_0 := arithEq_of_rows h (row := 1925) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_0
  simp only [← c117492, ← c117493, ← c117494] at e_1925_0
  have e_1925_1 := arithEq_of_rows h (row := 1925) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_1
  simp only [← c117495, ← c117496, k2, ← c117497] at e_1925_1
  have hr := e_1925_1
  simp only [e_1925_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18832 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1925 15) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9800)) := by
  have c117498 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117499 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117500 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117501 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117502 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117503 := (publicBatchWrapper89_copies3671 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1925_2 := arithEq_of_rows h (row := 1925) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_2
  simp only [← c117498, ← c117499, ← c117500] at e_1925_2
  have e_1925_3 := arithEq_of_rows h (row := 1925) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_3
  simp only [← c117501, ← c117502, k2, ← c117503] at e_1925_3
  have hr := e_1925_3
  simp only [e_1925_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18833 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1925 23) = bselect (a (.wire 48 11)) (a (.virt 200)) (a (.virt 9801)) := by
  have c117504 := (publicBatchWrapper89_copies3672 a h).1
  have c117505 := (publicBatchWrapper89_copies3672 a h).2.1
  have c117506 := (publicBatchWrapper89_copies3672 a h).2.2.1
  have c117507 := (publicBatchWrapper89_copies3672 a h).2.2.2.1
  have c117508 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.1
  have c117509 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1925_4 := arithEq_of_rows h (row := 1925) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_4
  simp only [← c117504, ← c117505, ← c117506] at e_1925_4
  have e_1925_5 := arithEq_of_rows h (row := 1925) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_5
  simp only [← c117507, ← c117508, k2, ← c117509] at e_1925_5
  have hr := e_1925_5
  simp only [e_1925_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18834 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1925 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9970)) := by
  have c117510 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.1
  have c117511 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.1
  have c117512 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.1
  have c117513 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.1
  have c117514 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117515 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1925_6 := arithEq_of_rows h (row := 1925) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_6
  simp only [← c117510, ← c117511, ← c117512] at e_1925_6
  have e_1925_7 := arithEq_of_rows h (row := 1925) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_7
  simp only [← c117513, ← c117514, k2, ← c117515] at e_1925_7
  have hr := e_1925_7
  simp only [e_1925_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18835 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1925 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9971)) := by
  have c117516 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117517 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117518 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117519 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117520 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117521 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1925_8 := arithEq_of_rows h (row := 1925) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_8
  simp only [← c117516, ← c117517, ← c117518] at e_1925_8
  have e_1925_9 := arithEq_of_rows h (row := 1925) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_9
  simp only [← c117519, ← c117520, k2, ← c117521] at e_1925_9
  have hr := e_1925_9
  simp only [e_1925_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18836 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1925 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9972)) := by
  have c117522 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117523 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117524 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117525 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117526 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117527 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1925_10 := arithEq_of_rows h (row := 1925) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_10
  simp only [← c117522, ← c117523, ← c117524] at e_1925_10
  have e_1925_11 := arithEq_of_rows h (row := 1925) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_11
  simp only [← c117525, ← c117526, k2, ← c117527] at e_1925_11
  have hr := e_1925_11
  simp only [e_1925_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18837 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1925 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9973)) := by
  have c117528 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117529 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117530 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117531 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117532 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117533 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1925_12 := arithEq_of_rows h (row := 1925) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_12
  simp only [← c117528, ← c117529, ← c117530] at e_1925_12
  have e_1925_13 := arithEq_of_rows h (row := 1925) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_13
  simp only [← c117531, ← c117532, k2, ← c117533] at e_1925_13
  have hr := e_1925_13
  simp only [e_1925_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18838 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1925 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9974)) := by
  have c117534 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117535 := (publicBatchWrapper89_copies3672 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117536 := (publicBatchWrapper89_copies3673 a h).1
  have c117537 := (publicBatchWrapper89_copies3673 a h).2.1
  have c117538 := (publicBatchWrapper89_copies3673 a h).2.2.1
  have c117539 := (publicBatchWrapper89_copies3673 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1925_14 := arithEq_of_rows h (row := 1925) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_14
  simp only [← c117534, ← c117535, ← c117536] at e_1925_14
  have e_1925_15 := arithEq_of_rows h (row := 1925) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_15
  simp only [← c117537, ← c117538, k2, ← c117539] at e_1925_15
  have hr := e_1925_15
  simp only [e_1925_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18839 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1925 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9975)) := by
  have c117540 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.1
  have c117541 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.1
  have c117542 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.1
  have c117543 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.1
  have c117544 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.1
  have c117545 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1925_16 := arithEq_of_rows h (row := 1925) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_16
  simp only [← c117540, ← c117541, ← c117542] at e_1925_16
  have e_1925_17 := arithEq_of_rows h (row := 1925) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_17
  simp only [← c117543, ← c117544, k2, ← c117545] at e_1925_17
  have hr := e_1925_17
  simp only [e_1925_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18840 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1925 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9976)) := by
  have c117546 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117547 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117548 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117549 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117550 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117551 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1925_18 := arithEq_of_rows h (row := 1925) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_18
  simp only [← c117546, ← c117547, ← c117548] at e_1925_18
  have e_1925_19 := arithEq_of_rows h (row := 1925) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1925_19
  simp only [← c117549, ← c117550, k2, ← c117551] at e_1925_19
  have hr := e_1925_19
  simp only [e_1925_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18841 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1926 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9977)) := by
  have c117552 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117553 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117554 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117555 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117556 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117557 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1926_0 := arithEq_of_rows h (row := 1926) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_0
  simp only [← c117552, ← c117553, ← c117554] at e_1926_0
  have e_1926_1 := arithEq_of_rows h (row := 1926) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_1
  simp only [← c117555, ← c117556, k2, ← c117557] at e_1926_1
  have hr := e_1926_1
  simp only [e_1926_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18842 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1926 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9978)) := by
  have c117558 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117559 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117560 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117561 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117562 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117563 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1926_2 := arithEq_of_rows h (row := 1926) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_2
  simp only [← c117558, ← c117559, ← c117560] at e_1926_2
  have e_1926_3 := arithEq_of_rows h (row := 1926) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_3
  simp only [← c117561, ← c117562, k2, ← c117563] at e_1926_3
  have hr := e_1926_3
  simp only [e_1926_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18843 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1926 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9979)) := by
  have c117564 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117565 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117566 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117567 := (publicBatchWrapper89_copies3673 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117568 := (publicBatchWrapper89_copies3674 a h).1
  have c117569 := (publicBatchWrapper89_copies3674 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1926_4 := arithEq_of_rows h (row := 1926) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_4
  simp only [← c117564, ← c117565, ← c117566] at e_1926_4
  have e_1926_5 := arithEq_of_rows h (row := 1926) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_5
  simp only [← c117567, ← c117568, k2, ← c117569] at e_1926_5
  have hr := e_1926_5
  simp only [e_1926_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18844 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1926 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9980)) := by
  have c117570 := (publicBatchWrapper89_copies3674 a h).2.2.1
  have c117571 := (publicBatchWrapper89_copies3674 a h).2.2.2.1
  have c117572 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.1
  have c117573 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.1
  have c117574 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.1
  have c117575 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1926_6 := arithEq_of_rows h (row := 1926) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_6
  simp only [← c117570, ← c117571, ← c117572] at e_1926_6
  have e_1926_7 := arithEq_of_rows h (row := 1926) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_7
  simp only [← c117573, ← c117574, k2, ← c117575] at e_1926_7
  have hr := e_1926_7
  simp only [e_1926_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18845 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1926 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9981)) := by
  have c117576 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.1
  have c117577 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.1
  have c117578 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117579 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117580 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117581 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1926_8 := arithEq_of_rows h (row := 1926) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_8
  simp only [← c117576, ← c117577, ← c117578] at e_1926_8
  have e_1926_9 := arithEq_of_rows h (row := 1926) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_9
  simp only [← c117579, ← c117580, k2, ← c117581] at e_1926_9
  have hr := e_1926_9
  simp only [e_1926_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18846 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1926 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9982)) := by
  have c117582 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117583 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117584 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117585 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117586 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117587 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1926_10 := arithEq_of_rows h (row := 1926) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_10
  simp only [← c117582, ← c117583, ← c117584] at e_1926_10
  have e_1926_11 := arithEq_of_rows h (row := 1926) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_11
  simp only [← c117585, ← c117586, k2, ← c117587] at e_1926_11
  have hr := e_1926_11
  simp only [e_1926_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18847 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1926 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9983)) := by
  have c117588 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117589 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117590 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117591 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117592 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117593 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1926_12 := arithEq_of_rows h (row := 1926) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_12
  simp only [← c117588, ← c117589, ← c117590] at e_1926_12
  have e_1926_13 := arithEq_of_rows h (row := 1926) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_13
  simp only [← c117591, ← c117592, k2, ← c117593] at e_1926_13
  have hr := e_1926_13
  simp only [e_1926_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18848 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1926 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9984)) := by
  have c117594 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117595 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117596 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117597 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117598 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117599 := (publicBatchWrapper89_copies3674 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1926_14 := arithEq_of_rows h (row := 1926) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_14
  simp only [← c117594, ← c117595, ← c117596] at e_1926_14
  have e_1926_15 := arithEq_of_rows h (row := 1926) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_15
  simp only [← c117597, ← c117598, k2, ← c117599] at e_1926_15
  have hr := e_1926_15
  simp only [e_1926_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18849 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1926 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9985)) := by
  have c117600 := (publicBatchWrapper89_copies3675 a h).1
  have c117601 := (publicBatchWrapper89_copies3675 a h).2.1
  have c117602 := (publicBatchWrapper89_copies3675 a h).2.2.1
  have c117603 := (publicBatchWrapper89_copies3675 a h).2.2.2.1
  have c117604 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.1
  have c117605 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1926_16 := arithEq_of_rows h (row := 1926) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_16
  simp only [← c117600, ← c117601, ← c117602] at e_1926_16
  have e_1926_17 := arithEq_of_rows h (row := 1926) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_17
  simp only [← c117603, ← c117604, k2, ← c117605] at e_1926_17
  have hr := e_1926_17
  simp only [e_1926_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18850 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1926 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9986)) := by
  have c117606 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.1
  have c117607 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.1
  have c117608 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.1
  have c117609 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.1
  have c117610 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117611 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1926_18 := arithEq_of_rows h (row := 1926) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_18
  simp only [← c117606, ← c117607, ← c117608] at e_1926_18
  have e_1926_19 := arithEq_of_rows h (row := 1926) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1926_19
  simp only [← c117609, ← c117610, k2, ← c117611] at e_1926_19
  have hr := e_1926_19
  simp only [e_1926_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18851 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1927 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9987)) := by
  have c117612 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117613 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117614 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117615 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117616 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117617 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1927_0 := arithEq_of_rows h (row := 1927) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_0
  simp only [← c117612, ← c117613, ← c117614] at e_1927_0
  have e_1927_1 := arithEq_of_rows h (row := 1927) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_1
  simp only [← c117615, ← c117616, k2, ← c117617] at e_1927_1
  have hr := e_1927_1
  simp only [e_1927_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18852 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1927 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9988)) := by
  have c117618 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117619 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117620 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117621 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117622 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117623 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1927_2 := arithEq_of_rows h (row := 1927) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_2
  simp only [← c117618, ← c117619, ← c117620] at e_1927_2
  have e_1927_3 := arithEq_of_rows h (row := 1927) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_3
  simp only [← c117621, ← c117622, k2, ← c117623] at e_1927_3
  have hr := e_1927_3
  simp only [e_1927_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18853 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1927 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9989)) := by
  have c117624 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117625 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117626 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117627 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117628 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117629 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1927_4 := arithEq_of_rows h (row := 1927) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_4
  simp only [← c117624, ← c117625, ← c117626] at e_1927_4
  have e_1927_5 := arithEq_of_rows h (row := 1927) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_5
  simp only [← c117627, ← c117628, k2, ← c117629] at e_1927_5
  have hr := e_1927_5
  simp only [e_1927_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18854 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1927 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9990)) := by
  have c117630 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117631 := (publicBatchWrapper89_copies3675 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117632 := (publicBatchWrapper89_copies3676 a h).1
  have c117633 := (publicBatchWrapper89_copies3676 a h).2.1
  have c117634 := (publicBatchWrapper89_copies3676 a h).2.2.1
  have c117635 := (publicBatchWrapper89_copies3676 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1927_6 := arithEq_of_rows h (row := 1927) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_6
  simp only [← c117630, ← c117631, ← c117632] at e_1927_6
  have e_1927_7 := arithEq_of_rows h (row := 1927) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_7
  simp only [← c117633, ← c117634, k2, ← c117635] at e_1927_7
  have hr := e_1927_7
  simp only [e_1927_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18855 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1927 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9991)) := by
  have c117636 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.1
  have c117637 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.1
  have c117638 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.1
  have c117639 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.1
  have c117640 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.1
  have c117641 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1927_8 := arithEq_of_rows h (row := 1927) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_8
  simp only [← c117636, ← c117637, ← c117638] at e_1927_8
  have e_1927_9 := arithEq_of_rows h (row := 1927) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_9
  simp only [← c117639, ← c117640, k2, ← c117641] at e_1927_9
  have hr := e_1927_9
  simp only [e_1927_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18856 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1927 47) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9992)) := by
  have c117642 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117643 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117644 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117645 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117646 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117647 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1927_10 := arithEq_of_rows h (row := 1927) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_10
  simp only [← c117642, ← c117643, ← c117644] at e_1927_10
  have e_1927_11 := arithEq_of_rows h (row := 1927) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_11
  simp only [← c117645, ← c117646, k2, ← c117647] at e_1927_11
  have hr := e_1927_11
  simp only [e_1927_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18857 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1927 55) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9993)) := by
  have c117648 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117649 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117650 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117651 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117652 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117653 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1927_12 := arithEq_of_rows h (row := 1927) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_12
  simp only [← c117648, ← c117649, ← c117650] at e_1927_12
  have e_1927_13 := arithEq_of_rows h (row := 1927) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_13
  simp only [← c117651, ← c117652, k2, ← c117653] at e_1927_13
  have hr := e_1927_13
  simp only [e_1927_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18858 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1927 63) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9994)) := by
  have c117654 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117655 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117656 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117657 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117658 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117659 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1927_14 := arithEq_of_rows h (row := 1927) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_14
  simp only [← c117654, ← c117655, ← c117656] at e_1927_14
  have e_1927_15 := arithEq_of_rows h (row := 1927) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_15
  simp only [← c117657, ← c117658, k2, ← c117659] at e_1927_15
  have hr := e_1927_15
  simp only [e_1927_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18859 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1927 71) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9995)) := by
  have c117660 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117661 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117662 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117663 := (publicBatchWrapper89_copies3676 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117664 := (publicBatchWrapper89_copies3677 a h).1
  have c117665 := (publicBatchWrapper89_copies3677 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1927_16 := arithEq_of_rows h (row := 1927) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_16
  simp only [← c117660, ← c117661, ← c117662] at e_1927_16
  have e_1927_17 := arithEq_of_rows h (row := 1927) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_17
  simp only [← c117663, ← c117664, k2, ← c117665] at e_1927_17
  have hr := e_1927_17
  simp only [e_1927_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18860 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1927 79) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9996)) := by
  have c117666 := (publicBatchWrapper89_copies3677 a h).2.2.1
  have c117667 := (publicBatchWrapper89_copies3677 a h).2.2.2.1
  have c117668 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.1
  have c117669 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.1
  have c117670 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.1
  have c117671 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1927_18 := arithEq_of_rows h (row := 1927) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_18
  simp only [← c117666, ← c117667, ← c117668] at e_1927_18
  have e_1927_19 := arithEq_of_rows h (row := 1927) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1927_19
  simp only [← c117669, ← c117670, k2, ← c117671] at e_1927_19
  have hr := e_1927_19
  simp only [e_1927_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18861 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1928 7) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9997)) := by
  have c117672 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.1
  have c117673 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.1
  have c117674 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117675 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117676 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117677 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1928_0 := arithEq_of_rows h (row := 1928) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_0
  simp only [← c117672, ← c117673, ← c117674] at e_1928_0
  have e_1928_1 := arithEq_of_rows h (row := 1928) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_1
  simp only [← c117675, ← c117676, k2, ← c117677] at e_1928_1
  have hr := e_1928_1
  simp only [e_1928_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18862 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1928 15) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9998)) := by
  have c117678 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117679 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117680 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117681 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117682 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117683 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1928_2 := arithEq_of_rows h (row := 1928) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_2
  simp only [← c117678, ← c117679, ← c117680] at e_1928_2
  have e_1928_3 := arithEq_of_rows h (row := 1928) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_3
  simp only [← c117681, ← c117682, k2, ← c117683] at e_1928_3
  have hr := e_1928_3
  simp only [e_1928_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18863 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1928 23) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 9999)) := by
  have c117684 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117685 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117686 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117687 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117688 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117689 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1928_4 := arithEq_of_rows h (row := 1928) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_4
  simp only [← c117684, ← c117685, ← c117686] at e_1928_4
  have e_1928_5 := arithEq_of_rows h (row := 1928) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_5
  simp only [← c117687, ← c117688, k2, ← c117689] at e_1928_5
  have hr := e_1928_5
  simp only [e_1928_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18864 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1928 31) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 10000)) := by
  have c117690 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117691 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117692 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117693 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117694 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117695 := (publicBatchWrapper89_copies3677 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1928_6 := arithEq_of_rows h (row := 1928) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_6
  simp only [← c117690, ← c117691, ← c117692] at e_1928_6
  have e_1928_7 := arithEq_of_rows h (row := 1928) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_7
  simp only [← c117693, ← c117694, k2, ← c117695] at e_1928_7
  have hr := e_1928_7
  simp only [e_1928_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18865 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1928 39) = bselect (a (.wire 48 55)) (a (.virt 200)) (a (.virt 10001)) := by
  have c117696 := (publicBatchWrapper89_copies3678 a h).1
  have c117697 := (publicBatchWrapper89_copies3678 a h).2.1
  have c117698 := (publicBatchWrapper89_copies3678 a h).2.2.1
  have c117699 := (publicBatchWrapper89_copies3678 a h).2.2.2.1
  have c117700 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.1
  have c117701 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1928_8 := arithEq_of_rows h (row := 1928) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_8
  simp only [← c117696, ← c117697, ← c117698] at e_1928_8
  have e_1928_9 := arithEq_of_rows h (row := 1928) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_9
  simp only [← c117699, ← c117700, k2, ← c117701] at e_1928_9
  have hr := e_1928_9
  simp only [e_1928_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18866 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1928 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10170)) := by
  have c117702 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.1
  have c117703 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.1
  have c117704 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.1
  have c117705 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.1
  have c117706 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117707 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1928_10 := arithEq_of_rows h (row := 1928) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_10
  simp only [← c117702, ← c117703, ← c117704] at e_1928_10
  have e_1928_11 := arithEq_of_rows h (row := 1928) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_11
  simp only [← c117705, ← c117706, k2, ← c117707] at e_1928_11
  have hr := e_1928_11
  simp only [e_1928_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18867 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1928 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10171)) := by
  have c117708 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117709 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117710 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117711 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117712 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117713 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1928_12 := arithEq_of_rows h (row := 1928) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_12
  simp only [← c117708, ← c117709, ← c117710] at e_1928_12
  have e_1928_13 := arithEq_of_rows h (row := 1928) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_13
  simp only [← c117711, ← c117712, k2, ← c117713] at e_1928_13
  have hr := e_1928_13
  simp only [e_1928_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18868 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1928 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10172)) := by
  have c117714 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117715 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117716 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117717 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117718 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117719 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1928_14 := arithEq_of_rows h (row := 1928) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_14
  simp only [← c117714, ← c117715, ← c117716] at e_1928_14
  have e_1928_15 := arithEq_of_rows h (row := 1928) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_15
  simp only [← c117717, ← c117718, k2, ← c117719] at e_1928_15
  have hr := e_1928_15
  simp only [e_1928_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18869 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1928 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10173)) := by
  have c117720 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117721 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117722 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117723 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117724 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117725 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1928_16 := arithEq_of_rows h (row := 1928) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_16
  simp only [← c117720, ← c117721, ← c117722] at e_1928_16
  have e_1928_17 := arithEq_of_rows h (row := 1928) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_17
  simp only [← c117723, ← c117724, k2, ← c117725] at e_1928_17
  have hr := e_1928_17
  simp only [e_1928_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18870 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1928 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10174)) := by
  have c117726 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117727 := (publicBatchWrapper89_copies3678 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117728 := (publicBatchWrapper89_copies3679 a h).1
  have c117729 := (publicBatchWrapper89_copies3679 a h).2.1
  have c117730 := (publicBatchWrapper89_copies3679 a h).2.2.1
  have c117731 := (publicBatchWrapper89_copies3679 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1928_18 := arithEq_of_rows h (row := 1928) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_18
  simp only [← c117726, ← c117727, ← c117728] at e_1928_18
  have e_1928_19 := arithEq_of_rows h (row := 1928) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1928_19
  simp only [← c117729, ← c117730, k2, ← c117731] at e_1928_19
  have hr := e_1928_19
  simp only [e_1928_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18871 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1929 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10175)) := by
  have c117732 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.1
  have c117733 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.1
  have c117734 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.1
  have c117735 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.1
  have c117736 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.1
  have c117737 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1929_0 := arithEq_of_rows h (row := 1929) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_0
  simp only [← c117732, ← c117733, ← c117734] at e_1929_0
  have e_1929_1 := arithEq_of_rows h (row := 1929) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_1
  simp only [← c117735, ← c117736, k2, ← c117737] at e_1929_1
  have hr := e_1929_1
  simp only [e_1929_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18872 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1929 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10176)) := by
  have c117738 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117739 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117740 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117741 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117742 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117743 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1929_2 := arithEq_of_rows h (row := 1929) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_2
  simp only [← c117738, ← c117739, ← c117740] at e_1929_2
  have e_1929_3 := arithEq_of_rows h (row := 1929) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_3
  simp only [← c117741, ← c117742, k2, ← c117743] at e_1929_3
  have hr := e_1929_3
  simp only [e_1929_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18873 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1929 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10177)) := by
  have c117744 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117745 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117746 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117747 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117748 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117749 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1929_4 := arithEq_of_rows h (row := 1929) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_4
  simp only [← c117744, ← c117745, ← c117746] at e_1929_4
  have e_1929_5 := arithEq_of_rows h (row := 1929) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_5
  simp only [← c117747, ← c117748, k2, ← c117749] at e_1929_5
  have hr := e_1929_5
  simp only [e_1929_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18874 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1929 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10178)) := by
  have c117750 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117751 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117752 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117753 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117754 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117755 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1929_6 := arithEq_of_rows h (row := 1929) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_6
  simp only [← c117750, ← c117751, ← c117752] at e_1929_6
  have e_1929_7 := arithEq_of_rows h (row := 1929) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_7
  simp only [← c117753, ← c117754, k2, ← c117755] at e_1929_7
  have hr := e_1929_7
  simp only [e_1929_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18875 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1929 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10179)) := by
  have c117756 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117757 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117758 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117759 := (publicBatchWrapper89_copies3679 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117760 := (publicBatchWrapper89_copies3680 a h).1
  have c117761 := (publicBatchWrapper89_copies3680 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1929_8 := arithEq_of_rows h (row := 1929) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_8
  simp only [← c117756, ← c117757, ← c117758] at e_1929_8
  have e_1929_9 := arithEq_of_rows h (row := 1929) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_9
  simp only [← c117759, ← c117760, k2, ← c117761] at e_1929_9
  have hr := e_1929_9
  simp only [e_1929_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18876 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1929 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10180)) := by
  have c117762 := (publicBatchWrapper89_copies3680 a h).2.2.1
  have c117763 := (publicBatchWrapper89_copies3680 a h).2.2.2.1
  have c117764 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.1
  have c117765 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.1
  have c117766 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.1
  have c117767 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1929_10 := arithEq_of_rows h (row := 1929) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_10
  simp only [← c117762, ← c117763, ← c117764] at e_1929_10
  have e_1929_11 := arithEq_of_rows h (row := 1929) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_11
  simp only [← c117765, ← c117766, k2, ← c117767] at e_1929_11
  have hr := e_1929_11
  simp only [e_1929_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18877 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1929 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10181)) := by
  have c117768 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.1
  have c117769 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.1
  have c117770 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117771 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117772 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117773 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1929_12 := arithEq_of_rows h (row := 1929) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_12
  simp only [← c117768, ← c117769, ← c117770] at e_1929_12
  have e_1929_13 := arithEq_of_rows h (row := 1929) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_13
  simp only [← c117771, ← c117772, k2, ← c117773] at e_1929_13
  have hr := e_1929_13
  simp only [e_1929_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18878 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1929 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10182)) := by
  have c117774 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117775 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117776 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117777 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117778 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117779 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1929_14 := arithEq_of_rows h (row := 1929) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_14
  simp only [← c117774, ← c117775, ← c117776] at e_1929_14
  have e_1929_15 := arithEq_of_rows h (row := 1929) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_15
  simp only [← c117777, ← c117778, k2, ← c117779] at e_1929_15
  have hr := e_1929_15
  simp only [e_1929_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18879 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1929 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10183)) := by
  have c117780 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117781 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117782 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117783 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117784 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117785 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1929_16 := arithEq_of_rows h (row := 1929) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_16
  simp only [← c117780, ← c117781, ← c117782] at e_1929_16
  have e_1929_17 := arithEq_of_rows h (row := 1929) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_17
  simp only [← c117783, ← c117784, k2, ← c117785] at e_1929_17
  have hr := e_1929_17
  simp only [e_1929_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18880 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1929 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10184)) := by
  have c117786 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117787 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117788 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117789 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117790 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117791 := (publicBatchWrapper89_copies3680 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1929_18 := arithEq_of_rows h (row := 1929) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_18
  simp only [← c117786, ← c117787, ← c117788] at e_1929_18
  have e_1929_19 := arithEq_of_rows h (row := 1929) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1929_19
  simp only [← c117789, ← c117790, k2, ← c117791] at e_1929_19
  have hr := e_1929_19
  simp only [e_1929_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18881 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1930 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10185)) := by
  have c117792 := (publicBatchWrapper89_copies3681 a h).1
  have c117793 := (publicBatchWrapper89_copies3681 a h).2.1
  have c117794 := (publicBatchWrapper89_copies3681 a h).2.2.1
  have c117795 := (publicBatchWrapper89_copies3681 a h).2.2.2.1
  have c117796 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.1
  have c117797 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1930_0 := arithEq_of_rows h (row := 1930) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_0
  simp only [← c117792, ← c117793, ← c117794] at e_1930_0
  have e_1930_1 := arithEq_of_rows h (row := 1930) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_1
  simp only [← c117795, ← c117796, k2, ← c117797] at e_1930_1
  have hr := e_1930_1
  simp only [e_1930_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18882 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1930 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10186)) := by
  have c117798 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.1
  have c117799 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.1
  have c117800 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.1
  have c117801 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.1
  have c117802 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117803 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1930_2 := arithEq_of_rows h (row := 1930) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_2
  simp only [← c117798, ← c117799, ← c117800] at e_1930_2
  have e_1930_3 := arithEq_of_rows h (row := 1930) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_3
  simp only [← c117801, ← c117802, k2, ← c117803] at e_1930_3
  have hr := e_1930_3
  simp only [e_1930_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18883 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1930 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10187)) := by
  have c117804 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117805 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117806 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117807 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117808 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117809 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1930_4 := arithEq_of_rows h (row := 1930) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_4
  simp only [← c117804, ← c117805, ← c117806] at e_1930_4
  have e_1930_5 := arithEq_of_rows h (row := 1930) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_5
  simp only [← c117807, ← c117808, k2, ← c117809] at e_1930_5
  have hr := e_1930_5
  simp only [e_1930_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18884 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1930 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10188)) := by
  have c117810 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117811 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117812 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117813 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117814 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117815 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1930_6 := arithEq_of_rows h (row := 1930) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_6
  simp only [← c117810, ← c117811, ← c117812] at e_1930_6
  have e_1930_7 := arithEq_of_rows h (row := 1930) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_7
  simp only [← c117813, ← c117814, k2, ← c117815] at e_1930_7
  have hr := e_1930_7
  simp only [e_1930_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18885 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1930 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10189)) := by
  have c117816 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117817 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117818 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117819 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117820 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117821 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1930_8 := arithEq_of_rows h (row := 1930) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_8
  simp only [← c117816, ← c117817, ← c117818] at e_1930_8
  have e_1930_9 := arithEq_of_rows h (row := 1930) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_9
  simp only [← c117819, ← c117820, k2, ← c117821] at e_1930_9
  have hr := e_1930_9
  simp only [e_1930_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18886 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1930 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10190)) := by
  have c117822 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117823 := (publicBatchWrapper89_copies3681 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117824 := (publicBatchWrapper89_copies3682 a h).1
  have c117825 := (publicBatchWrapper89_copies3682 a h).2.1
  have c117826 := (publicBatchWrapper89_copies3682 a h).2.2.1
  have c117827 := (publicBatchWrapper89_copies3682 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1930_10 := arithEq_of_rows h (row := 1930) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_10
  simp only [← c117822, ← c117823, ← c117824] at e_1930_10
  have e_1930_11 := arithEq_of_rows h (row := 1930) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_11
  simp only [← c117825, ← c117826, k2, ← c117827] at e_1930_11
  have hr := e_1930_11
  simp only [e_1930_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18887 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1930 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10191)) := by
  have c117828 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.1
  have c117829 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.1
  have c117830 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.1
  have c117831 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.1
  have c117832 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.1
  have c117833 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1930_12 := arithEq_of_rows h (row := 1930) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_12
  simp only [← c117828, ← c117829, ← c117830] at e_1930_12
  have e_1930_13 := arithEq_of_rows h (row := 1930) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_13
  simp only [← c117831, ← c117832, k2, ← c117833] at e_1930_13
  have hr := e_1930_13
  simp only [e_1930_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18888 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1930 63) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10192)) := by
  have c117834 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117835 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117836 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117837 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117838 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117839 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1930_14 := arithEq_of_rows h (row := 1930) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_14
  simp only [← c117834, ← c117835, ← c117836] at e_1930_14
  have e_1930_15 := arithEq_of_rows h (row := 1930) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_15
  simp only [← c117837, ← c117838, k2, ← c117839] at e_1930_15
  have hr := e_1930_15
  simp only [e_1930_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18889 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1930 71) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10193)) := by
  have c117840 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117841 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117842 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117843 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117844 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117845 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1930_16 := arithEq_of_rows h (row := 1930) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_16
  simp only [← c117840, ← c117841, ← c117842] at e_1930_16
  have e_1930_17 := arithEq_of_rows h (row := 1930) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_17
  simp only [← c117843, ← c117844, k2, ← c117845] at e_1930_17
  have hr := e_1930_17
  simp only [e_1930_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18890 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1930 79) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10194)) := by
  have c117846 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117847 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117848 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117849 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117850 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117851 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1930_18 := arithEq_of_rows h (row := 1930) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_18
  simp only [← c117846, ← c117847, ← c117848] at e_1930_18
  have e_1930_19 := arithEq_of_rows h (row := 1930) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1930_19
  simp only [← c117849, ← c117850, k2, ← c117851] at e_1930_19
  have hr := e_1930_19
  simp only [e_1930_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18891 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1931 7) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10195)) := by
  have c117852 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117853 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117854 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117855 := (publicBatchWrapper89_copies3682 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117856 := (publicBatchWrapper89_copies3683 a h).1
  have c117857 := (publicBatchWrapper89_copies3683 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1931_0 := arithEq_of_rows h (row := 1931) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_0
  simp only [← c117852, ← c117853, ← c117854] at e_1931_0
  have e_1931_1 := arithEq_of_rows h (row := 1931) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_1
  simp only [← c117855, ← c117856, k2, ← c117857] at e_1931_1
  have hr := e_1931_1
  simp only [e_1931_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18892 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1931 15) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10196)) := by
  have c117858 := (publicBatchWrapper89_copies3683 a h).2.2.1
  have c117859 := (publicBatchWrapper89_copies3683 a h).2.2.2.1
  have c117860 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.1
  have c117861 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.1
  have c117862 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.1
  have c117863 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1931_2 := arithEq_of_rows h (row := 1931) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_2
  simp only [← c117858, ← c117859, ← c117860] at e_1931_2
  have e_1931_3 := arithEq_of_rows h (row := 1931) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_3
  simp only [← c117861, ← c117862, k2, ← c117863] at e_1931_3
  have hr := e_1931_3
  simp only [e_1931_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18893 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1931 23) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10197)) := by
  have c117864 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.1
  have c117865 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.1
  have c117866 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117867 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117868 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117869 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1931_4 := arithEq_of_rows h (row := 1931) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_4
  simp only [← c117864, ← c117865, ← c117866] at e_1931_4
  have e_1931_5 := arithEq_of_rows h (row := 1931) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_5
  simp only [← c117867, ← c117868, k2, ← c117869] at e_1931_5
  have hr := e_1931_5
  simp only [e_1931_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18894 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1931 31) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10198)) := by
  have c117870 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117871 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117872 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117873 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117874 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117875 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1931_6 := arithEq_of_rows h (row := 1931) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_6
  simp only [← c117870, ← c117871, ← c117872] at e_1931_6
  have e_1931_7 := arithEq_of_rows h (row := 1931) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_7
  simp only [← c117873, ← c117874, k2, ← c117875] at e_1931_7
  have hr := e_1931_7
  simp only [e_1931_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18895 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1931 39) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10199)) := by
  have c117876 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117877 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117878 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117879 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117880 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117881 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1931_8 := arithEq_of_rows h (row := 1931) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_8
  simp only [← c117876, ← c117877, ← c117878] at e_1931_8
  have e_1931_9 := arithEq_of_rows h (row := 1931) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_9
  simp only [← c117879, ← c117880, k2, ← c117881] at e_1931_9
  have hr := e_1931_9
  simp only [e_1931_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18896 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1931 47) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10200)) := by
  have c117882 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117883 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117884 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117885 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117886 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117887 := (publicBatchWrapper89_copies3683 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1931_10 := arithEq_of_rows h (row := 1931) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_10
  simp only [← c117882, ← c117883, ← c117884] at e_1931_10
  have e_1931_11 := arithEq_of_rows h (row := 1931) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_11
  simp only [← c117885, ← c117886, k2, ← c117887] at e_1931_11
  have hr := e_1931_11
  simp only [e_1931_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18897 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1931 55) = bselect (a (.wire 50 19)) (a (.virt 200)) (a (.virt 10201)) := by
  have c117888 := (publicBatchWrapper89_copies3684 a h).1
  have c117889 := (publicBatchWrapper89_copies3684 a h).2.1
  have c117890 := (publicBatchWrapper89_copies3684 a h).2.2.1
  have c117891 := (publicBatchWrapper89_copies3684 a h).2.2.2.1
  have c117892 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.1
  have c117893 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1931_12 := arithEq_of_rows h (row := 1931) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_12
  simp only [← c117888, ← c117889, ← c117890] at e_1931_12
  have e_1931_13 := arithEq_of_rows h (row := 1931) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_13
  simp only [← c117891, ← c117892, k2, ← c117893] at e_1931_13
  have hr := e_1931_13
  simp only [e_1931_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18898 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1931 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10370)) := by
  have c117894 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.1
  have c117895 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.1
  have c117896 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.1
  have c117897 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.1
  have c117898 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117899 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1931_14 := arithEq_of_rows h (row := 1931) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_14
  simp only [← c117894, ← c117895, ← c117896] at e_1931_14
  have e_1931_15 := arithEq_of_rows h (row := 1931) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_15
  simp only [← c117897, ← c117898, k2, ← c117899] at e_1931_15
  have hr := e_1931_15
  simp only [e_1931_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18899 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1931 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10371)) := by
  have c117900 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117901 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117902 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117903 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117904 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117905 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1931_16 := arithEq_of_rows h (row := 1931) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_16
  simp only [← c117900, ← c117901, ← c117902] at e_1931_16
  have e_1931_17 := arithEq_of_rows h (row := 1931) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_17
  simp only [← c117903, ← c117904, k2, ← c117905] at e_1931_17
  have hr := e_1931_17
  simp only [e_1931_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18900 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1931 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10372)) := by
  have c117906 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117907 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117908 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117909 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117910 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117911 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1931_18 := arithEq_of_rows h (row := 1931) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_18
  simp only [← c117906, ← c117907, ← c117908] at e_1931_18
  have e_1931_19 := arithEq_of_rows h (row := 1931) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1931_19
  simp only [← c117909, ← c117910, k2, ← c117911] at e_1931_19
  have hr := e_1931_19
  simp only [e_1931_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18901 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1932 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10373)) := by
  have c117912 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117913 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117914 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117915 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117916 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117917 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1932_0 := arithEq_of_rows h (row := 1932) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_0
  simp only [← c117912, ← c117913, ← c117914] at e_1932_0
  have e_1932_1 := arithEq_of_rows h (row := 1932) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_1
  simp only [← c117915, ← c117916, k2, ← c117917] at e_1932_1
  have hr := e_1932_1
  simp only [e_1932_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18902 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1932 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10374)) := by
  have c117918 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117919 := (publicBatchWrapper89_copies3684 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117920 := (publicBatchWrapper89_copies3685 a h).1
  have c117921 := (publicBatchWrapper89_copies3685 a h).2.1
  have c117922 := (publicBatchWrapper89_copies3685 a h).2.2.1
  have c117923 := (publicBatchWrapper89_copies3685 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1932_2 := arithEq_of_rows h (row := 1932) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_2
  simp only [← c117918, ← c117919, ← c117920] at e_1932_2
  have e_1932_3 := arithEq_of_rows h (row := 1932) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_3
  simp only [← c117921, ← c117922, k2, ← c117923] at e_1932_3
  have hr := e_1932_3
  simp only [e_1932_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18903 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1932 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10375)) := by
  have c117924 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.1
  have c117925 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.1
  have c117926 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.1
  have c117927 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.1
  have c117928 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.1
  have c117929 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1932_4 := arithEq_of_rows h (row := 1932) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_4
  simp only [← c117924, ← c117925, ← c117926] at e_1932_4
  have e_1932_5 := arithEq_of_rows h (row := 1932) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_5
  simp only [← c117927, ← c117928, k2, ← c117929] at e_1932_5
  have hr := e_1932_5
  simp only [e_1932_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18904 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1932 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10376)) := by
  have c117930 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117931 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117932 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117933 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117934 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117935 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1932_6 := arithEq_of_rows h (row := 1932) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_6
  simp only [← c117930, ← c117931, ← c117932] at e_1932_6
  have e_1932_7 := arithEq_of_rows h (row := 1932) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_7
  simp only [← c117933, ← c117934, k2, ← c117935] at e_1932_7
  have hr := e_1932_7
  simp only [e_1932_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18905 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1932 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10377)) := by
  have c117936 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117937 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117938 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117939 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117940 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117941 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1932_8 := arithEq_of_rows h (row := 1932) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_8
  simp only [← c117936, ← c117937, ← c117938] at e_1932_8
  have e_1932_9 := arithEq_of_rows h (row := 1932) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_9
  simp only [← c117939, ← c117940, k2, ← c117941] at e_1932_9
  have hr := e_1932_9
  simp only [e_1932_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18906 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1932 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10378)) := by
  have c117942 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117943 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117944 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117945 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117946 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117947 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1932_10 := arithEq_of_rows h (row := 1932) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_10
  simp only [← c117942, ← c117943, ← c117944] at e_1932_10
  have e_1932_11 := arithEq_of_rows h (row := 1932) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_11
  simp only [← c117945, ← c117946, k2, ← c117947] at e_1932_11
  have hr := e_1932_11
  simp only [e_1932_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18907 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1932 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10379)) := by
  have c117948 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117949 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117950 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117951 := (publicBatchWrapper89_copies3685 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c117952 := (publicBatchWrapper89_copies3686 a h).1
  have c117953 := (publicBatchWrapper89_copies3686 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1932_12 := arithEq_of_rows h (row := 1932) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_12
  simp only [← c117948, ← c117949, ← c117950] at e_1932_12
  have e_1932_13 := arithEq_of_rows h (row := 1932) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_13
  simp only [← c117951, ← c117952, k2, ← c117953] at e_1932_13
  have hr := e_1932_13
  simp only [e_1932_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18908 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1932 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10380)) := by
  have c117954 := (publicBatchWrapper89_copies3686 a h).2.2.1
  have c117955 := (publicBatchWrapper89_copies3686 a h).2.2.2.1
  have c117956 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.1
  have c117957 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.1
  have c117958 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.1
  have c117959 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1932_14 := arithEq_of_rows h (row := 1932) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_14
  simp only [← c117954, ← c117955, ← c117956] at e_1932_14
  have e_1932_15 := arithEq_of_rows h (row := 1932) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_15
  simp only [← c117957, ← c117958, k2, ← c117959] at e_1932_15
  have hr := e_1932_15
  simp only [e_1932_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18909 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1932 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10381)) := by
  have c117960 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.1
  have c117961 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.1
  have c117962 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117963 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c117964 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117965 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1932_16 := arithEq_of_rows h (row := 1932) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_16
  simp only [← c117960, ← c117961, ← c117962] at e_1932_16
  have e_1932_17 := arithEq_of_rows h (row := 1932) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_17
  simp only [← c117963, ← c117964, k2, ← c117965] at e_1932_17
  have hr := e_1932_17
  simp only [e_1932_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18910 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1932 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10382)) := by
  have c117966 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117967 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117968 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117969 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117970 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117971 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1932_18 := arithEq_of_rows h (row := 1932) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_18
  simp only [← c117966, ← c117967, ← c117968] at e_1932_18
  have e_1932_19 := arithEq_of_rows h (row := 1932) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1932_19
  simp only [← c117969, ← c117970, k2, ← c117971] at e_1932_19
  have hr := e_1932_19
  simp only [e_1932_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18911 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1933 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10383)) := by
  have c117972 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117973 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117974 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117975 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117976 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117977 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1933_0 := arithEq_of_rows h (row := 1933) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_0
  simp only [← c117972, ← c117973, ← c117974] at e_1933_0
  have e_1933_1 := arithEq_of_rows h (row := 1933) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_1
  simp only [← c117975, ← c117976, k2, ← c117977] at e_1933_1
  have hr := e_1933_1
  simp only [e_1933_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18912 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1933 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10384)) := by
  have c117978 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117979 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117980 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117981 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117982 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117983 := (publicBatchWrapper89_copies3686 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1933_2 := arithEq_of_rows h (row := 1933) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_2
  simp only [← c117978, ← c117979, ← c117980] at e_1933_2
  have e_1933_3 := arithEq_of_rows h (row := 1933) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_3
  simp only [← c117981, ← c117982, k2, ← c117983] at e_1933_3
  have hr := e_1933_3
  simp only [e_1933_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18913 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1933 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10385)) := by
  have c117984 := (publicBatchWrapper89_copies3687 a h).1
  have c117985 := (publicBatchWrapper89_copies3687 a h).2.1
  have c117986 := (publicBatchWrapper89_copies3687 a h).2.2.1
  have c117987 := (publicBatchWrapper89_copies3687 a h).2.2.2.1
  have c117988 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.1
  have c117989 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1933_4 := arithEq_of_rows h (row := 1933) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_4
  simp only [← c117984, ← c117985, ← c117986] at e_1933_4
  have e_1933_5 := arithEq_of_rows h (row := 1933) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_5
  simp only [← c117987, ← c117988, k2, ← c117989] at e_1933_5
  have hr := e_1933_5
  simp only [e_1933_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18914 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1933 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10386)) := by
  have c117990 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.1
  have c117991 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.1
  have c117992 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.1
  have c117993 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.1
  have c117994 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.1
  have c117995 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1933_6 := arithEq_of_rows h (row := 1933) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_6
  simp only [← c117990, ← c117991, ← c117992] at e_1933_6
  have e_1933_7 := arithEq_of_rows h (row := 1933) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_7
  simp only [← c117993, ← c117994, k2, ← c117995] at e_1933_7
  have hr := e_1933_7
  simp only [e_1933_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18915 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1933 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10387)) := by
  have c117996 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117997 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117998 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c117999 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118000 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118001 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1933_8 := arithEq_of_rows h (row := 1933) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_8
  simp only [← c117996, ← c117997, ← c117998] at e_1933_8
  have e_1933_9 := arithEq_of_rows h (row := 1933) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_9
  simp only [← c117999, ← c118000, k2, ← c118001] at e_1933_9
  have hr := e_1933_9
  simp only [e_1933_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18916 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1933 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10388)) := by
  have c118002 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118003 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118004 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118005 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118006 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118007 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1933_10 := arithEq_of_rows h (row := 1933) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_10
  simp only [← c118002, ← c118003, ← c118004] at e_1933_10
  have e_1933_11 := arithEq_of_rows h (row := 1933) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_11
  simp only [← c118005, ← c118006, k2, ← c118007] at e_1933_11
  have hr := e_1933_11
  simp only [e_1933_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18917 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1933 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10389)) := by
  have c118008 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118009 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118010 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118011 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118012 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118013 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1933_12 := arithEq_of_rows h (row := 1933) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_12
  simp only [← c118008, ← c118009, ← c118010] at e_1933_12
  have e_1933_13 := arithEq_of_rows h (row := 1933) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_13
  simp only [← c118011, ← c118012, k2, ← c118013] at e_1933_13
  have hr := e_1933_13
  simp only [e_1933_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18918 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1933 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10390)) := by
  have c118014 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118015 := (publicBatchWrapper89_copies3687 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118016 := (publicBatchWrapper89_copies3688 a h).1
  have c118017 := (publicBatchWrapper89_copies3688 a h).2.1
  have c118018 := (publicBatchWrapper89_copies3688 a h).2.2.1
  have c118019 := (publicBatchWrapper89_copies3688 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1933_14 := arithEq_of_rows h (row := 1933) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_14
  simp only [← c118014, ← c118015, ← c118016] at e_1933_14
  have e_1933_15 := arithEq_of_rows h (row := 1933) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_15
  simp only [← c118017, ← c118018, k2, ← c118019] at e_1933_15
  have hr := e_1933_15
  simp only [e_1933_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18919 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1933 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10391)) := by
  have c118020 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.1
  have c118021 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.1
  have c118022 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.1
  have c118023 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.1
  have c118024 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.1
  have c118025 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1933_16 := arithEq_of_rows h (row := 1933) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_16
  simp only [← c118020, ← c118021, ← c118022] at e_1933_16
  have e_1933_17 := arithEq_of_rows h (row := 1933) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_17
  simp only [← c118023, ← c118024, k2, ← c118025] at e_1933_17
  have hr := e_1933_17
  simp only [e_1933_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18920 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1933 79) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10392)) := by
  have c118026 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118027 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118028 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118029 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118030 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118031 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1933_18 := arithEq_of_rows h (row := 1933) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_18
  simp only [← c118026, ← c118027, ← c118028] at e_1933_18
  have e_1933_19 := arithEq_of_rows h (row := 1933) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1933_19
  simp only [← c118029, ← c118030, k2, ← c118031] at e_1933_19
  have hr := e_1933_19
  simp only [e_1933_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18921 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1934 7) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10393)) := by
  have c118032 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118033 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118034 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118035 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118036 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118037 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1934_0 := arithEq_of_rows h (row := 1934) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_0
  simp only [← c118032, ← c118033, ← c118034] at e_1934_0
  have e_1934_1 := arithEq_of_rows h (row := 1934) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_1
  simp only [← c118035, ← c118036, k2, ← c118037] at e_1934_1
  have hr := e_1934_1
  simp only [e_1934_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18922 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1934 15) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10394)) := by
  have c118038 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118039 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118040 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118041 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118042 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118043 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1934_2 := arithEq_of_rows h (row := 1934) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_2
  simp only [← c118038, ← c118039, ← c118040] at e_1934_2
  have e_1934_3 := arithEq_of_rows h (row := 1934) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_3
  simp only [← c118041, ← c118042, k2, ← c118043] at e_1934_3
  have hr := e_1934_3
  simp only [e_1934_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18923 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1934 23) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10395)) := by
  have c118044 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118045 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118046 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118047 := (publicBatchWrapper89_copies3688 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118048 := (publicBatchWrapper89_copies3689 a h).1
  have c118049 := (publicBatchWrapper89_copies3689 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1934_4 := arithEq_of_rows h (row := 1934) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_4
  simp only [← c118044, ← c118045, ← c118046] at e_1934_4
  have e_1934_5 := arithEq_of_rows h (row := 1934) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_5
  simp only [← c118047, ← c118048, k2, ← c118049] at e_1934_5
  have hr := e_1934_5
  simp only [e_1934_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18924 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1934 31) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10396)) := by
  have c118050 := (publicBatchWrapper89_copies3689 a h).2.2.1
  have c118051 := (publicBatchWrapper89_copies3689 a h).2.2.2.1
  have c118052 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.1
  have c118053 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.1
  have c118054 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.1
  have c118055 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1934_6 := arithEq_of_rows h (row := 1934) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_6
  simp only [← c118050, ← c118051, ← c118052] at e_1934_6
  have e_1934_7 := arithEq_of_rows h (row := 1934) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_7
  simp only [← c118053, ← c118054, k2, ← c118055] at e_1934_7
  have hr := e_1934_7
  simp only [e_1934_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18925 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1934 39) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10397)) := by
  have c118056 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.1
  have c118057 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.1
  have c118058 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118059 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118060 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118061 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1934_8 := arithEq_of_rows h (row := 1934) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_8
  simp only [← c118056, ← c118057, ← c118058] at e_1934_8
  have e_1934_9 := arithEq_of_rows h (row := 1934) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_9
  simp only [← c118059, ← c118060, k2, ← c118061] at e_1934_9
  have hr := e_1934_9
  simp only [e_1934_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18926 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1934 47) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10398)) := by
  have c118062 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118063 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118064 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118065 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118066 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118067 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1934_10 := arithEq_of_rows h (row := 1934) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_10
  simp only [← c118062, ← c118063, ← c118064] at e_1934_10
  have e_1934_11 := arithEq_of_rows h (row := 1934) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_11
  simp only [← c118065, ← c118066, k2, ← c118067] at e_1934_11
  have hr := e_1934_11
  simp only [e_1934_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18927 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1934 55) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10399)) := by
  have c118068 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118069 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118070 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118071 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118072 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118073 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1934_12 := arithEq_of_rows h (row := 1934) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_12
  simp only [← c118068, ← c118069, ← c118070] at e_1934_12
  have e_1934_13 := arithEq_of_rows h (row := 1934) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_13
  simp only [← c118071, ← c118072, k2, ← c118073] at e_1934_13
  have hr := e_1934_13
  simp only [e_1934_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18928 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1934 63) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10400)) := by
  have c118074 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118075 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118076 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118077 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118078 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118079 := (publicBatchWrapper89_copies3689 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1934_14 := arithEq_of_rows h (row := 1934) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_14
  simp only [← c118074, ← c118075, ← c118076] at e_1934_14
  have e_1934_15 := arithEq_of_rows h (row := 1934) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_15
  simp only [← c118077, ← c118078, k2, ← c118079] at e_1934_15
  have hr := e_1934_15
  simp only [e_1934_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18929 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1934 71) = bselect (a (.wire 50 63)) (a (.virt 200)) (a (.virt 10401)) := by
  have c118080 := (publicBatchWrapper89_copies3690 a h).1
  have c118081 := (publicBatchWrapper89_copies3690 a h).2.1
  have c118082 := (publicBatchWrapper89_copies3690 a h).2.2.1
  have c118083 := (publicBatchWrapper89_copies3690 a h).2.2.2.1
  have c118084 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.1
  have c118085 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1934_16 := arithEq_of_rows h (row := 1934) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_16
  simp only [← c118080, ← c118081, ← c118082] at e_1934_16
  have e_1934_17 := arithEq_of_rows h (row := 1934) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_17
  simp only [← c118083, ← c118084, k2, ← c118085] at e_1934_17
  have hr := e_1934_17
  simp only [e_1934_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18930 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1934 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10570)) := by
  have c118086 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.1
  have c118087 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.1
  have c118088 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.1
  have c118089 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.1
  have c118090 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118091 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1934_18 := arithEq_of_rows h (row := 1934) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_18
  simp only [← c118086, ← c118087, ← c118088] at e_1934_18
  have e_1934_19 := arithEq_of_rows h (row := 1934) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1934_19
  simp only [← c118089, ← c118090, k2, ← c118091] at e_1934_19
  have hr := e_1934_19
  simp only [e_1934_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18931 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1935 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10571)) := by
  have c118092 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118093 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118094 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118095 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118096 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118097 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1935_0 := arithEq_of_rows h (row := 1935) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_0
  simp only [← c118092, ← c118093, ← c118094] at e_1935_0
  have e_1935_1 := arithEq_of_rows h (row := 1935) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_1
  simp only [← c118095, ← c118096, k2, ← c118097] at e_1935_1
  have hr := e_1935_1
  simp only [e_1935_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18932 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1935 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10572)) := by
  have c118098 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118099 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118100 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118101 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118102 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118103 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1935_2 := arithEq_of_rows h (row := 1935) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_2
  simp only [← c118098, ← c118099, ← c118100] at e_1935_2
  have e_1935_3 := arithEq_of_rows h (row := 1935) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_3
  simp only [← c118101, ← c118102, k2, ← c118103] at e_1935_3
  have hr := e_1935_3
  simp only [e_1935_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18933 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1935 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10573)) := by
  have c118104 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118105 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118106 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118107 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118108 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118109 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1935_4 := arithEq_of_rows h (row := 1935) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_4
  simp only [← c118104, ← c118105, ← c118106] at e_1935_4
  have e_1935_5 := arithEq_of_rows h (row := 1935) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_5
  simp only [← c118107, ← c118108, k2, ← c118109] at e_1935_5
  have hr := e_1935_5
  simp only [e_1935_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18934 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1935 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10574)) := by
  have c118110 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118111 := (publicBatchWrapper89_copies3690 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118112 := (publicBatchWrapper89_copies3691 a h).1
  have c118113 := (publicBatchWrapper89_copies3691 a h).2.1
  have c118114 := (publicBatchWrapper89_copies3691 a h).2.2.1
  have c118115 := (publicBatchWrapper89_copies3691 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1935_6 := arithEq_of_rows h (row := 1935) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_6
  simp only [← c118110, ← c118111, ← c118112] at e_1935_6
  have e_1935_7 := arithEq_of_rows h (row := 1935) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_7
  simp only [← c118113, ← c118114, k2, ← c118115] at e_1935_7
  have hr := e_1935_7
  simp only [e_1935_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18935 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1935 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10575)) := by
  have c118116 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.1
  have c118117 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.1
  have c118118 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.1
  have c118119 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.1
  have c118120 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.1
  have c118121 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1935_8 := arithEq_of_rows h (row := 1935) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_8
  simp only [← c118116, ← c118117, ← c118118] at e_1935_8
  have e_1935_9 := arithEq_of_rows h (row := 1935) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_9
  simp only [← c118119, ← c118120, k2, ← c118121] at e_1935_9
  have hr := e_1935_9
  simp only [e_1935_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18936 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1935 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10576)) := by
  have c118122 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118123 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118124 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118125 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118126 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118127 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1935_10 := arithEq_of_rows h (row := 1935) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_10
  simp only [← c118122, ← c118123, ← c118124] at e_1935_10
  have e_1935_11 := arithEq_of_rows h (row := 1935) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_11
  simp only [← c118125, ← c118126, k2, ← c118127] at e_1935_11
  have hr := e_1935_11
  simp only [e_1935_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18937 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1935 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10577)) := by
  have c118128 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118129 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118130 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118131 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118132 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118133 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1935_12 := arithEq_of_rows h (row := 1935) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_12
  simp only [← c118128, ← c118129, ← c118130] at e_1935_12
  have e_1935_13 := arithEq_of_rows h (row := 1935) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_13
  simp only [← c118131, ← c118132, k2, ← c118133] at e_1935_13
  have hr := e_1935_13
  simp only [e_1935_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18938 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1935 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10578)) := by
  have c118134 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118135 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118136 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118137 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118138 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118139 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1935_14 := arithEq_of_rows h (row := 1935) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_14
  simp only [← c118134, ← c118135, ← c118136] at e_1935_14
  have e_1935_15 := arithEq_of_rows h (row := 1935) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_15
  simp only [← c118137, ← c118138, k2, ← c118139] at e_1935_15
  have hr := e_1935_15
  simp only [e_1935_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18939 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1935 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10579)) := by
  have c118140 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118141 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118142 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118143 := (publicBatchWrapper89_copies3691 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118144 := (publicBatchWrapper89_copies3692 a h).1
  have c118145 := (publicBatchWrapper89_copies3692 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1935_16 := arithEq_of_rows h (row := 1935) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_16
  simp only [← c118140, ← c118141, ← c118142] at e_1935_16
  have e_1935_17 := arithEq_of_rows h (row := 1935) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_17
  simp only [← c118143, ← c118144, k2, ← c118145] at e_1935_17
  have hr := e_1935_17
  simp only [e_1935_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18940 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1935 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10580)) := by
  have c118146 := (publicBatchWrapper89_copies3692 a h).2.2.1
  have c118147 := (publicBatchWrapper89_copies3692 a h).2.2.2.1
  have c118148 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.1
  have c118149 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.1
  have c118150 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.1
  have c118151 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1935_18 := arithEq_of_rows h (row := 1935) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_18
  simp only [← c118146, ← c118147, ← c118148] at e_1935_18
  have e_1935_19 := arithEq_of_rows h (row := 1935) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1935_19
  simp only [← c118149, ← c118150, k2, ← c118151] at e_1935_19
  have hr := e_1935_19
  simp only [e_1935_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18941 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1936 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10581)) := by
  have c118152 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.1
  have c118153 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.1
  have c118154 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118155 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118156 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118157 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1936_0 := arithEq_of_rows h (row := 1936) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_0
  simp only [← c118152, ← c118153, ← c118154] at e_1936_0
  have e_1936_1 := arithEq_of_rows h (row := 1936) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_1
  simp only [← c118155, ← c118156, k2, ← c118157] at e_1936_1
  have hr := e_1936_1
  simp only [e_1936_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18942 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1936 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10582)) := by
  have c118158 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118159 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118160 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118161 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118162 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118163 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1936_2 := arithEq_of_rows h (row := 1936) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_2
  simp only [← c118158, ← c118159, ← c118160] at e_1936_2
  have e_1936_3 := arithEq_of_rows h (row := 1936) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_3
  simp only [← c118161, ← c118162, k2, ← c118163] at e_1936_3
  have hr := e_1936_3
  simp only [e_1936_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18943 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1936 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10583)) := by
  have c118164 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118165 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118166 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118167 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118168 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118169 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1936_4 := arithEq_of_rows h (row := 1936) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_4
  simp only [← c118164, ← c118165, ← c118166] at e_1936_4
  have e_1936_5 := arithEq_of_rows h (row := 1936) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_5
  simp only [← c118167, ← c118168, k2, ← c118169] at e_1936_5
  have hr := e_1936_5
  simp only [e_1936_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18944 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1936 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10584)) := by
  have c118170 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118171 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118172 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118173 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118174 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118175 := (publicBatchWrapper89_copies3692 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1936_6 := arithEq_of_rows h (row := 1936) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_6
  simp only [← c118170, ← c118171, ← c118172] at e_1936_6
  have e_1936_7 := arithEq_of_rows h (row := 1936) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_7
  simp only [← c118173, ← c118174, k2, ← c118175] at e_1936_7
  have hr := e_1936_7
  simp only [e_1936_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18945 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1936 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10585)) := by
  have c118176 := (publicBatchWrapper89_copies3693 a h).1
  have c118177 := (publicBatchWrapper89_copies3693 a h).2.1
  have c118178 := (publicBatchWrapper89_copies3693 a h).2.2.1
  have c118179 := (publicBatchWrapper89_copies3693 a h).2.2.2.1
  have c118180 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.1
  have c118181 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1936_8 := arithEq_of_rows h (row := 1936) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_8
  simp only [← c118176, ← c118177, ← c118178] at e_1936_8
  have e_1936_9 := arithEq_of_rows h (row := 1936) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_9
  simp only [← c118179, ← c118180, k2, ← c118181] at e_1936_9
  have hr := e_1936_9
  simp only [e_1936_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18946 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1936 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10586)) := by
  have c118182 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.1
  have c118183 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.1
  have c118184 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.1
  have c118185 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.1
  have c118186 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118187 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1936_10 := arithEq_of_rows h (row := 1936) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_10
  simp only [← c118182, ← c118183, ← c118184] at e_1936_10
  have e_1936_11 := arithEq_of_rows h (row := 1936) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_11
  simp only [← c118185, ← c118186, k2, ← c118187] at e_1936_11
  have hr := e_1936_11
  simp only [e_1936_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18947 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1936 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10587)) := by
  have c118188 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118189 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118190 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118191 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118192 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118193 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1936_12 := arithEq_of_rows h (row := 1936) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_12
  simp only [← c118188, ← c118189, ← c118190] at e_1936_12
  have e_1936_13 := arithEq_of_rows h (row := 1936) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_13
  simp only [← c118191, ← c118192, k2, ← c118193] at e_1936_13
  have hr := e_1936_13
  simp only [e_1936_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18948 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1936 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10588)) := by
  have c118194 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118195 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118196 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118197 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118198 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118199 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1936_14 := arithEq_of_rows h (row := 1936) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_14
  simp only [← c118194, ← c118195, ← c118196] at e_1936_14
  have e_1936_15 := arithEq_of_rows h (row := 1936) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_15
  simp only [← c118197, ← c118198, k2, ← c118199] at e_1936_15
  have hr := e_1936_15
  simp only [e_1936_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18949 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1936 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10589)) := by
  have c118200 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118201 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118202 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118203 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118204 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118205 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1936_16 := arithEq_of_rows h (row := 1936) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_16
  simp only [← c118200, ← c118201, ← c118202] at e_1936_16
  have e_1936_17 := arithEq_of_rows h (row := 1936) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_17
  simp only [← c118203, ← c118204, k2, ← c118205] at e_1936_17
  have hr := e_1936_17
  simp only [e_1936_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18950 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1936 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10590)) := by
  have c118206 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118207 := (publicBatchWrapper89_copies3693 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118208 := (publicBatchWrapper89_copies3694 a h).1
  have c118209 := (publicBatchWrapper89_copies3694 a h).2.1
  have c118210 := (publicBatchWrapper89_copies3694 a h).2.2.1
  have c118211 := (publicBatchWrapper89_copies3694 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1936_18 := arithEq_of_rows h (row := 1936) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_18
  simp only [← c118206, ← c118207, ← c118208] at e_1936_18
  have e_1936_19 := arithEq_of_rows h (row := 1936) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1936_19
  simp only [← c118209, ← c118210, k2, ← c118211] at e_1936_19
  have hr := e_1936_19
  simp only [e_1936_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18951 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1937 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10591)) := by
  have c118212 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.1
  have c118213 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.1
  have c118214 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.1
  have c118215 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.1
  have c118216 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.1
  have c118217 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1937_0 := arithEq_of_rows h (row := 1937) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_0
  simp only [← c118212, ← c118213, ← c118214] at e_1937_0
  have e_1937_1 := arithEq_of_rows h (row := 1937) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_1
  simp only [← c118215, ← c118216, k2, ← c118217] at e_1937_1
  have hr := e_1937_1
  simp only [e_1937_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18952 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1937 15) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10592)) := by
  have c118218 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118219 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118220 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118221 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118222 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118223 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1937_2 := arithEq_of_rows h (row := 1937) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_2
  simp only [← c118218, ← c118219, ← c118220] at e_1937_2
  have e_1937_3 := arithEq_of_rows h (row := 1937) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_3
  simp only [← c118221, ← c118222, k2, ← c118223] at e_1937_3
  have hr := e_1937_3
  simp only [e_1937_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18953 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1937 23) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10593)) := by
  have c118224 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118225 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118226 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118227 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118228 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118229 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1937_4 := arithEq_of_rows h (row := 1937) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_4
  simp only [← c118224, ← c118225, ← c118226] at e_1937_4
  have e_1937_5 := arithEq_of_rows h (row := 1937) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_5
  simp only [← c118227, ← c118228, k2, ← c118229] at e_1937_5
  have hr := e_1937_5
  simp only [e_1937_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18954 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1937 31) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10594)) := by
  have c118230 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118231 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118232 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118233 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118234 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118235 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1937_6 := arithEq_of_rows h (row := 1937) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_6
  simp only [← c118230, ← c118231, ← c118232] at e_1937_6
  have e_1937_7 := arithEq_of_rows h (row := 1937) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_7
  simp only [← c118233, ← c118234, k2, ← c118235] at e_1937_7
  have hr := e_1937_7
  simp only [e_1937_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18955 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1937 39) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10595)) := by
  have c118236 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118237 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118238 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118239 := (publicBatchWrapper89_copies3694 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118240 := (publicBatchWrapper89_copies3695 a h).1
  have c118241 := (publicBatchWrapper89_copies3695 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1937_8 := arithEq_of_rows h (row := 1937) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_8
  simp only [← c118236, ← c118237, ← c118238] at e_1937_8
  have e_1937_9 := arithEq_of_rows h (row := 1937) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_9
  simp only [← c118239, ← c118240, k2, ← c118241] at e_1937_9
  have hr := e_1937_9
  simp only [e_1937_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18956 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1937 47) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10596)) := by
  have c118242 := (publicBatchWrapper89_copies3695 a h).2.2.1
  have c118243 := (publicBatchWrapper89_copies3695 a h).2.2.2.1
  have c118244 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.1
  have c118245 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.1
  have c118246 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.1
  have c118247 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1937_10 := arithEq_of_rows h (row := 1937) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_10
  simp only [← c118242, ← c118243, ← c118244] at e_1937_10
  have e_1937_11 := arithEq_of_rows h (row := 1937) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_11
  simp only [← c118245, ← c118246, k2, ← c118247] at e_1937_11
  have hr := e_1937_11
  simp only [e_1937_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18957 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1937 55) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10597)) := by
  have c118248 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.1
  have c118249 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.1
  have c118250 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118251 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118252 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118253 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1937_12 := arithEq_of_rows h (row := 1937) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_12
  simp only [← c118248, ← c118249, ← c118250] at e_1937_12
  have e_1937_13 := arithEq_of_rows h (row := 1937) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_13
  simp only [← c118251, ← c118252, k2, ← c118253] at e_1937_13
  have hr := e_1937_13
  simp only [e_1937_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18958 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1937 63) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10598)) := by
  have c118254 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118255 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118256 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118257 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118258 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118259 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1937_14 := arithEq_of_rows h (row := 1937) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_14
  simp only [← c118254, ← c118255, ← c118256] at e_1937_14
  have e_1937_15 := arithEq_of_rows h (row := 1937) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_15
  simp only [← c118257, ← c118258, k2, ← c118259] at e_1937_15
  have hr := e_1937_15
  simp only [e_1937_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18959 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1937 71) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10599)) := by
  have c118260 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118261 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118262 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118263 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118264 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118265 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1937_16 := arithEq_of_rows h (row := 1937) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_16
  simp only [← c118260, ← c118261, ← c118262] at e_1937_16
  have e_1937_17 := arithEq_of_rows h (row := 1937) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_17
  simp only [← c118263, ← c118264, k2, ← c118265] at e_1937_17
  have hr := e_1937_17
  simp only [e_1937_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18960 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1937 79) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10600)) := by
  have c118266 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118267 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118268 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118269 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118270 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118271 := (publicBatchWrapper89_copies3695 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1937_18 := arithEq_of_rows h (row := 1937) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_18
  simp only [← c118266, ← c118267, ← c118268] at e_1937_18
  have e_1937_19 := arithEq_of_rows h (row := 1937) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1937_19
  simp only [← c118269, ← c118270, k2, ← c118271] at e_1937_19
  have hr := e_1937_19
  simp only [e_1937_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18961 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1938 7) = bselect (a (.wire 52 27)) (a (.virt 200)) (a (.virt 10601)) := by
  have c118272 := (publicBatchWrapper89_copies3696 a h).1
  have c118273 := (publicBatchWrapper89_copies3696 a h).2.1
  have c118274 := (publicBatchWrapper89_copies3696 a h).2.2.1
  have c118275 := (publicBatchWrapper89_copies3696 a h).2.2.2.1
  have c118276 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.1
  have c118277 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1938_0 := arithEq_of_rows h (row := 1938) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_0
  simp only [← c118272, ← c118273, ← c118274] at e_1938_0
  have e_1938_1 := arithEq_of_rows h (row := 1938) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_1
  simp only [← c118275, ← c118276, k2, ← c118277] at e_1938_1
  have hr := e_1938_1
  simp only [e_1938_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18962 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1938 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10770)) := by
  have c118278 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.1
  have c118279 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.1
  have c118280 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.1
  have c118281 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.1
  have c118282 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118283 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1938_2 := arithEq_of_rows h (row := 1938) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_2
  simp only [← c118278, ← c118279, ← c118280] at e_1938_2
  have e_1938_3 := arithEq_of_rows h (row := 1938) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_3
  simp only [← c118281, ← c118282, k2, ← c118283] at e_1938_3
  have hr := e_1938_3
  simp only [e_1938_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18963 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1938 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10771)) := by
  have c118284 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118285 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118286 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118287 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118288 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118289 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1938_4 := arithEq_of_rows h (row := 1938) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_4
  simp only [← c118284, ← c118285, ← c118286] at e_1938_4
  have e_1938_5 := arithEq_of_rows h (row := 1938) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_5
  simp only [← c118287, ← c118288, k2, ← c118289] at e_1938_5
  have hr := e_1938_5
  simp only [e_1938_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18964 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1938 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10772)) := by
  have c118290 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118291 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118292 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118293 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118294 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118295 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1938_6 := arithEq_of_rows h (row := 1938) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_6
  simp only [← c118290, ← c118291, ← c118292] at e_1938_6
  have e_1938_7 := arithEq_of_rows h (row := 1938) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_7
  simp only [← c118293, ← c118294, k2, ← c118295] at e_1938_7
  have hr := e_1938_7
  simp only [e_1938_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18965 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1938 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10773)) := by
  have c118296 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118297 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118298 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118299 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118300 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118301 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1938_8 := arithEq_of_rows h (row := 1938) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_8
  simp only [← c118296, ← c118297, ← c118298] at e_1938_8
  have e_1938_9 := arithEq_of_rows h (row := 1938) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_9
  simp only [← c118299, ← c118300, k2, ← c118301] at e_1938_9
  have hr := e_1938_9
  simp only [e_1938_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18966 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1938 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10774)) := by
  have c118302 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118303 := (publicBatchWrapper89_copies3696 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118304 := (publicBatchWrapper89_copies3697 a h).1
  have c118305 := (publicBatchWrapper89_copies3697 a h).2.1
  have c118306 := (publicBatchWrapper89_copies3697 a h).2.2.1
  have c118307 := (publicBatchWrapper89_copies3697 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1938_10 := arithEq_of_rows h (row := 1938) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_10
  simp only [← c118302, ← c118303, ← c118304] at e_1938_10
  have e_1938_11 := arithEq_of_rows h (row := 1938) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_11
  simp only [← c118305, ← c118306, k2, ← c118307] at e_1938_11
  have hr := e_1938_11
  simp only [e_1938_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18967 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1938 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10775)) := by
  have c118308 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.1
  have c118309 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.1
  have c118310 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.1
  have c118311 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.1
  have c118312 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.1
  have c118313 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1938_12 := arithEq_of_rows h (row := 1938) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_12
  simp only [← c118308, ← c118309, ← c118310] at e_1938_12
  have e_1938_13 := arithEq_of_rows h (row := 1938) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_13
  simp only [← c118311, ← c118312, k2, ← c118313] at e_1938_13
  have hr := e_1938_13
  simp only [e_1938_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18968 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1938 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10776)) := by
  have c118314 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118315 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118316 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118317 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118318 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118319 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1938_14 := arithEq_of_rows h (row := 1938) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_14
  simp only [← c118314, ← c118315, ← c118316] at e_1938_14
  have e_1938_15 := arithEq_of_rows h (row := 1938) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_15
  simp only [← c118317, ← c118318, k2, ← c118319] at e_1938_15
  have hr := e_1938_15
  simp only [e_1938_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18969 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1938 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10777)) := by
  have c118320 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118321 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118322 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118323 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118324 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118325 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1938_16 := arithEq_of_rows h (row := 1938) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_16
  simp only [← c118320, ← c118321, ← c118322] at e_1938_16
  have e_1938_17 := arithEq_of_rows h (row := 1938) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_17
  simp only [← c118323, ← c118324, k2, ← c118325] at e_1938_17
  have hr := e_1938_17
  simp only [e_1938_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18970 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1938 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10778)) := by
  have c118326 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118327 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118328 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118329 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118330 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118331 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1938_18 := arithEq_of_rows h (row := 1938) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_18
  simp only [← c118326, ← c118327, ← c118328] at e_1938_18
  have e_1938_19 := arithEq_of_rows h (row := 1938) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1938_19
  simp only [← c118329, ← c118330, k2, ← c118331] at e_1938_19
  have hr := e_1938_19
  simp only [e_1938_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18971 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1939 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10779)) := by
  have c118332 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118333 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118334 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118335 := (publicBatchWrapper89_copies3697 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118336 := (publicBatchWrapper89_copies3698 a h).1
  have c118337 := (publicBatchWrapper89_copies3698 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1939_0 := arithEq_of_rows h (row := 1939) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_0
  simp only [← c118332, ← c118333, ← c118334] at e_1939_0
  have e_1939_1 := arithEq_of_rows h (row := 1939) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_1
  simp only [← c118335, ← c118336, k2, ← c118337] at e_1939_1
  have hr := e_1939_1
  simp only [e_1939_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18972 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1939 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10780)) := by
  have c118338 := (publicBatchWrapper89_copies3698 a h).2.2.1
  have c118339 := (publicBatchWrapper89_copies3698 a h).2.2.2.1
  have c118340 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.1
  have c118341 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.1
  have c118342 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.1
  have c118343 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1939_2 := arithEq_of_rows h (row := 1939) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_2
  simp only [← c118338, ← c118339, ← c118340] at e_1939_2
  have e_1939_3 := arithEq_of_rows h (row := 1939) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_3
  simp only [← c118341, ← c118342, k2, ← c118343] at e_1939_3
  have hr := e_1939_3
  simp only [e_1939_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18973 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1939 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10781)) := by
  have c118344 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.1
  have c118345 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.1
  have c118346 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118347 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118348 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118349 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1939_4 := arithEq_of_rows h (row := 1939) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_4
  simp only [← c118344, ← c118345, ← c118346] at e_1939_4
  have e_1939_5 := arithEq_of_rows h (row := 1939) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_5
  simp only [← c118347, ← c118348, k2, ← c118349] at e_1939_5
  have hr := e_1939_5
  simp only [e_1939_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18974 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1939 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10782)) := by
  have c118350 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118351 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118352 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118353 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118354 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118355 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1939_6 := arithEq_of_rows h (row := 1939) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_6
  simp only [← c118350, ← c118351, ← c118352] at e_1939_6
  have e_1939_7 := arithEq_of_rows h (row := 1939) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_7
  simp only [← c118353, ← c118354, k2, ← c118355] at e_1939_7
  have hr := e_1939_7
  simp only [e_1939_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18975 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1939 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10783)) := by
  have c118356 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118357 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118358 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118359 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118360 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118361 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1939_8 := arithEq_of_rows h (row := 1939) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_8
  simp only [← c118356, ← c118357, ← c118358] at e_1939_8
  have e_1939_9 := arithEq_of_rows h (row := 1939) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_9
  simp only [← c118359, ← c118360, k2, ← c118361] at e_1939_9
  have hr := e_1939_9
  simp only [e_1939_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18976 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1939 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10784)) := by
  have c118362 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118363 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118364 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118365 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118366 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118367 := (publicBatchWrapper89_copies3698 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1939_10 := arithEq_of_rows h (row := 1939) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_10
  simp only [← c118362, ← c118363, ← c118364] at e_1939_10
  have e_1939_11 := arithEq_of_rows h (row := 1939) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_11
  simp only [← c118365, ← c118366, k2, ← c118367] at e_1939_11
  have hr := e_1939_11
  simp only [e_1939_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18977 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1939 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10785)) := by
  have c118368 := (publicBatchWrapper89_copies3699 a h).1
  have c118369 := (publicBatchWrapper89_copies3699 a h).2.1
  have c118370 := (publicBatchWrapper89_copies3699 a h).2.2.1
  have c118371 := (publicBatchWrapper89_copies3699 a h).2.2.2.1
  have c118372 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.1
  have c118373 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1939_12 := arithEq_of_rows h (row := 1939) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_12
  simp only [← c118368, ← c118369, ← c118370] at e_1939_12
  have e_1939_13 := arithEq_of_rows h (row := 1939) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_13
  simp only [← c118371, ← c118372, k2, ← c118373] at e_1939_13
  have hr := e_1939_13
  simp only [e_1939_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18978 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1939 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10786)) := by
  have c118374 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.1
  have c118375 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.1
  have c118376 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.1
  have c118377 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.1
  have c118378 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118379 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1939_14 := arithEq_of_rows h (row := 1939) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_14
  simp only [← c118374, ← c118375, ← c118376] at e_1939_14
  have e_1939_15 := arithEq_of_rows h (row := 1939) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_15
  simp only [← c118377, ← c118378, k2, ← c118379] at e_1939_15
  have hr := e_1939_15
  simp only [e_1939_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18979 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1939 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10787)) := by
  have c118380 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118381 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118382 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118383 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118384 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118385 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1939_16 := arithEq_of_rows h (row := 1939) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_16
  simp only [← c118380, ← c118381, ← c118382] at e_1939_16
  have e_1939_17 := arithEq_of_rows h (row := 1939) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_17
  simp only [← c118383, ← c118384, k2, ← c118385] at e_1939_17
  have hr := e_1939_17
  simp only [e_1939_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18980 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1939 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10788)) := by
  have c118386 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118387 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118388 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118389 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118390 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118391 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1939_18 := arithEq_of_rows h (row := 1939) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_18
  simp only [← c118386, ← c118387, ← c118388] at e_1939_18
  have e_1939_19 := arithEq_of_rows h (row := 1939) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1939_19
  simp only [← c118389, ← c118390, k2, ← c118391] at e_1939_19
  have hr := e_1939_19
  simp only [e_1939_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18981 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1940 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10789)) := by
  have c118392 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118393 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118394 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118395 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118396 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118397 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1940_0 := arithEq_of_rows h (row := 1940) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_0
  simp only [← c118392, ← c118393, ← c118394] at e_1940_0
  have e_1940_1 := arithEq_of_rows h (row := 1940) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_1
  simp only [← c118395, ← c118396, k2, ← c118397] at e_1940_1
  have hr := e_1940_1
  simp only [e_1940_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18982 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1940 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10790)) := by
  have c118398 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118399 := (publicBatchWrapper89_copies3699 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118400 := (publicBatchWrapper89_copies3700 a h).1
  have c118401 := (publicBatchWrapper89_copies3700 a h).2.1
  have c118402 := (publicBatchWrapper89_copies3700 a h).2.2.1
  have c118403 := (publicBatchWrapper89_copies3700 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1940_2 := arithEq_of_rows h (row := 1940) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_2
  simp only [← c118398, ← c118399, ← c118400] at e_1940_2
  have e_1940_3 := arithEq_of_rows h (row := 1940) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_3
  simp only [← c118401, ← c118402, k2, ← c118403] at e_1940_3
  have hr := e_1940_3
  simp only [e_1940_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18983 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1940 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10791)) := by
  have c118404 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.1
  have c118405 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.1
  have c118406 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.1
  have c118407 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.1
  have c118408 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.1
  have c118409 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1940_4 := arithEq_of_rows h (row := 1940) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_4
  simp only [← c118404, ← c118405, ← c118406] at e_1940_4
  have e_1940_5 := arithEq_of_rows h (row := 1940) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_5
  simp only [← c118407, ← c118408, k2, ← c118409] at e_1940_5
  have hr := e_1940_5
  simp only [e_1940_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18984 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1940 31) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10792)) := by
  have c118410 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118411 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118412 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118413 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118414 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118415 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1940_6 := arithEq_of_rows h (row := 1940) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_6
  simp only [← c118410, ← c118411, ← c118412] at e_1940_6
  have e_1940_7 := arithEq_of_rows h (row := 1940) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_7
  simp only [← c118413, ← c118414, k2, ← c118415] at e_1940_7
  have hr := e_1940_7
  simp only [e_1940_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18985 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1940 39) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10793)) := by
  have c118416 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118417 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118418 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118419 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118420 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118421 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1940_8 := arithEq_of_rows h (row := 1940) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_8
  simp only [← c118416, ← c118417, ← c118418] at e_1940_8
  have e_1940_9 := arithEq_of_rows h (row := 1940) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_9
  simp only [← c118419, ← c118420, k2, ← c118421] at e_1940_9
  have hr := e_1940_9
  simp only [e_1940_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18986 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1940 47) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10794)) := by
  have c118422 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118423 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118424 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118425 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118426 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118427 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1940_10 := arithEq_of_rows h (row := 1940) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_10
  simp only [← c118422, ← c118423, ← c118424] at e_1940_10
  have e_1940_11 := arithEq_of_rows h (row := 1940) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_11
  simp only [← c118425, ← c118426, k2, ← c118427] at e_1940_11
  have hr := e_1940_11
  simp only [e_1940_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18987 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1940 55) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10795)) := by
  have c118428 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118429 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118430 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118431 := (publicBatchWrapper89_copies3700 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118432 := (publicBatchWrapper89_copies3701 a h).1
  have c118433 := (publicBatchWrapper89_copies3701 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1940_12 := arithEq_of_rows h (row := 1940) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_12
  simp only [← c118428, ← c118429, ← c118430] at e_1940_12
  have e_1940_13 := arithEq_of_rows h (row := 1940) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_13
  simp only [← c118431, ← c118432, k2, ← c118433] at e_1940_13
  have hr := e_1940_13
  simp only [e_1940_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18988 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1940 63) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10796)) := by
  have c118434 := (publicBatchWrapper89_copies3701 a h).2.2.1
  have c118435 := (publicBatchWrapper89_copies3701 a h).2.2.2.1
  have c118436 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.1
  have c118437 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.1
  have c118438 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.1
  have c118439 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1940_14 := arithEq_of_rows h (row := 1940) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_14
  simp only [← c118434, ← c118435, ← c118436] at e_1940_14
  have e_1940_15 := arithEq_of_rows h (row := 1940) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_15
  simp only [← c118437, ← c118438, k2, ← c118439] at e_1940_15
  have hr := e_1940_15
  simp only [e_1940_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18989 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1940 71) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10797)) := by
  have c118440 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.1
  have c118441 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.1
  have c118442 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118443 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118444 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118445 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1940_16 := arithEq_of_rows h (row := 1940) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_16
  simp only [← c118440, ← c118441, ← c118442] at e_1940_16
  have e_1940_17 := arithEq_of_rows h (row := 1940) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_17
  simp only [← c118443, ← c118444, k2, ← c118445] at e_1940_17
  have hr := e_1940_17
  simp only [e_1940_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18990 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1940 79) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10798)) := by
  have c118446 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118447 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118448 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118449 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118450 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118451 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1940_18 := arithEq_of_rows h (row := 1940) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_18
  simp only [← c118446, ← c118447, ← c118448] at e_1940_18
  have e_1940_19 := arithEq_of_rows h (row := 1940) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1940_19
  simp only [← c118449, ← c118450, k2, ← c118451] at e_1940_19
  have hr := e_1940_19
  simp only [e_1940_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18991 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 7) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10799)) := by
  have c118452 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118453 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118454 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118455 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118456 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118457 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_0 := arithEq_of_rows h (row := 1941) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_0
  simp only [← c118452, ← c118453, ← c118454] at e_1941_0
  have e_1941_1 := arithEq_of_rows h (row := 1941) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_1
  simp only [← c118455, ← c118456, k2, ← c118457] at e_1941_1
  have hr := e_1941_1
  simp only [e_1941_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18992 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 15) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10800)) := by
  have c118458 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118459 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118460 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118461 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118462 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118463 := (publicBatchWrapper89_copies3701 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_2 := arithEq_of_rows h (row := 1941) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_2
  simp only [← c118458, ← c118459, ← c118460] at e_1941_2
  have e_1941_3 := arithEq_of_rows h (row := 1941) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_3
  simp only [← c118461, ← c118462, k2, ← c118463] at e_1941_3
  have hr := e_1941_3
  simp only [e_1941_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18993 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 23) = bselect (a (.wire 52 71)) (a (.virt 200)) (a (.virt 10801)) := by
  have c118464 := (publicBatchWrapper89_copies3702 a h).1
  have c118465 := (publicBatchWrapper89_copies3702 a h).2.1
  have c118466 := (publicBatchWrapper89_copies3702 a h).2.2.1
  have c118467 := (publicBatchWrapper89_copies3702 a h).2.2.2.1
  have c118468 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.1
  have c118469 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_4 := arithEq_of_rows h (row := 1941) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_4
  simp only [← c118464, ← c118465, ← c118466] at e_1941_4
  have e_1941_5 := arithEq_of_rows h (row := 1941) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_5
  simp only [← c118467, ← c118468, k2, ← c118469] at e_1941_5
  have hr := e_1941_5
  simp only [e_1941_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18994 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10970)) := by
  have c118470 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.1
  have c118471 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.1
  have c118472 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.1
  have c118473 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.1
  have c118474 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118475 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_6 := arithEq_of_rows h (row := 1941) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_6
  simp only [← c118470, ← c118471, ← c118472] at e_1941_6
  have e_1941_7 := arithEq_of_rows h (row := 1941) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_7
  simp only [← c118473, ← c118474, k2, ← c118475] at e_1941_7
  have hr := e_1941_7
  simp only [e_1941_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18995 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10971)) := by
  have c118476 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118477 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118478 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118479 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118480 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118481 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_8 := arithEq_of_rows h (row := 1941) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_8
  simp only [← c118476, ← c118477, ← c118478] at e_1941_8
  have e_1941_9 := arithEq_of_rows h (row := 1941) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_9
  simp only [← c118479, ← c118480, k2, ← c118481] at e_1941_9
  have hr := e_1941_9
  simp only [e_1941_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18996 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10972)) := by
  have c118482 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118483 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118484 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118485 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118486 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118487 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_10 := arithEq_of_rows h (row := 1941) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_10
  simp only [← c118482, ← c118483, ← c118484] at e_1941_10
  have e_1941_11 := arithEq_of_rows h (row := 1941) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_11
  simp only [← c118485, ← c118486, k2, ← c118487] at e_1941_11
  have hr := e_1941_11
  simp only [e_1941_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18997 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10973)) := by
  have c118488 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118489 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118490 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118491 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118492 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118493 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_12 := arithEq_of_rows h (row := 1941) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_12
  simp only [← c118488, ← c118489, ← c118490] at e_1941_12
  have e_1941_13 := arithEq_of_rows h (row := 1941) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_13
  simp only [← c118491, ← c118492, k2, ← c118493] at e_1941_13
  have hr := e_1941_13
  simp only [e_1941_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18998 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10974)) := by
  have c118494 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118495 := (publicBatchWrapper89_copies3702 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118496 := (publicBatchWrapper89_copies3703 a h).1
  have c118497 := (publicBatchWrapper89_copies3703 a h).2.1
  have c118498 := (publicBatchWrapper89_copies3703 a h).2.2.1
  have c118499 := (publicBatchWrapper89_copies3703 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_14 := arithEq_of_rows h (row := 1941) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_14
  simp only [← c118494, ← c118495, ← c118496] at e_1941_14
  have e_1941_15 := arithEq_of_rows h (row := 1941) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_15
  simp only [← c118497, ← c118498, k2, ← c118499] at e_1941_15
  have hr := e_1941_15
  simp only [e_1941_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f18999 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10975)) := by
  have c118500 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.1
  have c118501 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.1
  have c118502 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.1
  have c118503 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.1
  have c118504 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.1
  have c118505 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_16 := arithEq_of_rows h (row := 1941) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_16
  simp only [← c118500, ← c118501, ← c118502] at e_1941_16
  have e_1941_17 := arithEq_of_rows h (row := 1941) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_17
  simp only [← c118503, ← c118504, k2, ← c118505] at e_1941_17
  have hr := e_1941_17
  simp only [e_1941_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19000 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1941 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10976)) := by
  have c118506 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118507 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118508 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118509 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118510 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118511 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1941_18 := arithEq_of_rows h (row := 1941) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_18
  simp only [← c118506, ← c118507, ← c118508] at e_1941_18
  have e_1941_19 := arithEq_of_rows h (row := 1941) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1941_19
  simp only [← c118509, ← c118510, k2, ← c118511] at e_1941_19
  have hr := e_1941_19
  simp only [e_1941_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19001 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10977)) := by
  have c118512 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118513 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118514 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118515 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118516 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118517 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_0 := arithEq_of_rows h (row := 1942) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_0
  simp only [← c118512, ← c118513, ← c118514] at e_1942_0
  have e_1942_1 := arithEq_of_rows h (row := 1942) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_1
  simp only [← c118515, ← c118516, k2, ← c118517] at e_1942_1
  have hr := e_1942_1
  simp only [e_1942_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19002 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10978)) := by
  have c118518 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118519 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118520 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118521 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118522 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118523 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_2 := arithEq_of_rows h (row := 1942) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_2
  simp only [← c118518, ← c118519, ← c118520] at e_1942_2
  have e_1942_3 := arithEq_of_rows h (row := 1942) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_3
  simp only [← c118521, ← c118522, k2, ← c118523] at e_1942_3
  have hr := e_1942_3
  simp only [e_1942_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19003 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10979)) := by
  have c118524 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118525 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118526 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118527 := (publicBatchWrapper89_copies3703 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118528 := (publicBatchWrapper89_copies3704 a h).1
  have c118529 := (publicBatchWrapper89_copies3704 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_4 := arithEq_of_rows h (row := 1942) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_4
  simp only [← c118524, ← c118525, ← c118526] at e_1942_4
  have e_1942_5 := arithEq_of_rows h (row := 1942) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_5
  simp only [← c118527, ← c118528, k2, ← c118529] at e_1942_5
  have hr := e_1942_5
  simp only [e_1942_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19004 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10980)) := by
  have c118530 := (publicBatchWrapper89_copies3704 a h).2.2.1
  have c118531 := (publicBatchWrapper89_copies3704 a h).2.2.2.1
  have c118532 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.1
  have c118533 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.1
  have c118534 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.1
  have c118535 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_6 := arithEq_of_rows h (row := 1942) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_6
  simp only [← c118530, ← c118531, ← c118532] at e_1942_6
  have e_1942_7 := arithEq_of_rows h (row := 1942) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_7
  simp only [← c118533, ← c118534, k2, ← c118535] at e_1942_7
  have hr := e_1942_7
  simp only [e_1942_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19005 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10981)) := by
  have c118536 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.1
  have c118537 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.1
  have c118538 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118539 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118540 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118541 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_8 := arithEq_of_rows h (row := 1942) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_8
  simp only [← c118536, ← c118537, ← c118538] at e_1942_8
  have e_1942_9 := arithEq_of_rows h (row := 1942) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_9
  simp only [← c118539, ← c118540, k2, ← c118541] at e_1942_9
  have hr := e_1942_9
  simp only [e_1942_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19006 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10982)) := by
  have c118542 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118543 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118544 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118545 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118546 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118547 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_10 := arithEq_of_rows h (row := 1942) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_10
  simp only [← c118542, ← c118543, ← c118544] at e_1942_10
  have e_1942_11 := arithEq_of_rows h (row := 1942) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_11
  simp only [← c118545, ← c118546, k2, ← c118547] at e_1942_11
  have hr := e_1942_11
  simp only [e_1942_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19007 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10983)) := by
  have c118548 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118549 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118550 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118551 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118552 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118553 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_12 := arithEq_of_rows h (row := 1942) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_12
  simp only [← c118548, ← c118549, ← c118550] at e_1942_12
  have e_1942_13 := arithEq_of_rows h (row := 1942) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_13
  simp only [← c118551, ← c118552, k2, ← c118553] at e_1942_13
  have hr := e_1942_13
  simp only [e_1942_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19008 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10984)) := by
  have c118554 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118555 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118556 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118557 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118558 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118559 := (publicBatchWrapper89_copies3704 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_14 := arithEq_of_rows h (row := 1942) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_14
  simp only [← c118554, ← c118555, ← c118556] at e_1942_14
  have e_1942_15 := arithEq_of_rows h (row := 1942) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_15
  simp only [← c118557, ← c118558, k2, ← c118559] at e_1942_15
  have hr := e_1942_15
  simp only [e_1942_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19009 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10985)) := by
  have c118560 := (publicBatchWrapper89_copies3705 a h).1
  have c118561 := (publicBatchWrapper89_copies3705 a h).2.1
  have c118562 := (publicBatchWrapper89_copies3705 a h).2.2.1
  have c118563 := (publicBatchWrapper89_copies3705 a h).2.2.2.1
  have c118564 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.1
  have c118565 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_16 := arithEq_of_rows h (row := 1942) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_16
  simp only [← c118560, ← c118561, ← c118562] at e_1942_16
  have e_1942_17 := arithEq_of_rows h (row := 1942) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_17
  simp only [← c118563, ← c118564, k2, ← c118565] at e_1942_17
  have hr := e_1942_17
  simp only [e_1942_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19010 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1942 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10986)) := by
  have c118566 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.1
  have c118567 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.1
  have c118568 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.1
  have c118569 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.1
  have c118570 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118571 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1942_18 := arithEq_of_rows h (row := 1942) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_18
  simp only [← c118566, ← c118567, ← c118568] at e_1942_18
  have e_1942_19 := arithEq_of_rows h (row := 1942) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1942_19
  simp only [← c118569, ← c118570, k2, ← c118571] at e_1942_19
  have hr := e_1942_19
  simp only [e_1942_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19011 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10987)) := by
  have c118572 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118573 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118574 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118575 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118576 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118577 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_0 := arithEq_of_rows h (row := 1943) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_0
  simp only [← c118572, ← c118573, ← c118574] at e_1943_0
  have e_1943_1 := arithEq_of_rows h (row := 1943) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_1
  simp only [← c118575, ← c118576, k2, ← c118577] at e_1943_1
  have hr := e_1943_1
  simp only [e_1943_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19012 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10988)) := by
  have c118578 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118579 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118580 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118581 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118582 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118583 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_2 := arithEq_of_rows h (row := 1943) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_2
  simp only [← c118578, ← c118579, ← c118580] at e_1943_2
  have e_1943_3 := arithEq_of_rows h (row := 1943) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_3
  simp only [← c118581, ← c118582, k2, ← c118583] at e_1943_3
  have hr := e_1943_3
  simp only [e_1943_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19013 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10989)) := by
  have c118584 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118585 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118586 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118587 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118588 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118589 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_4 := arithEq_of_rows h (row := 1943) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_4
  simp only [← c118584, ← c118585, ← c118586] at e_1943_4
  have e_1943_5 := arithEq_of_rows h (row := 1943) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_5
  simp only [← c118587, ← c118588, k2, ← c118589] at e_1943_5
  have hr := e_1943_5
  simp only [e_1943_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19014 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10990)) := by
  have c118590 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118591 := (publicBatchWrapper89_copies3705 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118592 := (publicBatchWrapper89_copies3706 a h).1
  have c118593 := (publicBatchWrapper89_copies3706 a h).2.1
  have c118594 := (publicBatchWrapper89_copies3706 a h).2.2.1
  have c118595 := (publicBatchWrapper89_copies3706 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_6 := arithEq_of_rows h (row := 1943) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_6
  simp only [← c118590, ← c118591, ← c118592] at e_1943_6
  have e_1943_7 := arithEq_of_rows h (row := 1943) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_7
  simp only [← c118593, ← c118594, k2, ← c118595] at e_1943_7
  have hr := e_1943_7
  simp only [e_1943_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19015 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10991)) := by
  have c118596 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.1
  have c118597 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.1
  have c118598 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.1
  have c118599 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.1
  have c118600 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.1
  have c118601 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_8 := arithEq_of_rows h (row := 1943) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_8
  simp only [← c118596, ← c118597, ← c118598] at e_1943_8
  have e_1943_9 := arithEq_of_rows h (row := 1943) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_9
  simp only [← c118599, ← c118600, k2, ← c118601] at e_1943_9
  have hr := e_1943_9
  simp only [e_1943_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19016 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 47) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10992)) := by
  have c118602 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118603 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118604 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118605 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118606 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118607 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_10 := arithEq_of_rows h (row := 1943) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_10
  simp only [← c118602, ← c118603, ← c118604] at e_1943_10
  have e_1943_11 := arithEq_of_rows h (row := 1943) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_11
  simp only [← c118605, ← c118606, k2, ← c118607] at e_1943_11
  have hr := e_1943_11
  simp only [e_1943_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19017 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 55) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10993)) := by
  have c118608 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118609 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118610 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118611 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118612 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118613 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_12 := arithEq_of_rows h (row := 1943) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_12
  simp only [← c118608, ← c118609, ← c118610] at e_1943_12
  have e_1943_13 := arithEq_of_rows h (row := 1943) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_13
  simp only [← c118611, ← c118612, k2, ← c118613] at e_1943_13
  have hr := e_1943_13
  simp only [e_1943_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19018 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 63) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10994)) := by
  have c118614 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118615 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118616 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118617 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118618 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118619 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_14 := arithEq_of_rows h (row := 1943) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_14
  simp only [← c118614, ← c118615, ← c118616] at e_1943_14
  have e_1943_15 := arithEq_of_rows h (row := 1943) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_15
  simp only [← c118617, ← c118618, k2, ← c118619] at e_1943_15
  have hr := e_1943_15
  simp only [e_1943_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19019 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 71) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10995)) := by
  have c118620 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118621 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118622 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118623 := (publicBatchWrapper89_copies3706 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118624 := (publicBatchWrapper89_copies3707 a h).1
  have c118625 := (publicBatchWrapper89_copies3707 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_16 := arithEq_of_rows h (row := 1943) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_16
  simp only [← c118620, ← c118621, ← c118622] at e_1943_16
  have e_1943_17 := arithEq_of_rows h (row := 1943) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_17
  simp only [← c118623, ← c118624, k2, ← c118625] at e_1943_17
  have hr := e_1943_17
  simp only [e_1943_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19020 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1943 79) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10996)) := by
  have c118626 := (publicBatchWrapper89_copies3707 a h).2.2.1
  have c118627 := (publicBatchWrapper89_copies3707 a h).2.2.2.1
  have c118628 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.1
  have c118629 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.1
  have c118630 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.1
  have c118631 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1943_18 := arithEq_of_rows h (row := 1943) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_18
  simp only [← c118626, ← c118627, ← c118628] at e_1943_18
  have e_1943_19 := arithEq_of_rows h (row := 1943) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1943_19
  simp only [← c118629, ← c118630, k2, ← c118631] at e_1943_19
  have hr := e_1943_19
  simp only [e_1943_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19021 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 7) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10997)) := by
  have c118632 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.1
  have c118633 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.1
  have c118634 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118635 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118636 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118637 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_0 := arithEq_of_rows h (row := 1944) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_0
  simp only [← c118632, ← c118633, ← c118634] at e_1944_0
  have e_1944_1 := arithEq_of_rows h (row := 1944) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_1
  simp only [← c118635, ← c118636, k2, ← c118637] at e_1944_1
  have hr := e_1944_1
  simp only [e_1944_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19022 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 15) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10998)) := by
  have c118638 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118639 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118640 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118641 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118642 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118643 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_2 := arithEq_of_rows h (row := 1944) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_2
  simp only [← c118638, ← c118639, ← c118640] at e_1944_2
  have e_1944_3 := arithEq_of_rows h (row := 1944) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_3
  simp only [← c118641, ← c118642, k2, ← c118643] at e_1944_3
  have hr := e_1944_3
  simp only [e_1944_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19023 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 23) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 10999)) := by
  have c118644 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118645 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118646 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118647 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118648 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118649 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_4 := arithEq_of_rows h (row := 1944) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_4
  simp only [← c118644, ← c118645, ← c118646] at e_1944_4
  have e_1944_5 := arithEq_of_rows h (row := 1944) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_5
  simp only [← c118647, ← c118648, k2, ← c118649] at e_1944_5
  have hr := e_1944_5
  simp only [e_1944_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19024 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 31) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 11000)) := by
  have c118650 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118651 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118652 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118653 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118654 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118655 := (publicBatchWrapper89_copies3707 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_6 := arithEq_of_rows h (row := 1944) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_6
  simp only [← c118650, ← c118651, ← c118652] at e_1944_6
  have e_1944_7 := arithEq_of_rows h (row := 1944) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_7
  simp only [← c118653, ← c118654, k2, ← c118655] at e_1944_7
  have hr := e_1944_7
  simp only [e_1944_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19025 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 39) = bselect (a (.wire 53 35)) (a (.virt 200)) (a (.virt 11001)) := by
  have c118656 := (publicBatchWrapper89_copies3708 a h).1
  have c118657 := (publicBatchWrapper89_copies3708 a h).2.1
  have c118658 := (publicBatchWrapper89_copies3708 a h).2.2.1
  have c118659 := (publicBatchWrapper89_copies3708 a h).2.2.2.1
  have c118660 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.1
  have c118661 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_8 := arithEq_of_rows h (row := 1944) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_8
  simp only [← c118656, ← c118657, ← c118658] at e_1944_8
  have e_1944_9 := arithEq_of_rows h (row := 1944) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_9
  simp only [← c118659, ← c118660, k2, ← c118661] at e_1944_9
  have hr := e_1944_9
  simp only [e_1944_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19026 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11170)) := by
  have c118662 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.1
  have c118663 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.1
  have c118664 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.1
  have c118665 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.1
  have c118666 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118667 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_10 := arithEq_of_rows h (row := 1944) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_10
  simp only [← c118662, ← c118663, ← c118664] at e_1944_10
  have e_1944_11 := arithEq_of_rows h (row := 1944) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_11
  simp only [← c118665, ← c118666, k2, ← c118667] at e_1944_11
  have hr := e_1944_11
  simp only [e_1944_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19027 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11171)) := by
  have c118668 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118669 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118670 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118671 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118672 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118673 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_12 := arithEq_of_rows h (row := 1944) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_12
  simp only [← c118668, ← c118669, ← c118670] at e_1944_12
  have e_1944_13 := arithEq_of_rows h (row := 1944) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_13
  simp only [← c118671, ← c118672, k2, ← c118673] at e_1944_13
  have hr := e_1944_13
  simp only [e_1944_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19028 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11172)) := by
  have c118674 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118675 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118676 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118677 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118678 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118679 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_14 := arithEq_of_rows h (row := 1944) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_14
  simp only [← c118674, ← c118675, ← c118676] at e_1944_14
  have e_1944_15 := arithEq_of_rows h (row := 1944) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_15
  simp only [← c118677, ← c118678, k2, ← c118679] at e_1944_15
  have hr := e_1944_15
  simp only [e_1944_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19029 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11173)) := by
  have c118680 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118681 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118682 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118683 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118684 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118685 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_16 := arithEq_of_rows h (row := 1944) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_16
  simp only [← c118680, ← c118681, ← c118682] at e_1944_16
  have e_1944_17 := arithEq_of_rows h (row := 1944) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_17
  simp only [← c118683, ← c118684, k2, ← c118685] at e_1944_17
  have hr := e_1944_17
  simp only [e_1944_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19030 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1944 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11174)) := by
  have c118686 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118687 := (publicBatchWrapper89_copies3708 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118688 := (publicBatchWrapper89_copies3709 a h).1
  have c118689 := (publicBatchWrapper89_copies3709 a h).2.1
  have c118690 := (publicBatchWrapper89_copies3709 a h).2.2.1
  have c118691 := (publicBatchWrapper89_copies3709 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1944_18 := arithEq_of_rows h (row := 1944) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_18
  simp only [← c118686, ← c118687, ← c118688] at e_1944_18
  have e_1944_19 := arithEq_of_rows h (row := 1944) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1944_19
  simp only [← c118689, ← c118690, k2, ← c118691] at e_1944_19
  have hr := e_1944_19
  simp only [e_1944_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19031 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11175)) := by
  have c118692 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.1
  have c118693 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.1
  have c118694 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.1
  have c118695 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.1
  have c118696 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.1
  have c118697 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_0 := arithEq_of_rows h (row := 1945) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_0
  simp only [← c118692, ← c118693, ← c118694] at e_1945_0
  have e_1945_1 := arithEq_of_rows h (row := 1945) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_1
  simp only [← c118695, ← c118696, k2, ← c118697] at e_1945_1
  have hr := e_1945_1
  simp only [e_1945_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19032 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11176)) := by
  have c118698 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118699 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118700 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118701 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118702 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118703 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_2 := arithEq_of_rows h (row := 1945) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_2
  simp only [← c118698, ← c118699, ← c118700] at e_1945_2
  have e_1945_3 := arithEq_of_rows h (row := 1945) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_3
  simp only [← c118701, ← c118702, k2, ← c118703] at e_1945_3
  have hr := e_1945_3
  simp only [e_1945_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19033 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11177)) := by
  have c118704 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118705 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118706 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118707 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118708 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118709 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_4 := arithEq_of_rows h (row := 1945) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_4
  simp only [← c118704, ← c118705, ← c118706] at e_1945_4
  have e_1945_5 := arithEq_of_rows h (row := 1945) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_5
  simp only [← c118707, ← c118708, k2, ← c118709] at e_1945_5
  have hr := e_1945_5
  simp only [e_1945_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19034 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11178)) := by
  have c118710 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118711 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118712 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118713 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118714 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118715 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_6 := arithEq_of_rows h (row := 1945) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_6
  simp only [← c118710, ← c118711, ← c118712] at e_1945_6
  have e_1945_7 := arithEq_of_rows h (row := 1945) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_7
  simp only [← c118713, ← c118714, k2, ← c118715] at e_1945_7
  have hr := e_1945_7
  simp only [e_1945_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19035 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11179)) := by
  have c118716 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118717 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118718 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118719 := (publicBatchWrapper89_copies3709 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118720 := (publicBatchWrapper89_copies3710 a h).1
  have c118721 := (publicBatchWrapper89_copies3710 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_8 := arithEq_of_rows h (row := 1945) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_8
  simp only [← c118716, ← c118717, ← c118718] at e_1945_8
  have e_1945_9 := arithEq_of_rows h (row := 1945) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_9
  simp only [← c118719, ← c118720, k2, ← c118721] at e_1945_9
  have hr := e_1945_9
  simp only [e_1945_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19036 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11180)) := by
  have c118722 := (publicBatchWrapper89_copies3710 a h).2.2.1
  have c118723 := (publicBatchWrapper89_copies3710 a h).2.2.2.1
  have c118724 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.1
  have c118725 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.1
  have c118726 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.1
  have c118727 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_10 := arithEq_of_rows h (row := 1945) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_10
  simp only [← c118722, ← c118723, ← c118724] at e_1945_10
  have e_1945_11 := arithEq_of_rows h (row := 1945) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_11
  simp only [← c118725, ← c118726, k2, ← c118727] at e_1945_11
  have hr := e_1945_11
  simp only [e_1945_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19037 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11181)) := by
  have c118728 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.1
  have c118729 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.1
  have c118730 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118731 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118732 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118733 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_12 := arithEq_of_rows h (row := 1945) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_12
  simp only [← c118728, ← c118729, ← c118730] at e_1945_12
  have e_1945_13 := arithEq_of_rows h (row := 1945) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_13
  simp only [← c118731, ← c118732, k2, ← c118733] at e_1945_13
  have hr := e_1945_13
  simp only [e_1945_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19038 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11182)) := by
  have c118734 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118735 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118736 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118737 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118738 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118739 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_14 := arithEq_of_rows h (row := 1945) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_14
  simp only [← c118734, ← c118735, ← c118736] at e_1945_14
  have e_1945_15 := arithEq_of_rows h (row := 1945) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_15
  simp only [← c118737, ← c118738, k2, ← c118739] at e_1945_15
  have hr := e_1945_15
  simp only [e_1945_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19039 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11183)) := by
  have c118740 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118741 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118742 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118743 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118744 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118745 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_16 := arithEq_of_rows h (row := 1945) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_16
  simp only [← c118740, ← c118741, ← c118742] at e_1945_16
  have e_1945_17 := arithEq_of_rows h (row := 1945) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_17
  simp only [← c118743, ← c118744, k2, ← c118745] at e_1945_17
  have hr := e_1945_17
  simp only [e_1945_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19040 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1945 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11184)) := by
  have c118746 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118747 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118748 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118749 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118750 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118751 := (publicBatchWrapper89_copies3710 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1945_18 := arithEq_of_rows h (row := 1945) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_18
  simp only [← c118746, ← c118747, ← c118748] at e_1945_18
  have e_1945_19 := arithEq_of_rows h (row := 1945) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1945_19
  simp only [← c118749, ← c118750, k2, ← c118751] at e_1945_19
  have hr := e_1945_19
  simp only [e_1945_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11185)) := by
  have c118752 := (publicBatchWrapper89_copies3711 a h).1
  have c118753 := (publicBatchWrapper89_copies3711 a h).2.1
  have c118754 := (publicBatchWrapper89_copies3711 a h).2.2.1
  have c118755 := (publicBatchWrapper89_copies3711 a h).2.2.2.1
  have c118756 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.1
  have c118757 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_0 := arithEq_of_rows h (row := 1946) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_0
  simp only [← c118752, ← c118753, ← c118754] at e_1946_0
  have e_1946_1 := arithEq_of_rows h (row := 1946) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_1
  simp only [← c118755, ← c118756, k2, ← c118757] at e_1946_1
  have hr := e_1946_1
  simp only [e_1946_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11186)) := by
  have c118758 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.1
  have c118759 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.1
  have c118760 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.1
  have c118761 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.1
  have c118762 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118763 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_2 := arithEq_of_rows h (row := 1946) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_2
  simp only [← c118758, ← c118759, ← c118760] at e_1946_2
  have e_1946_3 := arithEq_of_rows h (row := 1946) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_3
  simp only [← c118761, ← c118762, k2, ← c118763] at e_1946_3
  have hr := e_1946_3
  simp only [e_1946_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11187)) := by
  have c118764 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118765 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118766 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118767 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118768 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118769 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_4 := arithEq_of_rows h (row := 1946) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_4
  simp only [← c118764, ← c118765, ← c118766] at e_1946_4
  have e_1946_5 := arithEq_of_rows h (row := 1946) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_5
  simp only [← c118767, ← c118768, k2, ← c118769] at e_1946_5
  have hr := e_1946_5
  simp only [e_1946_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11188)) := by
  have c118770 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118771 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118772 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118773 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118774 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118775 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_6 := arithEq_of_rows h (row := 1946) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_6
  simp only [← c118770, ← c118771, ← c118772] at e_1946_6
  have e_1946_7 := arithEq_of_rows h (row := 1946) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_7
  simp only [← c118773, ← c118774, k2, ← c118775] at e_1946_7
  have hr := e_1946_7
  simp only [e_1946_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11189)) := by
  have c118776 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118777 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118778 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118779 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118780 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118781 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_8 := arithEq_of_rows h (row := 1946) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_8
  simp only [← c118776, ← c118777, ← c118778] at e_1946_8
  have e_1946_9 := arithEq_of_rows h (row := 1946) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_9
  simp only [← c118779, ← c118780, k2, ← c118781] at e_1946_9
  have hr := e_1946_9
  simp only [e_1946_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11190)) := by
  have c118782 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118783 := (publicBatchWrapper89_copies3711 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118784 := (publicBatchWrapper89_copies3712 a h).1
  have c118785 := (publicBatchWrapper89_copies3712 a h).2.1
  have c118786 := (publicBatchWrapper89_copies3712 a h).2.2.1
  have c118787 := (publicBatchWrapper89_copies3712 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_10 := arithEq_of_rows h (row := 1946) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_10
  simp only [← c118782, ← c118783, ← c118784] at e_1946_10
  have e_1946_11 := arithEq_of_rows h (row := 1946) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_11
  simp only [← c118785, ← c118786, k2, ← c118787] at e_1946_11
  have hr := e_1946_11
  simp only [e_1946_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11191)) := by
  have c118788 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.1
  have c118789 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.1
  have c118790 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.1
  have c118791 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.1
  have c118792 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.1
  have c118793 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_12 := arithEq_of_rows h (row := 1946) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_12
  simp only [← c118788, ← c118789, ← c118790] at e_1946_12
  have e_1946_13 := arithEq_of_rows h (row := 1946) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_13
  simp only [← c118791, ← c118792, k2, ← c118793] at e_1946_13
  have hr := e_1946_13
  simp only [e_1946_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 63) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11192)) := by
  have c118794 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118795 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118796 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118797 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118798 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118799 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_14 := arithEq_of_rows h (row := 1946) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_14
  simp only [← c118794, ← c118795, ← c118796] at e_1946_14
  have e_1946_15 := arithEq_of_rows h (row := 1946) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_15
  simp only [← c118797, ← c118798, k2, ← c118799] at e_1946_15
  have hr := e_1946_15
  simp only [e_1946_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 71) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11193)) := by
  have c118800 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118801 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118802 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118803 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118804 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118805 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_16 := arithEq_of_rows h (row := 1946) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_16
  simp only [← c118800, ← c118801, ← c118802] at e_1946_16
  have e_1946_17 := arithEq_of_rows h (row := 1946) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_17
  simp only [← c118803, ← c118804, k2, ← c118805] at e_1946_17
  have hr := e_1946_17
  simp only [e_1946_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1946 79) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11194)) := by
  have c118806 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118807 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118808 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118809 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118810 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118811 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1946_18 := arithEq_of_rows h (row := 1946) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_18
  simp only [← c118806, ← c118807, ← c118808] at e_1946_18
  have e_1946_19 := arithEq_of_rows h (row := 1946) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1946_19
  simp only [← c118809, ← c118810, k2, ← c118811] at e_1946_19
  have hr := e_1946_19
  simp only [e_1946_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 7) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11195)) := by
  have c118812 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118813 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118814 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118815 := (publicBatchWrapper89_copies3712 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118816 := (publicBatchWrapper89_copies3713 a h).1
  have c118817 := (publicBatchWrapper89_copies3713 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_0 := arithEq_of_rows h (row := 1947) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_0
  simp only [← c118812, ← c118813, ← c118814] at e_1947_0
  have e_1947_1 := arithEq_of_rows h (row := 1947) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_1
  simp only [← c118815, ← c118816, k2, ← c118817] at e_1947_1
  have hr := e_1947_1
  simp only [e_1947_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 15) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11196)) := by
  have c118818 := (publicBatchWrapper89_copies3713 a h).2.2.1
  have c118819 := (publicBatchWrapper89_copies3713 a h).2.2.2.1
  have c118820 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.1
  have c118821 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.1
  have c118822 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.1
  have c118823 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_2 := arithEq_of_rows h (row := 1947) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_2
  simp only [← c118818, ← c118819, ← c118820] at e_1947_2
  have e_1947_3 := arithEq_of_rows h (row := 1947) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_3
  simp only [← c118821, ← c118822, k2, ← c118823] at e_1947_3
  have hr := e_1947_3
  simp only [e_1947_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 23) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11197)) := by
  have c118824 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.1
  have c118825 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.1
  have c118826 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118827 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118828 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118829 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_4 := arithEq_of_rows h (row := 1947) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_4
  simp only [← c118824, ← c118825, ← c118826] at e_1947_4
  have e_1947_5 := arithEq_of_rows h (row := 1947) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_5
  simp only [← c118827, ← c118828, k2, ← c118829] at e_1947_5
  have hr := e_1947_5
  simp only [e_1947_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 31) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11198)) := by
  have c118830 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118831 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118832 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118833 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118834 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118835 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_6 := arithEq_of_rows h (row := 1947) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_6
  simp only [← c118830, ← c118831, ← c118832] at e_1947_6
  have e_1947_7 := arithEq_of_rows h (row := 1947) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_7
  simp only [← c118833, ← c118834, k2, ← c118835] at e_1947_7
  have hr := e_1947_7
  simp only [e_1947_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 39) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11199)) := by
  have c118836 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118837 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118838 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118839 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118840 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118841 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_8 := arithEq_of_rows h (row := 1947) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_8
  simp only [← c118836, ← c118837, ← c118838] at e_1947_8
  have e_1947_9 := arithEq_of_rows h (row := 1947) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_9
  simp only [← c118839, ← c118840, k2, ← c118841] at e_1947_9
  have hr := e_1947_9
  simp only [e_1947_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 47) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11200)) := by
  have c118842 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118843 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118844 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118845 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118846 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118847 := (publicBatchWrapper89_copies3713 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_10 := arithEq_of_rows h (row := 1947) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_10
  simp only [← c118842, ← c118843, ← c118844] at e_1947_10
  have e_1947_11 := arithEq_of_rows h (row := 1947) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_11
  simp only [← c118845, ← c118846, k2, ← c118847] at e_1947_11
  have hr := e_1947_11
  simp only [e_1947_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 55) = bselect (a (.wire 53 79)) (a (.virt 200)) (a (.virt 11201)) := by
  have c118848 := (publicBatchWrapper89_copies3714 a h).1
  have c118849 := (publicBatchWrapper89_copies3714 a h).2.1
  have c118850 := (publicBatchWrapper89_copies3714 a h).2.2.1
  have c118851 := (publicBatchWrapper89_copies3714 a h).2.2.2.1
  have c118852 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.1
  have c118853 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_12 := arithEq_of_rows h (row := 1947) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_12
  simp only [← c118848, ← c118849, ← c118850] at e_1947_12
  have e_1947_13 := arithEq_of_rows h (row := 1947) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_13
  simp only [← c118851, ← c118852, k2, ← c118853] at e_1947_13
  have hr := e_1947_13
  simp only [e_1947_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11370)) := by
  have c118854 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.1
  have c118855 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.1
  have c118856 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.1
  have c118857 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.1
  have c118858 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118859 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_14 := arithEq_of_rows h (row := 1947) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_14
  simp only [← c118854, ← c118855, ← c118856] at e_1947_14
  have e_1947_15 := arithEq_of_rows h (row := 1947) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_15
  simp only [← c118857, ← c118858, k2, ← c118859] at e_1947_15
  have hr := e_1947_15
  simp only [e_1947_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11371)) := by
  have c118860 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118861 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118862 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118863 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118864 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118865 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_16 := arithEq_of_rows h (row := 1947) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_16
  simp only [← c118860, ← c118861, ← c118862] at e_1947_16
  have e_1947_17 := arithEq_of_rows h (row := 1947) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_17
  simp only [← c118863, ← c118864, k2, ← c118865] at e_1947_17
  have hr := e_1947_17
  simp only [e_1947_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1947 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11372)) := by
  have c118866 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118867 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118868 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118869 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118870 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118871 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1947_18 := arithEq_of_rows h (row := 1947) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_18
  simp only [← c118866, ← c118867, ← c118868] at e_1947_18
  have e_1947_19 := arithEq_of_rows h (row := 1947) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1947_19
  simp only [← c118869, ← c118870, k2, ← c118871] at e_1947_19
  have hr := e_1947_19
  simp only [e_1947_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11373)) := by
  have c118872 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118873 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118874 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118875 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118876 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118877 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_0 := arithEq_of_rows h (row := 1948) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_0
  simp only [← c118872, ← c118873, ← c118874] at e_1948_0
  have e_1948_1 := arithEq_of_rows h (row := 1948) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_1
  simp only [← c118875, ← c118876, k2, ← c118877] at e_1948_1
  have hr := e_1948_1
  simp only [e_1948_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11374)) := by
  have c118878 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118879 := (publicBatchWrapper89_copies3714 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118880 := (publicBatchWrapper89_copies3715 a h).1
  have c118881 := (publicBatchWrapper89_copies3715 a h).2.1
  have c118882 := (publicBatchWrapper89_copies3715 a h).2.2.1
  have c118883 := (publicBatchWrapper89_copies3715 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_2 := arithEq_of_rows h (row := 1948) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_2
  simp only [← c118878, ← c118879, ← c118880] at e_1948_2
  have e_1948_3 := arithEq_of_rows h (row := 1948) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_3
  simp only [← c118881, ← c118882, k2, ← c118883] at e_1948_3
  have hr := e_1948_3
  simp only [e_1948_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11375)) := by
  have c118884 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.1
  have c118885 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.1
  have c118886 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.1
  have c118887 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.1
  have c118888 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.1
  have c118889 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_4 := arithEq_of_rows h (row := 1948) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_4
  simp only [← c118884, ← c118885, ← c118886] at e_1948_4
  have e_1948_5 := arithEq_of_rows h (row := 1948) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_5
  simp only [← c118887, ← c118888, k2, ← c118889] at e_1948_5
  have hr := e_1948_5
  simp only [e_1948_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11376)) := by
  have c118890 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118891 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118892 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118893 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118894 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118895 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_6 := arithEq_of_rows h (row := 1948) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_6
  simp only [← c118890, ← c118891, ← c118892] at e_1948_6
  have e_1948_7 := arithEq_of_rows h (row := 1948) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_7
  simp only [← c118893, ← c118894, k2, ← c118895] at e_1948_7
  have hr := e_1948_7
  simp only [e_1948_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11377)) := by
  have c118896 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118897 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118898 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118899 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118900 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118901 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_8 := arithEq_of_rows h (row := 1948) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_8
  simp only [← c118896, ← c118897, ← c118898] at e_1948_8
  have e_1948_9 := arithEq_of_rows h (row := 1948) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_9
  simp only [← c118899, ← c118900, k2, ← c118901] at e_1948_9
  have hr := e_1948_9
  simp only [e_1948_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11378)) := by
  have c118902 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118903 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118904 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118905 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118906 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118907 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_10 := arithEq_of_rows h (row := 1948) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_10
  simp only [← c118902, ← c118903, ← c118904] at e_1948_10
  have e_1948_11 := arithEq_of_rows h (row := 1948) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_11
  simp only [← c118905, ← c118906, k2, ← c118907] at e_1948_11
  have hr := e_1948_11
  simp only [e_1948_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11379)) := by
  have c118908 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118909 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118910 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118911 := (publicBatchWrapper89_copies3715 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118912 := (publicBatchWrapper89_copies3716 a h).1
  have c118913 := (publicBatchWrapper89_copies3716 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_12 := arithEq_of_rows h (row := 1948) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_12
  simp only [← c118908, ← c118909, ← c118910] at e_1948_12
  have e_1948_13 := arithEq_of_rows h (row := 1948) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_13
  simp only [← c118911, ← c118912, k2, ← c118913] at e_1948_13
  have hr := e_1948_13
  simp only [e_1948_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11380)) := by
  have c118914 := (publicBatchWrapper89_copies3716 a h).2.2.1
  have c118915 := (publicBatchWrapper89_copies3716 a h).2.2.2.1
  have c118916 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.1
  have c118917 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.1
  have c118918 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.1
  have c118919 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_14 := arithEq_of_rows h (row := 1948) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_14
  simp only [← c118914, ← c118915, ← c118916] at e_1948_14
  have e_1948_15 := arithEq_of_rows h (row := 1948) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_15
  simp only [← c118917, ← c118918, k2, ← c118919] at e_1948_15
  have hr := e_1948_15
  simp only [e_1948_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11381)) := by
  have c118920 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.1
  have c118921 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.1
  have c118922 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118923 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118924 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118925 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_16 := arithEq_of_rows h (row := 1948) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_16
  simp only [← c118920, ← c118921, ← c118922] at e_1948_16
  have e_1948_17 := arithEq_of_rows h (row := 1948) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_17
  simp only [← c118923, ← c118924, k2, ← c118925] at e_1948_17
  have hr := e_1948_17
  simp only [e_1948_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1948 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11382)) := by
  have c118926 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118927 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118928 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118929 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118930 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118931 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1948_18 := arithEq_of_rows h (row := 1948) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_18
  simp only [← c118926, ← c118927, ← c118928] at e_1948_18
  have e_1948_19 := arithEq_of_rows h (row := 1948) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1948_19
  simp only [← c118929, ← c118930, k2, ← c118931] at e_1948_19
  have hr := e_1948_19
  simp only [e_1948_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11383)) := by
  have c118932 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118933 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118934 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118935 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118936 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118937 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_0 := arithEq_of_rows h (row := 1949) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_0
  simp only [← c118932, ← c118933, ← c118934] at e_1949_0
  have e_1949_1 := arithEq_of_rows h (row := 1949) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_1
  simp only [← c118935, ← c118936, k2, ← c118937] at e_1949_1
  have hr := e_1949_1
  simp only [e_1949_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11384)) := by
  have c118938 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118939 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118940 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118941 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118942 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118943 := (publicBatchWrapper89_copies3716 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_2 := arithEq_of_rows h (row := 1949) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_2
  simp only [← c118938, ← c118939, ← c118940] at e_1949_2
  have e_1949_3 := arithEq_of_rows h (row := 1949) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_3
  simp only [← c118941, ← c118942, k2, ← c118943] at e_1949_3
  have hr := e_1949_3
  simp only [e_1949_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11385)) := by
  have c118944 := (publicBatchWrapper89_copies3717 a h).1
  have c118945 := (publicBatchWrapper89_copies3717 a h).2.1
  have c118946 := (publicBatchWrapper89_copies3717 a h).2.2.1
  have c118947 := (publicBatchWrapper89_copies3717 a h).2.2.2.1
  have c118948 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.1
  have c118949 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_4 := arithEq_of_rows h (row := 1949) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_4
  simp only [← c118944, ← c118945, ← c118946] at e_1949_4
  have e_1949_5 := arithEq_of_rows h (row := 1949) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_5
  simp only [← c118947, ← c118948, k2, ← c118949] at e_1949_5
  have hr := e_1949_5
  simp only [e_1949_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11386)) := by
  have c118950 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.1
  have c118951 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.1
  have c118952 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.1
  have c118953 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.1
  have c118954 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118955 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_6 := arithEq_of_rows h (row := 1949) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_6
  simp only [← c118950, ← c118951, ← c118952] at e_1949_6
  have e_1949_7 := arithEq_of_rows h (row := 1949) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_7
  simp only [← c118953, ← c118954, k2, ← c118955] at e_1949_7
  have hr := e_1949_7
  simp only [e_1949_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11387)) := by
  have c118956 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118957 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118958 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118959 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118960 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118961 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_8 := arithEq_of_rows h (row := 1949) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_8
  simp only [← c118956, ← c118957, ← c118958] at e_1949_8
  have e_1949_9 := arithEq_of_rows h (row := 1949) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_9
  simp only [← c118959, ← c118960, k2, ← c118961] at e_1949_9
  have hr := e_1949_9
  simp only [e_1949_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11388)) := by
  have c118962 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118963 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118964 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118965 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118966 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118967 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_10 := arithEq_of_rows h (row := 1949) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_10
  simp only [← c118962, ← c118963, ← c118964] at e_1949_10
  have e_1949_11 := arithEq_of_rows h (row := 1949) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_11
  simp only [← c118965, ← c118966, k2, ← c118967] at e_1949_11
  have hr := e_1949_11
  simp only [e_1949_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11389)) := by
  have c118968 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118969 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118970 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118971 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118972 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118973 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_12 := arithEq_of_rows h (row := 1949) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_12
  simp only [← c118968, ← c118969, ← c118970] at e_1949_12
  have e_1949_13 := arithEq_of_rows h (row := 1949) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_13
  simp only [← c118971, ← c118972, k2, ← c118973] at e_1949_13
  have hr := e_1949_13
  simp only [e_1949_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11390)) := by
  have c118974 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118975 := (publicBatchWrapper89_copies3717 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c118976 := (publicBatchWrapper89_copies3718 a h).1
  have c118977 := (publicBatchWrapper89_copies3718 a h).2.1
  have c118978 := (publicBatchWrapper89_copies3718 a h).2.2.1
  have c118979 := (publicBatchWrapper89_copies3718 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_14 := arithEq_of_rows h (row := 1949) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_14
  simp only [← c118974, ← c118975, ← c118976] at e_1949_14
  have e_1949_15 := arithEq_of_rows h (row := 1949) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_15
  simp only [← c118977, ← c118978, k2, ← c118979] at e_1949_15
  have hr := e_1949_15
  simp only [e_1949_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11391)) := by
  have c118980 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.1
  have c118981 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.1
  have c118982 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.1
  have c118983 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.1
  have c118984 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.1
  have c118985 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_16 := arithEq_of_rows h (row := 1949) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_16
  simp only [← c118980, ← c118981, ← c118982] at e_1949_16
  have e_1949_17 := arithEq_of_rows h (row := 1949) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_17
  simp only [← c118983, ← c118984, k2, ← c118985] at e_1949_17
  have hr := e_1949_17
  simp only [e_1949_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1949 79) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11392)) := by
  have c118986 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.1
  have c118987 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c118988 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118989 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118990 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118991 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1949_18 := arithEq_of_rows h (row := 1949) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_18
  simp only [← c118986, ← c118987, ← c118988] at e_1949_18
  have e_1949_19 := arithEq_of_rows h (row := 1949) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1949_19
  simp only [← c118989, ← c118990, k2, ← c118991] at e_1949_19
  have hr := e_1949_19
  simp only [e_1949_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 7) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11393)) := by
  have c118992 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118993 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118994 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118995 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118996 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118997 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_0 := arithEq_of_rows h (row := 1950) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_0
  simp only [← c118992, ← c118993, ← c118994] at e_1950_0
  have e_1950_1 := arithEq_of_rows h (row := 1950) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_1
  simp only [← c118995, ← c118996, k2, ← c118997] at e_1950_1
  have hr := e_1950_1
  simp only [e_1950_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 15) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11394)) := by
  have c118998 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c118999 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119000 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119001 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119002 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119003 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_2 := arithEq_of_rows h (row := 1950) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_2
  simp only [← c118998, ← c118999, ← c119000] at e_1950_2
  have e_1950_3 := arithEq_of_rows h (row := 1950) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_3
  simp only [← c119001, ← c119002, k2, ← c119003] at e_1950_3
  have hr := e_1950_3
  simp only [e_1950_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 23) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11395)) := by
  have c119004 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119005 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119006 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119007 := (publicBatchWrapper89_copies3718 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119008 := (publicBatchWrapper89_copies3719 a h).1
  have c119009 := (publicBatchWrapper89_copies3719 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_4 := arithEq_of_rows h (row := 1950) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_4
  simp only [← c119004, ← c119005, ← c119006] at e_1950_4
  have e_1950_5 := arithEq_of_rows h (row := 1950) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_5
  simp only [← c119007, ← c119008, k2, ← c119009] at e_1950_5
  have hr := e_1950_5
  simp only [e_1950_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 31) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11396)) := by
  have c119010 := (publicBatchWrapper89_copies3719 a h).2.2.1
  have c119011 := (publicBatchWrapper89_copies3719 a h).2.2.2.1
  have c119012 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.1
  have c119013 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.1
  have c119014 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.1
  have c119015 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_6 := arithEq_of_rows h (row := 1950) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_6
  simp only [← c119010, ← c119011, ← c119012] at e_1950_6
  have e_1950_7 := arithEq_of_rows h (row := 1950) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_7
  simp only [← c119013, ← c119014, k2, ← c119015] at e_1950_7
  have hr := e_1950_7
  simp only [e_1950_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 39) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11397)) := by
  have c119016 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.1
  have c119017 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.1
  have c119018 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119019 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119020 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119021 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_8 := arithEq_of_rows h (row := 1950) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_8
  simp only [← c119016, ← c119017, ← c119018] at e_1950_8
  have e_1950_9 := arithEq_of_rows h (row := 1950) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_9
  simp only [← c119019, ← c119020, k2, ← c119021] at e_1950_9
  have hr := e_1950_9
  simp only [e_1950_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 47) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11398)) := by
  have c119022 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119023 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119024 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119025 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119026 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119027 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_10 := arithEq_of_rows h (row := 1950) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_10
  simp only [← c119022, ← c119023, ← c119024] at e_1950_10
  have e_1950_11 := arithEq_of_rows h (row := 1950) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_11
  simp only [← c119025, ← c119026, k2, ← c119027] at e_1950_11
  have hr := e_1950_11
  simp only [e_1950_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 55) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11399)) := by
  have c119028 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119029 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119030 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119031 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119032 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119033 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_12 := arithEq_of_rows h (row := 1950) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_12
  simp only [← c119028, ← c119029, ← c119030] at e_1950_12
  have e_1950_13 := arithEq_of_rows h (row := 1950) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_13
  simp only [← c119031, ← c119032, k2, ← c119033] at e_1950_13
  have hr := e_1950_13
  simp only [e_1950_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 63) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11400)) := by
  have c119034 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119035 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119036 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119037 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119038 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119039 := (publicBatchWrapper89_copies3719 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_14 := arithEq_of_rows h (row := 1950) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_14
  simp only [← c119034, ← c119035, ← c119036] at e_1950_14
  have e_1950_15 := arithEq_of_rows h (row := 1950) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_15
  simp only [← c119037, ← c119038, k2, ← c119039] at e_1950_15
  have hr := e_1950_15
  simp only [e_1950_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 71) = bselect (a (.wire 55 43)) (a (.virt 200)) (a (.virt 11401)) := by
  have c119040 := (publicBatchWrapper89_copies3720 a h).1
  have c119041 := (publicBatchWrapper89_copies3720 a h).2.1
  have c119042 := (publicBatchWrapper89_copies3720 a h).2.2.1
  have c119043 := (publicBatchWrapper89_copies3720 a h).2.2.2.1
  have c119044 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.1
  have c119045 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_16 := arithEq_of_rows h (row := 1950) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_16
  simp only [← c119040, ← c119041, ← c119042] at e_1950_16
  have e_1950_17 := arithEq_of_rows h (row := 1950) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_17
  simp only [← c119043, ← c119044, k2, ← c119045] at e_1950_17
  have hr := e_1950_17
  simp only [e_1950_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1950 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11570)) := by
  have c119046 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.1
  have c119047 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.1
  have c119048 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.1
  have c119049 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.1
  have c119050 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119051 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1950_18 := arithEq_of_rows h (row := 1950) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_18
  simp only [← c119046, ← c119047, ← c119048] at e_1950_18
  have e_1950_19 := arithEq_of_rows h (row := 1950) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1950_19
  simp only [← c119049, ← c119050, k2, ← c119051] at e_1950_19
  have hr := e_1950_19
  simp only [e_1950_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11571)) := by
  have c119052 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119053 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119054 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119055 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119056 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119057 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_0 := arithEq_of_rows h (row := 1951) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_0
  simp only [← c119052, ← c119053, ← c119054] at e_1951_0
  have e_1951_1 := arithEq_of_rows h (row := 1951) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_1
  simp only [← c119055, ← c119056, k2, ← c119057] at e_1951_1
  have hr := e_1951_1
  simp only [e_1951_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11572)) := by
  have c119058 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119059 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119060 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119061 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119062 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119063 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_2 := arithEq_of_rows h (row := 1951) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_2
  simp only [← c119058, ← c119059, ← c119060] at e_1951_2
  have e_1951_3 := arithEq_of_rows h (row := 1951) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_3
  simp only [← c119061, ← c119062, k2, ← c119063] at e_1951_3
  have hr := e_1951_3
  simp only [e_1951_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11573)) := by
  have c119064 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119065 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119066 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119067 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119068 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119069 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_4 := arithEq_of_rows h (row := 1951) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_4
  simp only [← c119064, ← c119065, ← c119066] at e_1951_4
  have e_1951_5 := arithEq_of_rows h (row := 1951) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_5
  simp only [← c119067, ← c119068, k2, ← c119069] at e_1951_5
  have hr := e_1951_5
  simp only [e_1951_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11574)) := by
  have c119070 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119071 := (publicBatchWrapper89_copies3720 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119072 := (publicBatchWrapper89_copies3721 a h).1
  have c119073 := (publicBatchWrapper89_copies3721 a h).2.1
  have c119074 := (publicBatchWrapper89_copies3721 a h).2.2.1
  have c119075 := (publicBatchWrapper89_copies3721 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_6 := arithEq_of_rows h (row := 1951) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_6
  simp only [← c119070, ← c119071, ← c119072] at e_1951_6
  have e_1951_7 := arithEq_of_rows h (row := 1951) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_7
  simp only [← c119073, ← c119074, k2, ← c119075] at e_1951_7
  have hr := e_1951_7
  simp only [e_1951_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11575)) := by
  have c119076 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.1
  have c119077 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.1
  have c119078 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.1
  have c119079 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.1
  have c119080 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.1
  have c119081 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_8 := arithEq_of_rows h (row := 1951) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_8
  simp only [← c119076, ← c119077, ← c119078] at e_1951_8
  have e_1951_9 := arithEq_of_rows h (row := 1951) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_9
  simp only [← c119079, ← c119080, k2, ← c119081] at e_1951_9
  have hr := e_1951_9
  simp only [e_1951_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11576)) := by
  have c119082 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119083 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119084 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119085 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119086 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119087 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_10 := arithEq_of_rows h (row := 1951) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_10
  simp only [← c119082, ← c119083, ← c119084] at e_1951_10
  have e_1951_11 := arithEq_of_rows h (row := 1951) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_11
  simp only [← c119085, ← c119086, k2, ← c119087] at e_1951_11
  have hr := e_1951_11
  simp only [e_1951_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11577)) := by
  have c119088 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119089 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119090 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119091 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119092 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119093 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_12 := arithEq_of_rows h (row := 1951) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_12
  simp only [← c119088, ← c119089, ← c119090] at e_1951_12
  have e_1951_13 := arithEq_of_rows h (row := 1951) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_13
  simp only [← c119091, ← c119092, k2, ← c119093] at e_1951_13
  have hr := e_1951_13
  simp only [e_1951_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11578)) := by
  have c119094 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119095 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119096 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119097 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119098 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119099 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_14 := arithEq_of_rows h (row := 1951) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_14
  simp only [← c119094, ← c119095, ← c119096] at e_1951_14
  have e_1951_15 := arithEq_of_rows h (row := 1951) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_15
  simp only [← c119097, ← c119098, k2, ← c119099] at e_1951_15
  have hr := e_1951_15
  simp only [e_1951_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11579)) := by
  have c119100 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119101 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119102 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119103 := (publicBatchWrapper89_copies3721 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119104 := (publicBatchWrapper89_copies3722 a h).1
  have c119105 := (publicBatchWrapper89_copies3722 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_16 := arithEq_of_rows h (row := 1951) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_16
  simp only [← c119100, ← c119101, ← c119102] at e_1951_16
  have e_1951_17 := arithEq_of_rows h (row := 1951) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_17
  simp only [← c119103, ← c119104, k2, ← c119105] at e_1951_17
  have hr := e_1951_17
  simp only [e_1951_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1951 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11580)) := by
  have c119106 := (publicBatchWrapper89_copies3722 a h).2.2.1
  have c119107 := (publicBatchWrapper89_copies3722 a h).2.2.2.1
  have c119108 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.1
  have c119109 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.1
  have c119110 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.1
  have c119111 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1951_18 := arithEq_of_rows h (row := 1951) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_18
  simp only [← c119106, ← c119107, ← c119108] at e_1951_18
  have e_1951_19 := arithEq_of_rows h (row := 1951) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1951_19
  simp only [← c119109, ← c119110, k2, ← c119111] at e_1951_19
  have hr := e_1951_19
  simp only [e_1951_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11581)) := by
  have c119112 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.1
  have c119113 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.1
  have c119114 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119115 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119116 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119117 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_0 := arithEq_of_rows h (row := 1952) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_0
  simp only [← c119112, ← c119113, ← c119114] at e_1952_0
  have e_1952_1 := arithEq_of_rows h (row := 1952) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_1
  simp only [← c119115, ← c119116, k2, ← c119117] at e_1952_1
  have hr := e_1952_1
  simp only [e_1952_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11582)) := by
  have c119118 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119119 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119120 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119121 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119122 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119123 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_2 := arithEq_of_rows h (row := 1952) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_2
  simp only [← c119118, ← c119119, ← c119120] at e_1952_2
  have e_1952_3 := arithEq_of_rows h (row := 1952) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_3
  simp only [← c119121, ← c119122, k2, ← c119123] at e_1952_3
  have hr := e_1952_3
  simp only [e_1952_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11583)) := by
  have c119124 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119125 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119126 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119127 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119128 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119129 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_4 := arithEq_of_rows h (row := 1952) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_4
  simp only [← c119124, ← c119125, ← c119126] at e_1952_4
  have e_1952_5 := arithEq_of_rows h (row := 1952) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_5
  simp only [← c119127, ← c119128, k2, ← c119129] at e_1952_5
  have hr := e_1952_5
  simp only [e_1952_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11584)) := by
  have c119130 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119131 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119132 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119133 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119134 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119135 := (publicBatchWrapper89_copies3722 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_6 := arithEq_of_rows h (row := 1952) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_6
  simp only [← c119130, ← c119131, ← c119132] at e_1952_6
  have e_1952_7 := arithEq_of_rows h (row := 1952) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_7
  simp only [← c119133, ← c119134, k2, ← c119135] at e_1952_7
  have hr := e_1952_7
  simp only [e_1952_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11585)) := by
  have c119136 := (publicBatchWrapper89_copies3723 a h).1
  have c119137 := (publicBatchWrapper89_copies3723 a h).2.1
  have c119138 := (publicBatchWrapper89_copies3723 a h).2.2.1
  have c119139 := (publicBatchWrapper89_copies3723 a h).2.2.2.1
  have c119140 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.1
  have c119141 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_8 := arithEq_of_rows h (row := 1952) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_8
  simp only [← c119136, ← c119137, ← c119138] at e_1952_8
  have e_1952_9 := arithEq_of_rows h (row := 1952) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_9
  simp only [← c119139, ← c119140, k2, ← c119141] at e_1952_9
  have hr := e_1952_9
  simp only [e_1952_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11586)) := by
  have c119142 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.1
  have c119143 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.1
  have c119144 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.1
  have c119145 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.1
  have c119146 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119147 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_10 := arithEq_of_rows h (row := 1952) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_10
  simp only [← c119142, ← c119143, ← c119144] at e_1952_10
  have e_1952_11 := arithEq_of_rows h (row := 1952) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_11
  simp only [← c119145, ← c119146, k2, ← c119147] at e_1952_11
  have hr := e_1952_11
  simp only [e_1952_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11587)) := by
  have c119148 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119149 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119150 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119151 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119152 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119153 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_12 := arithEq_of_rows h (row := 1952) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_12
  simp only [← c119148, ← c119149, ← c119150] at e_1952_12
  have e_1952_13 := arithEq_of_rows h (row := 1952) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_13
  simp only [← c119151, ← c119152, k2, ← c119153] at e_1952_13
  have hr := e_1952_13
  simp only [e_1952_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11588)) := by
  have c119154 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119155 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119156 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119157 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119158 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119159 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_14 := arithEq_of_rows h (row := 1952) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_14
  simp only [← c119154, ← c119155, ← c119156] at e_1952_14
  have e_1952_15 := arithEq_of_rows h (row := 1952) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_15
  simp only [← c119157, ← c119158, k2, ← c119159] at e_1952_15
  have hr := e_1952_15
  simp only [e_1952_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11589)) := by
  have c119160 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119161 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119162 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119163 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119164 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119165 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_16 := arithEq_of_rows h (row := 1952) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_16
  simp only [← c119160, ← c119161, ← c119162] at e_1952_16
  have e_1952_17 := arithEq_of_rows h (row := 1952) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_17
  simp only [← c119163, ← c119164, k2, ← c119165] at e_1952_17
  have hr := e_1952_17
  simp only [e_1952_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1952 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11590)) := by
  have c119166 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119167 := (publicBatchWrapper89_copies3723 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119168 := (publicBatchWrapper89_copies3724 a h).1
  have c119169 := (publicBatchWrapper89_copies3724 a h).2.1
  have c119170 := (publicBatchWrapper89_copies3724 a h).2.2.1
  have c119171 := (publicBatchWrapper89_copies3724 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1952_18 := arithEq_of_rows h (row := 1952) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_18
  simp only [← c119166, ← c119167, ← c119168] at e_1952_18
  have e_1952_19 := arithEq_of_rows h (row := 1952) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1952_19
  simp only [← c119169, ← c119170, k2, ← c119171] at e_1952_19
  have hr := e_1952_19
  simp only [e_1952_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11591)) := by
  have c119172 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.1
  have c119173 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.1
  have c119174 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.1
  have c119175 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.1
  have c119176 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.1
  have c119177 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_0 := arithEq_of_rows h (row := 1953) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_0
  simp only [← c119172, ← c119173, ← c119174] at e_1953_0
  have e_1953_1 := arithEq_of_rows h (row := 1953) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_1
  simp only [← c119175, ← c119176, k2, ← c119177] at e_1953_1
  have hr := e_1953_1
  simp only [e_1953_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 15) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11592)) := by
  have c119178 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119179 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119180 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119181 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119182 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119183 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_2 := arithEq_of_rows h (row := 1953) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_2
  simp only [← c119178, ← c119179, ← c119180] at e_1953_2
  have e_1953_3 := arithEq_of_rows h (row := 1953) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_3
  simp only [← c119181, ← c119182, k2, ← c119183] at e_1953_3
  have hr := e_1953_3
  simp only [e_1953_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 23) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11593)) := by
  have c119184 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119185 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119186 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119187 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119188 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119189 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_4 := arithEq_of_rows h (row := 1953) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_4
  simp only [← c119184, ← c119185, ← c119186] at e_1953_4
  have e_1953_5 := arithEq_of_rows h (row := 1953) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_5
  simp only [← c119187, ← c119188, k2, ← c119189] at e_1953_5
  have hr := e_1953_5
  simp only [e_1953_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19114 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 31) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11594)) := by
  have c119190 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119191 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119192 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119193 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119194 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119195 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_6 := arithEq_of_rows h (row := 1953) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_6
  simp only [← c119190, ← c119191, ← c119192] at e_1953_6
  have e_1953_7 := arithEq_of_rows h (row := 1953) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_7
  simp only [← c119193, ← c119194, k2, ← c119195] at e_1953_7
  have hr := e_1953_7
  simp only [e_1953_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19115 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 39) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11595)) := by
  have c119196 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119197 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119198 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119199 := (publicBatchWrapper89_copies3724 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119200 := (publicBatchWrapper89_copies3725 a h).1
  have c119201 := (publicBatchWrapper89_copies3725 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_8 := arithEq_of_rows h (row := 1953) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_8
  simp only [← c119196, ← c119197, ← c119198] at e_1953_8
  have e_1953_9 := arithEq_of_rows h (row := 1953) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_9
  simp only [← c119199, ← c119200, k2, ← c119201] at e_1953_9
  have hr := e_1953_9
  simp only [e_1953_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19116 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 47) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11596)) := by
  have c119202 := (publicBatchWrapper89_copies3725 a h).2.2.1
  have c119203 := (publicBatchWrapper89_copies3725 a h).2.2.2.1
  have c119204 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.1
  have c119205 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.1
  have c119206 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.1
  have c119207 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_10 := arithEq_of_rows h (row := 1953) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_10
  simp only [← c119202, ← c119203, ← c119204] at e_1953_10
  have e_1953_11 := arithEq_of_rows h (row := 1953) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_11
  simp only [← c119205, ← c119206, k2, ← c119207] at e_1953_11
  have hr := e_1953_11
  simp only [e_1953_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19117 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 55) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11597)) := by
  have c119208 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.1
  have c119209 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.1
  have c119210 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119211 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119212 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119213 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_12 := arithEq_of_rows h (row := 1953) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_12
  simp only [← c119208, ← c119209, ← c119210] at e_1953_12
  have e_1953_13 := arithEq_of_rows h (row := 1953) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_13
  simp only [← c119211, ← c119212, k2, ← c119213] at e_1953_13
  have hr := e_1953_13
  simp only [e_1953_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19118 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 63) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11598)) := by
  have c119214 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119215 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119216 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119217 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119218 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119219 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_14 := arithEq_of_rows h (row := 1953) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_14
  simp only [← c119214, ← c119215, ← c119216] at e_1953_14
  have e_1953_15 := arithEq_of_rows h (row := 1953) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_15
  simp only [← c119217, ← c119218, k2, ← c119219] at e_1953_15
  have hr := e_1953_15
  simp only [e_1953_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19119 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 71) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11599)) := by
  have c119220 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119221 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119222 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119223 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119224 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119225 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_16 := arithEq_of_rows h (row := 1953) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_16
  simp only [← c119220, ← c119221, ← c119222] at e_1953_16
  have e_1953_17 := arithEq_of_rows h (row := 1953) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_17
  simp only [← c119223, ← c119224, k2, ← c119225] at e_1953_17
  have hr := e_1953_17
  simp only [e_1953_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19120 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1953 79) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11600)) := by
  have c119226 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119227 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119228 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119229 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119230 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119231 := (publicBatchWrapper89_copies3725 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1953_18 := arithEq_of_rows h (row := 1953) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_18
  simp only [← c119226, ← c119227, ← c119228] at e_1953_18
  have e_1953_19 := arithEq_of_rows h (row := 1953) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1953_19
  simp only [← c119229, ← c119230, k2, ← c119231] at e_1953_19
  have hr := e_1953_19
  simp only [e_1953_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19121 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 7) = bselect (a (.wire 57 7)) (a (.virt 200)) (a (.virt 11601)) := by
  have c119232 := (publicBatchWrapper89_copies3726 a h).1
  have c119233 := (publicBatchWrapper89_copies3726 a h).2.1
  have c119234 := (publicBatchWrapper89_copies3726 a h).2.2.1
  have c119235 := (publicBatchWrapper89_copies3726 a h).2.2.2.1
  have c119236 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.1
  have c119237 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_0 := arithEq_of_rows h (row := 1954) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_0
  simp only [← c119232, ← c119233, ← c119234] at e_1954_0
  have e_1954_1 := arithEq_of_rows h (row := 1954) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_1
  simp only [← c119235, ← c119236, k2, ← c119237] at e_1954_1
  have hr := e_1954_1
  simp only [e_1954_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19122 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11770)) := by
  have c119238 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.1
  have c119239 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.1
  have c119240 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.1
  have c119241 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.1
  have c119242 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119243 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_2 := arithEq_of_rows h (row := 1954) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_2
  simp only [← c119238, ← c119239, ← c119240] at e_1954_2
  have e_1954_3 := arithEq_of_rows h (row := 1954) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_3
  simp only [← c119241, ← c119242, k2, ← c119243] at e_1954_3
  have hr := e_1954_3
  simp only [e_1954_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19123 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11771)) := by
  have c119244 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119245 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119246 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119247 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119248 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119249 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_4 := arithEq_of_rows h (row := 1954) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_4
  simp only [← c119244, ← c119245, ← c119246] at e_1954_4
  have e_1954_5 := arithEq_of_rows h (row := 1954) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_5
  simp only [← c119247, ← c119248, k2, ← c119249] at e_1954_5
  have hr := e_1954_5
  simp only [e_1954_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19124 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11772)) := by
  have c119250 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119251 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119252 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119253 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119254 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119255 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_6 := arithEq_of_rows h (row := 1954) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_6
  simp only [← c119250, ← c119251, ← c119252] at e_1954_6
  have e_1954_7 := arithEq_of_rows h (row := 1954) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_7
  simp only [← c119253, ← c119254, k2, ← c119255] at e_1954_7
  have hr := e_1954_7
  simp only [e_1954_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19125 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11773)) := by
  have c119256 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119257 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119258 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119259 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119260 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119261 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_8 := arithEq_of_rows h (row := 1954) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_8
  simp only [← c119256, ← c119257, ← c119258] at e_1954_8
  have e_1954_9 := arithEq_of_rows h (row := 1954) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_9
  simp only [← c119259, ← c119260, k2, ← c119261] at e_1954_9
  have hr := e_1954_9
  simp only [e_1954_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19126 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11774)) := by
  have c119262 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119263 := (publicBatchWrapper89_copies3726 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119264 := (publicBatchWrapper89_copies3727 a h).1
  have c119265 := (publicBatchWrapper89_copies3727 a h).2.1
  have c119266 := (publicBatchWrapper89_copies3727 a h).2.2.1
  have c119267 := (publicBatchWrapper89_copies3727 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_10 := arithEq_of_rows h (row := 1954) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_10
  simp only [← c119262, ← c119263, ← c119264] at e_1954_10
  have e_1954_11 := arithEq_of_rows h (row := 1954) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_11
  simp only [← c119265, ← c119266, k2, ← c119267] at e_1954_11
  have hr := e_1954_11
  simp only [e_1954_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19127 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11775)) := by
  have c119268 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.1
  have c119269 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.1
  have c119270 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.1
  have c119271 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.1
  have c119272 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.1
  have c119273 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_12 := arithEq_of_rows h (row := 1954) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_12
  simp only [← c119268, ← c119269, ← c119270] at e_1954_12
  have e_1954_13 := arithEq_of_rows h (row := 1954) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_13
  simp only [← c119271, ← c119272, k2, ← c119273] at e_1954_13
  have hr := e_1954_13
  simp only [e_1954_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19128 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11776)) := by
  have c119274 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119275 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119276 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119277 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119278 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119279 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_14 := arithEq_of_rows h (row := 1954) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_14
  simp only [← c119274, ← c119275, ← c119276] at e_1954_14
  have e_1954_15 := arithEq_of_rows h (row := 1954) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_15
  simp only [← c119277, ← c119278, k2, ← c119279] at e_1954_15
  have hr := e_1954_15
  simp only [e_1954_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19129 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11777)) := by
  have c119280 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119281 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119282 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119283 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119284 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119285 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_16 := arithEq_of_rows h (row := 1954) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_16
  simp only [← c119280, ← c119281, ← c119282] at e_1954_16
  have e_1954_17 := arithEq_of_rows h (row := 1954) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_17
  simp only [← c119283, ← c119284, k2, ← c119285] at e_1954_17
  have hr := e_1954_17
  simp only [e_1954_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19130 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1954 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11778)) := by
  have c119286 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119287 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119288 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119289 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119290 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119291 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1954_18 := arithEq_of_rows h (row := 1954) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_18
  simp only [← c119286, ← c119287, ← c119288] at e_1954_18
  have e_1954_19 := arithEq_of_rows h (row := 1954) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1954_19
  simp only [← c119289, ← c119290, k2, ← c119291] at e_1954_19
  have hr := e_1954_19
  simp only [e_1954_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19131 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11779)) := by
  have c119292 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119293 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119294 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119295 := (publicBatchWrapper89_copies3727 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119296 := (publicBatchWrapper89_copies3728 a h).1
  have c119297 := (publicBatchWrapper89_copies3728 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_0 := arithEq_of_rows h (row := 1955) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_0
  simp only [← c119292, ← c119293, ← c119294] at e_1955_0
  have e_1955_1 := arithEq_of_rows h (row := 1955) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_1
  simp only [← c119295, ← c119296, k2, ← c119297] at e_1955_1
  have hr := e_1955_1
  simp only [e_1955_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19132 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11780)) := by
  have c119298 := (publicBatchWrapper89_copies3728 a h).2.2.1
  have c119299 := (publicBatchWrapper89_copies3728 a h).2.2.2.1
  have c119300 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.1
  have c119301 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.1
  have c119302 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.1
  have c119303 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_2 := arithEq_of_rows h (row := 1955) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_2
  simp only [← c119298, ← c119299, ← c119300] at e_1955_2
  have e_1955_3 := arithEq_of_rows h (row := 1955) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_3
  simp only [← c119301, ← c119302, k2, ← c119303] at e_1955_3
  have hr := e_1955_3
  simp only [e_1955_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19133 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11781)) := by
  have c119304 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.1
  have c119305 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.1
  have c119306 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119307 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119308 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119309 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_4 := arithEq_of_rows h (row := 1955) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_4
  simp only [← c119304, ← c119305, ← c119306] at e_1955_4
  have e_1955_5 := arithEq_of_rows h (row := 1955) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_5
  simp only [← c119307, ← c119308, k2, ← c119309] at e_1955_5
  have hr := e_1955_5
  simp only [e_1955_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19134 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11782)) := by
  have c119310 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119311 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119312 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119313 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119314 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119315 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_6 := arithEq_of_rows h (row := 1955) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_6
  simp only [← c119310, ← c119311, ← c119312] at e_1955_6
  have e_1955_7 := arithEq_of_rows h (row := 1955) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_7
  simp only [← c119313, ← c119314, k2, ← c119315] at e_1955_7
  have hr := e_1955_7
  simp only [e_1955_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19135 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11783)) := by
  have c119316 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119317 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119318 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119319 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119320 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119321 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_8 := arithEq_of_rows h (row := 1955) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_8
  simp only [← c119316, ← c119317, ← c119318] at e_1955_8
  have e_1955_9 := arithEq_of_rows h (row := 1955) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_9
  simp only [← c119319, ← c119320, k2, ← c119321] at e_1955_9
  have hr := e_1955_9
  simp only [e_1955_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19136 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11784)) := by
  have c119322 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119323 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119324 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119325 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119326 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119327 := (publicBatchWrapper89_copies3728 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_10 := arithEq_of_rows h (row := 1955) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_10
  simp only [← c119322, ← c119323, ← c119324] at e_1955_10
  have e_1955_11 := arithEq_of_rows h (row := 1955) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_11
  simp only [← c119325, ← c119326, k2, ← c119327] at e_1955_11
  have hr := e_1955_11
  simp only [e_1955_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19137 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11785)) := by
  have c119328 := (publicBatchWrapper89_copies3729 a h).1
  have c119329 := (publicBatchWrapper89_copies3729 a h).2.1
  have c119330 := (publicBatchWrapper89_copies3729 a h).2.2.1
  have c119331 := (publicBatchWrapper89_copies3729 a h).2.2.2.1
  have c119332 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.1
  have c119333 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_12 := arithEq_of_rows h (row := 1955) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_12
  simp only [← c119328, ← c119329, ← c119330] at e_1955_12
  have e_1955_13 := arithEq_of_rows h (row := 1955) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_13
  simp only [← c119331, ← c119332, k2, ← c119333] at e_1955_13
  have hr := e_1955_13
  simp only [e_1955_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19138 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11786)) := by
  have c119334 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.1
  have c119335 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.1
  have c119336 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.1
  have c119337 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.1
  have c119338 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119339 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_14 := arithEq_of_rows h (row := 1955) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_14
  simp only [← c119334, ← c119335, ← c119336] at e_1955_14
  have e_1955_15 := arithEq_of_rows h (row := 1955) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_15
  simp only [← c119337, ← c119338, k2, ← c119339] at e_1955_15
  have hr := e_1955_15
  simp only [e_1955_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19139 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11787)) := by
  have c119340 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119341 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119342 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119343 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119344 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119345 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_16 := arithEq_of_rows h (row := 1955) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_16
  simp only [← c119340, ← c119341, ← c119342] at e_1955_16
  have e_1955_17 := arithEq_of_rows h (row := 1955) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_17
  simp only [← c119343, ← c119344, k2, ← c119345] at e_1955_17
  have hr := e_1955_17
  simp only [e_1955_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19140 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1955 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11788)) := by
  have c119346 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119347 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119348 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119349 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119350 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119351 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1955_18 := arithEq_of_rows h (row := 1955) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_18
  simp only [← c119346, ← c119347, ← c119348] at e_1955_18
  have e_1955_19 := arithEq_of_rows h (row := 1955) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1955_19
  simp only [← c119349, ← c119350, k2, ← c119351] at e_1955_19
  have hr := e_1955_19
  simp only [e_1955_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19141 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11789)) := by
  have c119352 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119353 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119354 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119355 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119356 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119357 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_0 := arithEq_of_rows h (row := 1956) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_0
  simp only [← c119352, ← c119353, ← c119354] at e_1956_0
  have e_1956_1 := arithEq_of_rows h (row := 1956) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_1
  simp only [← c119355, ← c119356, k2, ← c119357] at e_1956_1
  have hr := e_1956_1
  simp only [e_1956_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19142 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11790)) := by
  have c119358 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119359 := (publicBatchWrapper89_copies3729 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119360 := (publicBatchWrapper89_copies3730 a h).1
  have c119361 := (publicBatchWrapper89_copies3730 a h).2.1
  have c119362 := (publicBatchWrapper89_copies3730 a h).2.2.1
  have c119363 := (publicBatchWrapper89_copies3730 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_2 := arithEq_of_rows h (row := 1956) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_2
  simp only [← c119358, ← c119359, ← c119360] at e_1956_2
  have e_1956_3 := arithEq_of_rows h (row := 1956) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_3
  simp only [← c119361, ← c119362, k2, ← c119363] at e_1956_3
  have hr := e_1956_3
  simp only [e_1956_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19143 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11791)) := by
  have c119364 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.1
  have c119365 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.1
  have c119366 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.1
  have c119367 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.1
  have c119368 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.1
  have c119369 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_4 := arithEq_of_rows h (row := 1956) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_4
  simp only [← c119364, ← c119365, ← c119366] at e_1956_4
  have e_1956_5 := arithEq_of_rows h (row := 1956) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_5
  simp only [← c119367, ← c119368, k2, ← c119369] at e_1956_5
  have hr := e_1956_5
  simp only [e_1956_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19144 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 31) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11792)) := by
  have c119370 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119371 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119372 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119373 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119374 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119375 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_6 := arithEq_of_rows h (row := 1956) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_6
  simp only [← c119370, ← c119371, ← c119372] at e_1956_6
  have e_1956_7 := arithEq_of_rows h (row := 1956) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_7
  simp only [← c119373, ← c119374, k2, ← c119375] at e_1956_7
  have hr := e_1956_7
  simp only [e_1956_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19145 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 39) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11793)) := by
  have c119376 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119377 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119378 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119379 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119380 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119381 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_8 := arithEq_of_rows h (row := 1956) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_8
  simp only [← c119376, ← c119377, ← c119378] at e_1956_8
  have e_1956_9 := arithEq_of_rows h (row := 1956) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_9
  simp only [← c119379, ← c119380, k2, ← c119381] at e_1956_9
  have hr := e_1956_9
  simp only [e_1956_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19146 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 47) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11794)) := by
  have c119382 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119383 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119384 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119385 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119386 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119387 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_10 := arithEq_of_rows h (row := 1956) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_10
  simp only [← c119382, ← c119383, ← c119384] at e_1956_10
  have e_1956_11 := arithEq_of_rows h (row := 1956) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_11
  simp only [← c119385, ← c119386, k2, ← c119387] at e_1956_11
  have hr := e_1956_11
  simp only [e_1956_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19147 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 55) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11795)) := by
  have c119388 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119389 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119390 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119391 := (publicBatchWrapper89_copies3730 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119392 := (publicBatchWrapper89_copies3731 a h).1
  have c119393 := (publicBatchWrapper89_copies3731 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_12 := arithEq_of_rows h (row := 1956) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_12
  simp only [← c119388, ← c119389, ← c119390] at e_1956_12
  have e_1956_13 := arithEq_of_rows h (row := 1956) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_13
  simp only [← c119391, ← c119392, k2, ← c119393] at e_1956_13
  have hr := e_1956_13
  simp only [e_1956_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19148 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 63) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11796)) := by
  have c119394 := (publicBatchWrapper89_copies3731 a h).2.2.1
  have c119395 := (publicBatchWrapper89_copies3731 a h).2.2.2.1
  have c119396 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.1
  have c119397 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.1
  have c119398 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.1
  have c119399 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_14 := arithEq_of_rows h (row := 1956) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_14
  simp only [← c119394, ← c119395, ← c119396] at e_1956_14
  have e_1956_15 := arithEq_of_rows h (row := 1956) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_15
  simp only [← c119397, ← c119398, k2, ← c119399] at e_1956_15
  have hr := e_1956_15
  simp only [e_1956_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19149 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 71) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11797)) := by
  have c119400 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.1
  have c119401 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.1
  have c119402 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119403 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119404 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119405 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_16 := arithEq_of_rows h (row := 1956) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_16
  simp only [← c119400, ← c119401, ← c119402] at e_1956_16
  have e_1956_17 := arithEq_of_rows h (row := 1956) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_17
  simp only [← c119403, ← c119404, k2, ← c119405] at e_1956_17
  have hr := e_1956_17
  simp only [e_1956_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19150 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1956 79) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11798)) := by
  have c119406 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119407 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119408 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119409 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119410 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119411 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1956_18 := arithEq_of_rows h (row := 1956) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_18
  simp only [← c119406, ← c119407, ← c119408] at e_1956_18
  have e_1956_19 := arithEq_of_rows h (row := 1956) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1956_19
  simp only [← c119409, ← c119410, k2, ← c119411] at e_1956_19
  have hr := e_1956_19
  simp only [e_1956_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19151 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 7) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11799)) := by
  have c119412 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119413 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119414 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119415 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119416 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119417 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_0 := arithEq_of_rows h (row := 1957) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_0
  simp only [← c119412, ← c119413, ← c119414] at e_1957_0
  have e_1957_1 := arithEq_of_rows h (row := 1957) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_1
  simp only [← c119415, ← c119416, k2, ← c119417] at e_1957_1
  have hr := e_1957_1
  simp only [e_1957_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19152 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 15) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11800)) := by
  have c119418 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119419 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119420 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119421 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119422 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119423 := (publicBatchWrapper89_copies3731 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_2 := arithEq_of_rows h (row := 1957) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_2
  simp only [← c119418, ← c119419, ← c119420] at e_1957_2
  have e_1957_3 := arithEq_of_rows h (row := 1957) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_3
  simp only [← c119421, ← c119422, k2, ← c119423] at e_1957_3
  have hr := e_1957_3
  simp only [e_1957_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19153 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 23) = bselect (a (.wire 57 51)) (a (.virt 200)) (a (.virt 11801)) := by
  have c119424 := (publicBatchWrapper89_copies3732 a h).1
  have c119425 := (publicBatchWrapper89_copies3732 a h).2.1
  have c119426 := (publicBatchWrapper89_copies3732 a h).2.2.1
  have c119427 := (publicBatchWrapper89_copies3732 a h).2.2.2.1
  have c119428 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.1
  have c119429 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_4 := arithEq_of_rows h (row := 1957) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_4
  simp only [← c119424, ← c119425, ← c119426] at e_1957_4
  have e_1957_5 := arithEq_of_rows h (row := 1957) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_5
  simp only [← c119427, ← c119428, k2, ← c119429] at e_1957_5
  have hr := e_1957_5
  simp only [e_1957_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19154 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11970)) := by
  have c119430 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.1
  have c119431 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.1
  have c119432 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.1
  have c119433 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.1
  have c119434 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119435 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_6 := arithEq_of_rows h (row := 1957) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_6
  simp only [← c119430, ← c119431, ← c119432] at e_1957_6
  have e_1957_7 := arithEq_of_rows h (row := 1957) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_7
  simp only [← c119433, ← c119434, k2, ← c119435] at e_1957_7
  have hr := e_1957_7
  simp only [e_1957_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19155 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11971)) := by
  have c119436 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119437 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119438 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119439 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119440 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119441 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_8 := arithEq_of_rows h (row := 1957) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_8
  simp only [← c119436, ← c119437, ← c119438] at e_1957_8
  have e_1957_9 := arithEq_of_rows h (row := 1957) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_9
  simp only [← c119439, ← c119440, k2, ← c119441] at e_1957_9
  have hr := e_1957_9
  simp only [e_1957_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19156 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11972)) := by
  have c119442 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119443 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119444 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119445 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119446 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119447 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_10 := arithEq_of_rows h (row := 1957) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_10
  simp only [← c119442, ← c119443, ← c119444] at e_1957_10
  have e_1957_11 := arithEq_of_rows h (row := 1957) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_11
  simp only [← c119445, ← c119446, k2, ← c119447] at e_1957_11
  have hr := e_1957_11
  simp only [e_1957_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19157 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11973)) := by
  have c119448 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119449 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119450 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119451 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119452 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119453 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_12 := arithEq_of_rows h (row := 1957) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_12
  simp only [← c119448, ← c119449, ← c119450] at e_1957_12
  have e_1957_13 := arithEq_of_rows h (row := 1957) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_13
  simp only [← c119451, ← c119452, k2, ← c119453] at e_1957_13
  have hr := e_1957_13
  simp only [e_1957_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19158 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11974)) := by
  have c119454 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119455 := (publicBatchWrapper89_copies3732 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119456 := (publicBatchWrapper89_copies3733 a h).1
  have c119457 := (publicBatchWrapper89_copies3733 a h).2.1
  have c119458 := (publicBatchWrapper89_copies3733 a h).2.2.1
  have c119459 := (publicBatchWrapper89_copies3733 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_14 := arithEq_of_rows h (row := 1957) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_14
  simp only [← c119454, ← c119455, ← c119456] at e_1957_14
  have e_1957_15 := arithEq_of_rows h (row := 1957) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_15
  simp only [← c119457, ← c119458, k2, ← c119459] at e_1957_15
  have hr := e_1957_15
  simp only [e_1957_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19159 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11975)) := by
  have c119460 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.1
  have c119461 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.1
  have c119462 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.1
  have c119463 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.1
  have c119464 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.1
  have c119465 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_16 := arithEq_of_rows h (row := 1957) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_16
  simp only [← c119460, ← c119461, ← c119462] at e_1957_16
  have e_1957_17 := arithEq_of_rows h (row := 1957) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_17
  simp only [← c119463, ← c119464, k2, ← c119465] at e_1957_17
  have hr := e_1957_17
  simp only [e_1957_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19160 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1957 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11976)) := by
  have c119466 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119467 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119468 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119469 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119470 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119471 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1957_18 := arithEq_of_rows h (row := 1957) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_18
  simp only [← c119466, ← c119467, ← c119468] at e_1957_18
  have e_1957_19 := arithEq_of_rows h (row := 1957) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1957_19
  simp only [← c119469, ← c119470, k2, ← c119471] at e_1957_19
  have hr := e_1957_19
  simp only [e_1957_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19161 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11977)) := by
  have c119472 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119473 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119474 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119475 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119476 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119477 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_0 := arithEq_of_rows h (row := 1958) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_0
  simp only [← c119472, ← c119473, ← c119474] at e_1958_0
  have e_1958_1 := arithEq_of_rows h (row := 1958) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_1
  simp only [← c119475, ← c119476, k2, ← c119477] at e_1958_1
  have hr := e_1958_1
  simp only [e_1958_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19162 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11978)) := by
  have c119478 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119479 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119480 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119481 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119482 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119483 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_2 := arithEq_of_rows h (row := 1958) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_2
  simp only [← c119478, ← c119479, ← c119480] at e_1958_2
  have e_1958_3 := arithEq_of_rows h (row := 1958) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_3
  simp only [← c119481, ← c119482, k2, ← c119483] at e_1958_3
  have hr := e_1958_3
  simp only [e_1958_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19163 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11979)) := by
  have c119484 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119485 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119486 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119487 := (publicBatchWrapper89_copies3733 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119488 := (publicBatchWrapper89_copies3734 a h).1
  have c119489 := (publicBatchWrapper89_copies3734 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_4 := arithEq_of_rows h (row := 1958) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_4
  simp only [← c119484, ← c119485, ← c119486] at e_1958_4
  have e_1958_5 := arithEq_of_rows h (row := 1958) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_5
  simp only [← c119487, ← c119488, k2, ← c119489] at e_1958_5
  have hr := e_1958_5
  simp only [e_1958_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19164 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11980)) := by
  have c119490 := (publicBatchWrapper89_copies3734 a h).2.2.1
  have c119491 := (publicBatchWrapper89_copies3734 a h).2.2.2.1
  have c119492 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.1
  have c119493 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.1
  have c119494 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.1
  have c119495 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_6 := arithEq_of_rows h (row := 1958) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_6
  simp only [← c119490, ← c119491, ← c119492] at e_1958_6
  have e_1958_7 := arithEq_of_rows h (row := 1958) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_7
  simp only [← c119493, ← c119494, k2, ← c119495] at e_1958_7
  have hr := e_1958_7
  simp only [e_1958_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19165 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11981)) := by
  have c119496 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.1
  have c119497 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.1
  have c119498 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119499 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119500 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119501 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_8 := arithEq_of_rows h (row := 1958) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_8
  simp only [← c119496, ← c119497, ← c119498] at e_1958_8
  have e_1958_9 := arithEq_of_rows h (row := 1958) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_9
  simp only [← c119499, ← c119500, k2, ← c119501] at e_1958_9
  have hr := e_1958_9
  simp only [e_1958_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19166 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11982)) := by
  have c119502 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119503 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119504 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119505 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119506 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119507 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_10 := arithEq_of_rows h (row := 1958) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_10
  simp only [← c119502, ← c119503, ← c119504] at e_1958_10
  have e_1958_11 := arithEq_of_rows h (row := 1958) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_11
  simp only [← c119505, ← c119506, k2, ← c119507] at e_1958_11
  have hr := e_1958_11
  simp only [e_1958_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19167 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11983)) := by
  have c119508 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119509 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119510 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119511 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119512 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119513 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_12 := arithEq_of_rows h (row := 1958) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_12
  simp only [← c119508, ← c119509, ← c119510] at e_1958_12
  have e_1958_13 := arithEq_of_rows h (row := 1958) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_13
  simp only [← c119511, ← c119512, k2, ← c119513] at e_1958_13
  have hr := e_1958_13
  simp only [e_1958_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19168 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11984)) := by
  have c119514 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119515 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119516 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119517 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119518 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119519 := (publicBatchWrapper89_copies3734 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_14 := arithEq_of_rows h (row := 1958) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_14
  simp only [← c119514, ← c119515, ← c119516] at e_1958_14
  have e_1958_15 := arithEq_of_rows h (row := 1958) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_15
  simp only [← c119517, ← c119518, k2, ← c119519] at e_1958_15
  have hr := e_1958_15
  simp only [e_1958_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19169 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11985)) := by
  have c119520 := (publicBatchWrapper89_copies3735 a h).1
  have c119521 := (publicBatchWrapper89_copies3735 a h).2.1
  have c119522 := (publicBatchWrapper89_copies3735 a h).2.2.1
  have c119523 := (publicBatchWrapper89_copies3735 a h).2.2.2.1
  have c119524 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.1
  have c119525 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_16 := arithEq_of_rows h (row := 1958) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_16
  simp only [← c119520, ← c119521, ← c119522] at e_1958_16
  have e_1958_17 := arithEq_of_rows h (row := 1958) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_17
  simp only [← c119523, ← c119524, k2, ← c119525] at e_1958_17
  have hr := e_1958_17
  simp only [e_1958_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19170 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1958 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11986)) := by
  have c119526 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.1
  have c119527 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.1
  have c119528 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.1
  have c119529 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.1
  have c119530 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119531 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1958_18 := arithEq_of_rows h (row := 1958) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_18
  simp only [← c119526, ← c119527, ← c119528] at e_1958_18
  have e_1958_19 := arithEq_of_rows h (row := 1958) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1958_19
  simp only [← c119529, ← c119530, k2, ← c119531] at e_1958_19
  have hr := e_1958_19
  simp only [e_1958_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19171 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11987)) := by
  have c119532 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119533 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119534 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119535 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119536 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119537 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_0 := arithEq_of_rows h (row := 1959) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_0
  simp only [← c119532, ← c119533, ← c119534] at e_1959_0
  have e_1959_1 := arithEq_of_rows h (row := 1959) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_1
  simp only [← c119535, ← c119536, k2, ← c119537] at e_1959_1
  have hr := e_1959_1
  simp only [e_1959_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19172 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11988)) := by
  have c119538 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119539 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119540 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119541 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119542 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119543 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_2 := arithEq_of_rows h (row := 1959) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_2
  simp only [← c119538, ← c119539, ← c119540] at e_1959_2
  have e_1959_3 := arithEq_of_rows h (row := 1959) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_3
  simp only [← c119541, ← c119542, k2, ← c119543] at e_1959_3
  have hr := e_1959_3
  simp only [e_1959_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19173 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11989)) := by
  have c119544 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119545 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119546 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119547 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119548 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119549 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_4 := arithEq_of_rows h (row := 1959) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_4
  simp only [← c119544, ← c119545, ← c119546] at e_1959_4
  have e_1959_5 := arithEq_of_rows h (row := 1959) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_5
  simp only [← c119547, ← c119548, k2, ← c119549] at e_1959_5
  have hr := e_1959_5
  simp only [e_1959_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19174 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11990)) := by
  have c119550 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119551 := (publicBatchWrapper89_copies3735 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119552 := (publicBatchWrapper89_copies3736 a h).1
  have c119553 := (publicBatchWrapper89_copies3736 a h).2.1
  have c119554 := (publicBatchWrapper89_copies3736 a h).2.2.1
  have c119555 := (publicBatchWrapper89_copies3736 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_6 := arithEq_of_rows h (row := 1959) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_6
  simp only [← c119550, ← c119551, ← c119552] at e_1959_6
  have e_1959_7 := arithEq_of_rows h (row := 1959) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_7
  simp only [← c119553, ← c119554, k2, ← c119555] at e_1959_7
  have hr := e_1959_7
  simp only [e_1959_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19175 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11991)) := by
  have c119556 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.1
  have c119557 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.1
  have c119558 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.1
  have c119559 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.1
  have c119560 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.1
  have c119561 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_8 := arithEq_of_rows h (row := 1959) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_8
  simp only [← c119556, ← c119557, ← c119558] at e_1959_8
  have e_1959_9 := arithEq_of_rows h (row := 1959) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_9
  simp only [← c119559, ← c119560, k2, ← c119561] at e_1959_9
  have hr := e_1959_9
  simp only [e_1959_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19176 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 47) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11992)) := by
  have c119562 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119563 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119564 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119565 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119566 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119567 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_10 := arithEq_of_rows h (row := 1959) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_10
  simp only [← c119562, ← c119563, ← c119564] at e_1959_10
  have e_1959_11 := arithEq_of_rows h (row := 1959) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_11
  simp only [← c119565, ← c119566, k2, ← c119567] at e_1959_11
  have hr := e_1959_11
  simp only [e_1959_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19177 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 55) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11993)) := by
  have c119568 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119569 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119570 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119571 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119572 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119573 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_12 := arithEq_of_rows h (row := 1959) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_12
  simp only [← c119568, ← c119569, ← c119570] at e_1959_12
  have e_1959_13 := arithEq_of_rows h (row := 1959) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_13
  simp only [← c119571, ← c119572, k2, ← c119573] at e_1959_13
  have hr := e_1959_13
  simp only [e_1959_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19178 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 63) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11994)) := by
  have c119574 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119575 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119576 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119577 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119578 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119579 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_14 := arithEq_of_rows h (row := 1959) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_14
  simp only [← c119574, ← c119575, ← c119576] at e_1959_14
  have e_1959_15 := arithEq_of_rows h (row := 1959) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_15
  simp only [← c119577, ← c119578, k2, ← c119579] at e_1959_15
  have hr := e_1959_15
  simp only [e_1959_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19179 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 71) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11995)) := by
  have c119580 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119581 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119582 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119583 := (publicBatchWrapper89_copies3736 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119584 := (publicBatchWrapper89_copies3737 a h).1
  have c119585 := (publicBatchWrapper89_copies3737 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_16 := arithEq_of_rows h (row := 1959) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_16
  simp only [← c119580, ← c119581, ← c119582] at e_1959_16
  have e_1959_17 := arithEq_of_rows h (row := 1959) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_17
  simp only [← c119583, ← c119584, k2, ← c119585] at e_1959_17
  have hr := e_1959_17
  simp only [e_1959_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19180 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1959 79) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11996)) := by
  have c119586 := (publicBatchWrapper89_copies3737 a h).2.2.1
  have c119587 := (publicBatchWrapper89_copies3737 a h).2.2.2.1
  have c119588 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.1
  have c119589 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.1
  have c119590 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.1
  have c119591 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1959_18 := arithEq_of_rows h (row := 1959) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_18
  simp only [← c119586, ← c119587, ← c119588] at e_1959_18
  have e_1959_19 := arithEq_of_rows h (row := 1959) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1959_19
  simp only [← c119589, ← c119590, k2, ← c119591] at e_1959_19
  have hr := e_1959_19
  simp only [e_1959_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19181 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 7) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11997)) := by
  have c119592 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.1
  have c119593 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.1
  have c119594 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119595 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119596 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119597 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_0 := arithEq_of_rows h (row := 1960) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_0
  simp only [← c119592, ← c119593, ← c119594] at e_1960_0
  have e_1960_1 := arithEq_of_rows h (row := 1960) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_1
  simp only [← c119595, ← c119596, k2, ← c119597] at e_1960_1
  have hr := e_1960_1
  simp only [e_1960_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19182 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 15) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11998)) := by
  have c119598 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119599 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119600 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119601 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119602 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119603 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_2 := arithEq_of_rows h (row := 1960) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_2
  simp only [← c119598, ← c119599, ← c119600] at e_1960_2
  have e_1960_3 := arithEq_of_rows h (row := 1960) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_3
  simp only [← c119601, ← c119602, k2, ← c119603] at e_1960_3
  have hr := e_1960_3
  simp only [e_1960_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19183 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 23) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 11999)) := by
  have c119604 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119605 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119606 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119607 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119608 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119609 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_4 := arithEq_of_rows h (row := 1960) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_4
  simp only [← c119604, ← c119605, ← c119606] at e_1960_4
  have e_1960_5 := arithEq_of_rows h (row := 1960) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_5
  simp only [← c119607, ← c119608, k2, ← c119609] at e_1960_5
  have hr := e_1960_5
  simp only [e_1960_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19184 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 31) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 12000)) := by
  have c119610 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119611 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119612 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119613 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119614 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119615 := (publicBatchWrapper89_copies3737 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_6 := arithEq_of_rows h (row := 1960) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_6
  simp only [← c119610, ← c119611, ← c119612] at e_1960_6
  have e_1960_7 := arithEq_of_rows h (row := 1960) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_7
  simp only [← c119613, ← c119614, k2, ← c119615] at e_1960_7
  have hr := e_1960_7
  simp only [e_1960_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19185 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 39) = bselect (a (.wire 58 15)) (a (.virt 200)) (a (.virt 12001)) := by
  have c119616 := (publicBatchWrapper89_copies3738 a h).1
  have c119617 := (publicBatchWrapper89_copies3738 a h).2.1
  have c119618 := (publicBatchWrapper89_copies3738 a h).2.2.1
  have c119619 := (publicBatchWrapper89_copies3738 a h).2.2.2.1
  have c119620 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.1
  have c119621 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_8 := arithEq_of_rows h (row := 1960) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_8
  simp only [← c119616, ← c119617, ← c119618] at e_1960_8
  have e_1960_9 := arithEq_of_rows h (row := 1960) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_9
  simp only [← c119619, ← c119620, k2, ← c119621] at e_1960_9
  have hr := e_1960_9
  simp only [e_1960_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19186 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12170)) := by
  have c119622 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.1
  have c119623 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.1
  have c119624 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.1
  have c119625 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.1
  have c119626 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119627 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_10 := arithEq_of_rows h (row := 1960) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_10
  simp only [← c119622, ← c119623, ← c119624] at e_1960_10
  have e_1960_11 := arithEq_of_rows h (row := 1960) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_11
  simp only [← c119625, ← c119626, k2, ← c119627] at e_1960_11
  have hr := e_1960_11
  simp only [e_1960_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19187 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12171)) := by
  have c119628 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119629 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119630 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119631 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119632 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119633 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_12 := arithEq_of_rows h (row := 1960) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_12
  simp only [← c119628, ← c119629, ← c119630] at e_1960_12
  have e_1960_13 := arithEq_of_rows h (row := 1960) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_13
  simp only [← c119631, ← c119632, k2, ← c119633] at e_1960_13
  have hr := e_1960_13
  simp only [e_1960_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19188 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12172)) := by
  have c119634 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119635 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119636 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119637 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119638 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119639 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_14 := arithEq_of_rows h (row := 1960) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_14
  simp only [← c119634, ← c119635, ← c119636] at e_1960_14
  have e_1960_15 := arithEq_of_rows h (row := 1960) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_15
  simp only [← c119637, ← c119638, k2, ← c119639] at e_1960_15
  have hr := e_1960_15
  simp only [e_1960_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19189 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12173)) := by
  have c119640 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119641 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119642 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119643 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119644 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119645 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_16 := arithEq_of_rows h (row := 1960) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_16
  simp only [← c119640, ← c119641, ← c119642] at e_1960_16
  have e_1960_17 := arithEq_of_rows h (row := 1960) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_17
  simp only [← c119643, ← c119644, k2, ← c119645] at e_1960_17
  have hr := e_1960_17
  simp only [e_1960_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19190 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1960 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12174)) := by
  have c119646 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119647 := (publicBatchWrapper89_copies3738 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119648 := (publicBatchWrapper89_copies3739 a h).1
  have c119649 := (publicBatchWrapper89_copies3739 a h).2.1
  have c119650 := (publicBatchWrapper89_copies3739 a h).2.2.1
  have c119651 := (publicBatchWrapper89_copies3739 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1960_18 := arithEq_of_rows h (row := 1960) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_18
  simp only [← c119646, ← c119647, ← c119648] at e_1960_18
  have e_1960_19 := arithEq_of_rows h (row := 1960) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1960_19
  simp only [← c119649, ← c119650, k2, ← c119651] at e_1960_19
  have hr := e_1960_19
  simp only [e_1960_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19191 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12175)) := by
  have c119652 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.1
  have c119653 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.1
  have c119654 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.1
  have c119655 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.1
  have c119656 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.1
  have c119657 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_0 := arithEq_of_rows h (row := 1961) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_0
  simp only [← c119652, ← c119653, ← c119654] at e_1961_0
  have e_1961_1 := arithEq_of_rows h (row := 1961) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_1
  simp only [← c119655, ← c119656, k2, ← c119657] at e_1961_1
  have hr := e_1961_1
  simp only [e_1961_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19192 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12176)) := by
  have c119658 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119659 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119660 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119661 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119662 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119663 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_2 := arithEq_of_rows h (row := 1961) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_2
  simp only [← c119658, ← c119659, ← c119660] at e_1961_2
  have e_1961_3 := arithEq_of_rows h (row := 1961) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_3
  simp only [← c119661, ← c119662, k2, ← c119663] at e_1961_3
  have hr := e_1961_3
  simp only [e_1961_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19193 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12177)) := by
  have c119664 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119665 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119666 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119667 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119668 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119669 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_4 := arithEq_of_rows h (row := 1961) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_4
  simp only [← c119664, ← c119665, ← c119666] at e_1961_4
  have e_1961_5 := arithEq_of_rows h (row := 1961) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_5
  simp only [← c119667, ← c119668, k2, ← c119669] at e_1961_5
  have hr := e_1961_5
  simp only [e_1961_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19194 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12178)) := by
  have c119670 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119671 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119672 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119673 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119674 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119675 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_6 := arithEq_of_rows h (row := 1961) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_6
  simp only [← c119670, ← c119671, ← c119672] at e_1961_6
  have e_1961_7 := arithEq_of_rows h (row := 1961) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_7
  simp only [← c119673, ← c119674, k2, ← c119675] at e_1961_7
  have hr := e_1961_7
  simp only [e_1961_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19195 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12179)) := by
  have c119676 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119677 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119678 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119679 := (publicBatchWrapper89_copies3739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119680 := (publicBatchWrapper89_copies3740 a h).1
  have c119681 := (publicBatchWrapper89_copies3740 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_8 := arithEq_of_rows h (row := 1961) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_8
  simp only [← c119676, ← c119677, ← c119678] at e_1961_8
  have e_1961_9 := arithEq_of_rows h (row := 1961) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_9
  simp only [← c119679, ← c119680, k2, ← c119681] at e_1961_9
  have hr := e_1961_9
  simp only [e_1961_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19196 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12180)) := by
  have c119682 := (publicBatchWrapper89_copies3740 a h).2.2.1
  have c119683 := (publicBatchWrapper89_copies3740 a h).2.2.2.1
  have c119684 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.1
  have c119685 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.1
  have c119686 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.1
  have c119687 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_10 := arithEq_of_rows h (row := 1961) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_10
  simp only [← c119682, ← c119683, ← c119684] at e_1961_10
  have e_1961_11 := arithEq_of_rows h (row := 1961) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_11
  simp only [← c119685, ← c119686, k2, ← c119687] at e_1961_11
  have hr := e_1961_11
  simp only [e_1961_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19197 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12181)) := by
  have c119688 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.1
  have c119689 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.1
  have c119690 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119691 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119692 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119693 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_12 := arithEq_of_rows h (row := 1961) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_12
  simp only [← c119688, ← c119689, ← c119690] at e_1961_12
  have e_1961_13 := arithEq_of_rows h (row := 1961) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_13
  simp only [← c119691, ← c119692, k2, ← c119693] at e_1961_13
  have hr := e_1961_13
  simp only [e_1961_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19198 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12182)) := by
  have c119694 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119695 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119696 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119697 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119698 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119699 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_14 := arithEq_of_rows h (row := 1961) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_14
  simp only [← c119694, ← c119695, ← c119696] at e_1961_14
  have e_1961_15 := arithEq_of_rows h (row := 1961) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_15
  simp only [← c119697, ← c119698, k2, ← c119699] at e_1961_15
  have hr := e_1961_15
  simp only [e_1961_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19199 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12183)) := by
  have c119700 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119701 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119702 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119703 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119704 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119705 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_16 := arithEq_of_rows h (row := 1961) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_16
  simp only [← c119700, ← c119701, ← c119702] at e_1961_16
  have e_1961_17 := arithEq_of_rows h (row := 1961) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_17
  simp only [← c119703, ← c119704, k2, ← c119705] at e_1961_17
  have hr := e_1961_17
  simp only [e_1961_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19200 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1961 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12184)) := by
  have c119706 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119707 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119708 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119709 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119710 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119711 := (publicBatchWrapper89_copies3740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1961_18 := arithEq_of_rows h (row := 1961) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_18
  simp only [← c119706, ← c119707, ← c119708] at e_1961_18
  have e_1961_19 := arithEq_of_rows h (row := 1961) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1961_19
  simp only [← c119709, ← c119710, k2, ← c119711] at e_1961_19
  have hr := e_1961_19
  simp only [e_1961_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19201 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12185)) := by
  have c119712 := (publicBatchWrapper89_copies3741 a h).1
  have c119713 := (publicBatchWrapper89_copies3741 a h).2.1
  have c119714 := (publicBatchWrapper89_copies3741 a h).2.2.1
  have c119715 := (publicBatchWrapper89_copies3741 a h).2.2.2.1
  have c119716 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.1
  have c119717 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_0 := arithEq_of_rows h (row := 1962) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_0
  simp only [← c119712, ← c119713, ← c119714] at e_1962_0
  have e_1962_1 := arithEq_of_rows h (row := 1962) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_1
  simp only [← c119715, ← c119716, k2, ← c119717] at e_1962_1
  have hr := e_1962_1
  simp only [e_1962_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19202 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12186)) := by
  have c119718 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.1
  have c119719 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.1
  have c119720 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.1
  have c119721 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.1
  have c119722 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119723 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_2 := arithEq_of_rows h (row := 1962) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_2
  simp only [← c119718, ← c119719, ← c119720] at e_1962_2
  have e_1962_3 := arithEq_of_rows h (row := 1962) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_3
  simp only [← c119721, ← c119722, k2, ← c119723] at e_1962_3
  have hr := e_1962_3
  simp only [e_1962_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19203 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12187)) := by
  have c119724 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119725 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119726 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119727 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119728 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119729 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_4 := arithEq_of_rows h (row := 1962) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_4
  simp only [← c119724, ← c119725, ← c119726] at e_1962_4
  have e_1962_5 := arithEq_of_rows h (row := 1962) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_5
  simp only [← c119727, ← c119728, k2, ← c119729] at e_1962_5
  have hr := e_1962_5
  simp only [e_1962_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19204 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12188)) := by
  have c119730 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119731 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119732 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119733 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119734 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119735 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_6 := arithEq_of_rows h (row := 1962) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_6
  simp only [← c119730, ← c119731, ← c119732] at e_1962_6
  have e_1962_7 := arithEq_of_rows h (row := 1962) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_7
  simp only [← c119733, ← c119734, k2, ← c119735] at e_1962_7
  have hr := e_1962_7
  simp only [e_1962_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19205 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12189)) := by
  have c119736 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119737 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119738 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119739 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119740 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119741 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_8 := arithEq_of_rows h (row := 1962) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_8
  simp only [← c119736, ← c119737, ← c119738] at e_1962_8
  have e_1962_9 := arithEq_of_rows h (row := 1962) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_9
  simp only [← c119739, ← c119740, k2, ← c119741] at e_1962_9
  have hr := e_1962_9
  simp only [e_1962_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19206 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12190)) := by
  have c119742 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119743 := (publicBatchWrapper89_copies3741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119744 := (publicBatchWrapper89_copies3742 a h).1
  have c119745 := (publicBatchWrapper89_copies3742 a h).2.1
  have c119746 := (publicBatchWrapper89_copies3742 a h).2.2.1
  have c119747 := (publicBatchWrapper89_copies3742 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_10 := arithEq_of_rows h (row := 1962) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_10
  simp only [← c119742, ← c119743, ← c119744] at e_1962_10
  have e_1962_11 := arithEq_of_rows h (row := 1962) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_11
  simp only [← c119745, ← c119746, k2, ← c119747] at e_1962_11
  have hr := e_1962_11
  simp only [e_1962_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19207 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12191)) := by
  have c119748 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.1
  have c119749 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.1
  have c119750 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.1
  have c119751 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.1
  have c119752 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.1
  have c119753 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_12 := arithEq_of_rows h (row := 1962) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_12
  simp only [← c119748, ← c119749, ← c119750] at e_1962_12
  have e_1962_13 := arithEq_of_rows h (row := 1962) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_13
  simp only [← c119751, ← c119752, k2, ← c119753] at e_1962_13
  have hr := e_1962_13
  simp only [e_1962_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19208 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 63) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12192)) := by
  have c119754 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119755 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119756 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119757 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119758 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119759 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_14 := arithEq_of_rows h (row := 1962) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_14
  simp only [← c119754, ← c119755, ← c119756] at e_1962_14
  have e_1962_15 := arithEq_of_rows h (row := 1962) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_15
  simp only [← c119757, ← c119758, k2, ← c119759] at e_1962_15
  have hr := e_1962_15
  simp only [e_1962_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19209 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 71) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12193)) := by
  have c119760 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119761 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119762 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119763 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119764 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119765 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_16 := arithEq_of_rows h (row := 1962) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_16
  simp only [← c119760, ← c119761, ← c119762] at e_1962_16
  have e_1962_17 := arithEq_of_rows h (row := 1962) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_17
  simp only [← c119763, ← c119764, k2, ← c119765] at e_1962_17
  have hr := e_1962_17
  simp only [e_1962_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19210 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1962 79) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12194)) := by
  have c119766 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119767 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119768 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119769 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119770 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119771 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1962_18 := arithEq_of_rows h (row := 1962) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_18
  simp only [← c119766, ← c119767, ← c119768] at e_1962_18
  have e_1962_19 := arithEq_of_rows h (row := 1962) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1962_19
  simp only [← c119769, ← c119770, k2, ← c119771] at e_1962_19
  have hr := e_1962_19
  simp only [e_1962_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19211 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 7) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12195)) := by
  have c119772 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119773 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119774 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119775 := (publicBatchWrapper89_copies3742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119776 := (publicBatchWrapper89_copies3743 a h).1
  have c119777 := (publicBatchWrapper89_copies3743 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_0 := arithEq_of_rows h (row := 1963) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_0
  simp only [← c119772, ← c119773, ← c119774] at e_1963_0
  have e_1963_1 := arithEq_of_rows h (row := 1963) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_1
  simp only [← c119775, ← c119776, k2, ← c119777] at e_1963_1
  have hr := e_1963_1
  simp only [e_1963_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19212 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 15) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12196)) := by
  have c119778 := (publicBatchWrapper89_copies3743 a h).2.2.1
  have c119779 := (publicBatchWrapper89_copies3743 a h).2.2.2.1
  have c119780 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.1
  have c119781 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.1
  have c119782 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.1
  have c119783 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_2 := arithEq_of_rows h (row := 1963) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_2
  simp only [← c119778, ← c119779, ← c119780] at e_1963_2
  have e_1963_3 := arithEq_of_rows h (row := 1963) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_3
  simp only [← c119781, ← c119782, k2, ← c119783] at e_1963_3
  have hr := e_1963_3
  simp only [e_1963_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19213 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 23) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12197)) := by
  have c119784 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.1
  have c119785 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.1
  have c119786 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119787 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119788 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119789 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_4 := arithEq_of_rows h (row := 1963) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_4
  simp only [← c119784, ← c119785, ← c119786] at e_1963_4
  have e_1963_5 := arithEq_of_rows h (row := 1963) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_5
  simp only [← c119787, ← c119788, k2, ← c119789] at e_1963_5
  have hr := e_1963_5
  simp only [e_1963_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19214 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 31) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12198)) := by
  have c119790 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119791 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119792 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119793 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119794 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119795 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_6 := arithEq_of_rows h (row := 1963) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_6
  simp only [← c119790, ← c119791, ← c119792] at e_1963_6
  have e_1963_7 := arithEq_of_rows h (row := 1963) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_7
  simp only [← c119793, ← c119794, k2, ← c119795] at e_1963_7
  have hr := e_1963_7
  simp only [e_1963_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19215 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 39) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12199)) := by
  have c119796 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119797 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119798 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119799 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119800 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119801 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_8 := arithEq_of_rows h (row := 1963) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_8
  simp only [← c119796, ← c119797, ← c119798] at e_1963_8
  have e_1963_9 := arithEq_of_rows h (row := 1963) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_9
  simp only [← c119799, ← c119800, k2, ← c119801] at e_1963_9
  have hr := e_1963_9
  simp only [e_1963_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19216 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 47) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12200)) := by
  have c119802 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119803 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119804 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119805 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119806 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119807 := (publicBatchWrapper89_copies3743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_10 := arithEq_of_rows h (row := 1963) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_10
  simp only [← c119802, ← c119803, ← c119804] at e_1963_10
  have e_1963_11 := arithEq_of_rows h (row := 1963) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_11
  simp only [← c119805, ← c119806, k2, ← c119807] at e_1963_11
  have hr := e_1963_11
  simp only [e_1963_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19217 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 55) = bselect (a (.wire 58 59)) (a (.virt 200)) (a (.virt 12201)) := by
  have c119808 := (publicBatchWrapper89_copies3744 a h).1
  have c119809 := (publicBatchWrapper89_copies3744 a h).2.1
  have c119810 := (publicBatchWrapper89_copies3744 a h).2.2.1
  have c119811 := (publicBatchWrapper89_copies3744 a h).2.2.2.1
  have c119812 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.1
  have c119813 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_12 := arithEq_of_rows h (row := 1963) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_12
  simp only [← c119808, ← c119809, ← c119810] at e_1963_12
  have e_1963_13 := arithEq_of_rows h (row := 1963) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_13
  simp only [← c119811, ← c119812, k2, ← c119813] at e_1963_13
  have hr := e_1963_13
  simp only [e_1963_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19218 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12370)) := by
  have c119814 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.1
  have c119815 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.1
  have c119816 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.1
  have c119817 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.1
  have c119818 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119819 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_14 := arithEq_of_rows h (row := 1963) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_14
  simp only [← c119814, ← c119815, ← c119816] at e_1963_14
  have e_1963_15 := arithEq_of_rows h (row := 1963) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_15
  simp only [← c119817, ← c119818, k2, ← c119819] at e_1963_15
  have hr := e_1963_15
  simp only [e_1963_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19219 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12371)) := by
  have c119820 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119821 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119822 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119823 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119824 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119825 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_16 := arithEq_of_rows h (row := 1963) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_16
  simp only [← c119820, ← c119821, ← c119822] at e_1963_16
  have e_1963_17 := arithEq_of_rows h (row := 1963) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_17
  simp only [← c119823, ← c119824, k2, ← c119825] at e_1963_17
  have hr := e_1963_17
  simp only [e_1963_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19220 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1963 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12372)) := by
  have c119826 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119827 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119828 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119829 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119830 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119831 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1963_18 := arithEq_of_rows h (row := 1963) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_18
  simp only [← c119826, ← c119827, ← c119828] at e_1963_18
  have e_1963_19 := arithEq_of_rows h (row := 1963) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1963_19
  simp only [← c119829, ← c119830, k2, ← c119831] at e_1963_19
  have hr := e_1963_19
  simp only [e_1963_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19221 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12373)) := by
  have c119832 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119833 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119834 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119835 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119836 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119837 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_0 := arithEq_of_rows h (row := 1964) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_0
  simp only [← c119832, ← c119833, ← c119834] at e_1964_0
  have e_1964_1 := arithEq_of_rows h (row := 1964) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_1
  simp only [← c119835, ← c119836, k2, ← c119837] at e_1964_1
  have hr := e_1964_1
  simp only [e_1964_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19222 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12374)) := by
  have c119838 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119839 := (publicBatchWrapper89_copies3744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119840 := (publicBatchWrapper89_copies3745 a h).1
  have c119841 := (publicBatchWrapper89_copies3745 a h).2.1
  have c119842 := (publicBatchWrapper89_copies3745 a h).2.2.1
  have c119843 := (publicBatchWrapper89_copies3745 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_2 := arithEq_of_rows h (row := 1964) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_2
  simp only [← c119838, ← c119839, ← c119840] at e_1964_2
  have e_1964_3 := arithEq_of_rows h (row := 1964) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_3
  simp only [← c119841, ← c119842, k2, ← c119843] at e_1964_3
  have hr := e_1964_3
  simp only [e_1964_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19223 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12375)) := by
  have c119844 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.1
  have c119845 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.1
  have c119846 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.1
  have c119847 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.1
  have c119848 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.1
  have c119849 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_4 := arithEq_of_rows h (row := 1964) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_4
  simp only [← c119844, ← c119845, ← c119846] at e_1964_4
  have e_1964_5 := arithEq_of_rows h (row := 1964) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_5
  simp only [← c119847, ← c119848, k2, ← c119849] at e_1964_5
  have hr := e_1964_5
  simp only [e_1964_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19224 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12376)) := by
  have c119850 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119851 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119852 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119853 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119854 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119855 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_6 := arithEq_of_rows h (row := 1964) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_6
  simp only [← c119850, ← c119851, ← c119852] at e_1964_6
  have e_1964_7 := arithEq_of_rows h (row := 1964) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_7
  simp only [← c119853, ← c119854, k2, ← c119855] at e_1964_7
  have hr := e_1964_7
  simp only [e_1964_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19225 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12377)) := by
  have c119856 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119857 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119858 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119859 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119860 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119861 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_8 := arithEq_of_rows h (row := 1964) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_8
  simp only [← c119856, ← c119857, ← c119858] at e_1964_8
  have e_1964_9 := arithEq_of_rows h (row := 1964) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_9
  simp only [← c119859, ← c119860, k2, ← c119861] at e_1964_9
  have hr := e_1964_9
  simp only [e_1964_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19226 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12378)) := by
  have c119862 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119863 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119864 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119865 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119866 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119867 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_10 := arithEq_of_rows h (row := 1964) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_10
  simp only [← c119862, ← c119863, ← c119864] at e_1964_10
  have e_1964_11 := arithEq_of_rows h (row := 1964) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_11
  simp only [← c119865, ← c119866, k2, ← c119867] at e_1964_11
  have hr := e_1964_11
  simp only [e_1964_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19227 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12379)) := by
  have c119868 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119869 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119870 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119871 := (publicBatchWrapper89_copies3745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119872 := (publicBatchWrapper89_copies3746 a h).1
  have c119873 := (publicBatchWrapper89_copies3746 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_12 := arithEq_of_rows h (row := 1964) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_12
  simp only [← c119868, ← c119869, ← c119870] at e_1964_12
  have e_1964_13 := arithEq_of_rows h (row := 1964) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_13
  simp only [← c119871, ← c119872, k2, ← c119873] at e_1964_13
  have hr := e_1964_13
  simp only [e_1964_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19228 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12380)) := by
  have c119874 := (publicBatchWrapper89_copies3746 a h).2.2.1
  have c119875 := (publicBatchWrapper89_copies3746 a h).2.2.2.1
  have c119876 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.1
  have c119877 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.1
  have c119878 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.1
  have c119879 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_14 := arithEq_of_rows h (row := 1964) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_14
  simp only [← c119874, ← c119875, ← c119876] at e_1964_14
  have e_1964_15 := arithEq_of_rows h (row := 1964) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_15
  simp only [← c119877, ← c119878, k2, ← c119879] at e_1964_15
  have hr := e_1964_15
  simp only [e_1964_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19229 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12381)) := by
  have c119880 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.1
  have c119881 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.1
  have c119882 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119883 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119884 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119885 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_16 := arithEq_of_rows h (row := 1964) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_16
  simp only [← c119880, ← c119881, ← c119882] at e_1964_16
  have e_1964_17 := arithEq_of_rows h (row := 1964) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_17
  simp only [← c119883, ← c119884, k2, ← c119885] at e_1964_17
  have hr := e_1964_17
  simp only [e_1964_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19230 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1964 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12382)) := by
  have c119886 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119887 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119888 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119889 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119890 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119891 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1964_18 := arithEq_of_rows h (row := 1964) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_18
  simp only [← c119886, ← c119887, ← c119888] at e_1964_18
  have e_1964_19 := arithEq_of_rows h (row := 1964) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1964_19
  simp only [← c119889, ← c119890, k2, ← c119891] at e_1964_19
  have hr := e_1964_19
  simp only [e_1964_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19231 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12383)) := by
  have c119892 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119893 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119894 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119895 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119896 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119897 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_0 := arithEq_of_rows h (row := 1965) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_0
  simp only [← c119892, ← c119893, ← c119894] at e_1965_0
  have e_1965_1 := arithEq_of_rows h (row := 1965) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_1
  simp only [← c119895, ← c119896, k2, ← c119897] at e_1965_1
  have hr := e_1965_1
  simp only [e_1965_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19232 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12384)) := by
  have c119898 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119899 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119900 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119901 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119902 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119903 := (publicBatchWrapper89_copies3746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_2 := arithEq_of_rows h (row := 1965) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_2
  simp only [← c119898, ← c119899, ← c119900] at e_1965_2
  have e_1965_3 := arithEq_of_rows h (row := 1965) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_3
  simp only [← c119901, ← c119902, k2, ← c119903] at e_1965_3
  have hr := e_1965_3
  simp only [e_1965_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19233 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12385)) := by
  have c119904 := (publicBatchWrapper89_copies3747 a h).1
  have c119905 := (publicBatchWrapper89_copies3747 a h).2.1
  have c119906 := (publicBatchWrapper89_copies3747 a h).2.2.1
  have c119907 := (publicBatchWrapper89_copies3747 a h).2.2.2.1
  have c119908 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.1
  have c119909 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_4 := arithEq_of_rows h (row := 1965) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_4
  simp only [← c119904, ← c119905, ← c119906] at e_1965_4
  have e_1965_5 := arithEq_of_rows h (row := 1965) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_5
  simp only [← c119907, ← c119908, k2, ← c119909] at e_1965_5
  have hr := e_1965_5
  simp only [e_1965_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19234 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12386)) := by
  have c119910 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.1
  have c119911 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.1
  have c119912 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.1
  have c119913 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.1
  have c119914 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119915 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_6 := arithEq_of_rows h (row := 1965) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_6
  simp only [← c119910, ← c119911, ← c119912] at e_1965_6
  have e_1965_7 := arithEq_of_rows h (row := 1965) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_7
  simp only [← c119913, ← c119914, k2, ← c119915] at e_1965_7
  have hr := e_1965_7
  simp only [e_1965_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19235 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12387)) := by
  have c119916 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119917 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119918 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119919 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119920 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119921 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_8 := arithEq_of_rows h (row := 1965) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_8
  simp only [← c119916, ← c119917, ← c119918] at e_1965_8
  have e_1965_9 := arithEq_of_rows h (row := 1965) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_9
  simp only [← c119919, ← c119920, k2, ← c119921] at e_1965_9
  have hr := e_1965_9
  simp only [e_1965_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19236 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12388)) := by
  have c119922 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119923 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119924 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119925 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119926 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119927 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_10 := arithEq_of_rows h (row := 1965) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_10
  simp only [← c119922, ← c119923, ← c119924] at e_1965_10
  have e_1965_11 := arithEq_of_rows h (row := 1965) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_11
  simp only [← c119925, ← c119926, k2, ← c119927] at e_1965_11
  have hr := e_1965_11
  simp only [e_1965_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19237 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12389)) := by
  have c119928 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119929 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119930 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119931 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119932 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119933 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_12 := arithEq_of_rows h (row := 1965) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_12
  simp only [← c119928, ← c119929, ← c119930] at e_1965_12
  have e_1965_13 := arithEq_of_rows h (row := 1965) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_13
  simp only [← c119931, ← c119932, k2, ← c119933] at e_1965_13
  have hr := e_1965_13
  simp only [e_1965_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19238 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12390)) := by
  have c119934 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119935 := (publicBatchWrapper89_copies3747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119936 := (publicBatchWrapper89_copies3748 a h).1
  have c119937 := (publicBatchWrapper89_copies3748 a h).2.1
  have c119938 := (publicBatchWrapper89_copies3748 a h).2.2.1
  have c119939 := (publicBatchWrapper89_copies3748 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_14 := arithEq_of_rows h (row := 1965) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_14
  simp only [← c119934, ← c119935, ← c119936] at e_1965_14
  have e_1965_15 := arithEq_of_rows h (row := 1965) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_15
  simp only [← c119937, ← c119938, k2, ← c119939] at e_1965_15
  have hr := e_1965_15
  simp only [e_1965_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19239 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12391)) := by
  have c119940 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.1
  have c119941 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.1
  have c119942 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.1
  have c119943 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.1
  have c119944 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.1
  have c119945 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_16 := arithEq_of_rows h (row := 1965) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_16
  simp only [← c119940, ← c119941, ← c119942] at e_1965_16
  have e_1965_17 := arithEq_of_rows h (row := 1965) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_17
  simp only [← c119943, ← c119944, k2, ← c119945] at e_1965_17
  have hr := e_1965_17
  simp only [e_1965_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19240 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1965 79) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12392)) := by
  have c119946 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119947 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119948 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119949 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119950 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119951 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1965_18 := arithEq_of_rows h (row := 1965) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_18
  simp only [← c119946, ← c119947, ← c119948] at e_1965_18
  have e_1965_19 := arithEq_of_rows h (row := 1965) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1965_19
  simp only [← c119949, ← c119950, k2, ← c119951] at e_1965_19
  have hr := e_1965_19
  simp only [e_1965_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19241 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 7) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12393)) := by
  have c119952 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119953 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119954 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119955 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119956 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119957 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_0 := arithEq_of_rows h (row := 1966) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_0
  simp only [← c119952, ← c119953, ← c119954] at e_1966_0
  have e_1966_1 := arithEq_of_rows h (row := 1966) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_1
  simp only [← c119955, ← c119956, k2, ← c119957] at e_1966_1
  have hr := e_1966_1
  simp only [e_1966_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19242 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 15) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12394)) := by
  have c119958 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119959 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119960 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119961 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119962 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119963 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_2 := arithEq_of_rows h (row := 1966) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_2
  simp only [← c119958, ← c119959, ← c119960] at e_1966_2
  have e_1966_3 := arithEq_of_rows h (row := 1966) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_3
  simp only [← c119961, ← c119962, k2, ← c119963] at e_1966_3
  have hr := e_1966_3
  simp only [e_1966_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19243 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 23) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12395)) := by
  have c119964 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119965 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119966 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119967 := (publicBatchWrapper89_copies3748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c119968 := (publicBatchWrapper89_copies3749 a h).1
  have c119969 := (publicBatchWrapper89_copies3749 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_4 := arithEq_of_rows h (row := 1966) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_4
  simp only [← c119964, ← c119965, ← c119966] at e_1966_4
  have e_1966_5 := arithEq_of_rows h (row := 1966) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_5
  simp only [← c119967, ← c119968, k2, ← c119969] at e_1966_5
  have hr := e_1966_5
  simp only [e_1966_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19244 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 31) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12396)) := by
  have c119970 := (publicBatchWrapper89_copies3749 a h).2.2.1
  have c119971 := (publicBatchWrapper89_copies3749 a h).2.2.2.1
  have c119972 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.1
  have c119973 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.1
  have c119974 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.1
  have c119975 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_6 := arithEq_of_rows h (row := 1966) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_6
  simp only [← c119970, ← c119971, ← c119972] at e_1966_6
  have e_1966_7 := arithEq_of_rows h (row := 1966) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_7
  simp only [← c119973, ← c119974, k2, ← c119975] at e_1966_7
  have hr := e_1966_7
  simp only [e_1966_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19245 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 39) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12397)) := by
  have c119976 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.1
  have c119977 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.1
  have c119978 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.1
  have c119979 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c119980 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119981 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_8 := arithEq_of_rows h (row := 1966) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_8
  simp only [← c119976, ← c119977, ← c119978] at e_1966_8
  have e_1966_9 := arithEq_of_rows h (row := 1966) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_9
  simp only [← c119979, ← c119980, k2, ← c119981] at e_1966_9
  have hr := e_1966_9
  simp only [e_1966_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19246 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 47) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12398)) := by
  have c119982 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119983 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119984 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119985 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119986 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119987 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_10 := arithEq_of_rows h (row := 1966) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_10
  simp only [← c119982, ← c119983, ← c119984] at e_1966_10
  have e_1966_11 := arithEq_of_rows h (row := 1966) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_11
  simp only [← c119985, ← c119986, k2, ← c119987] at e_1966_11
  have hr := e_1966_11
  simp only [e_1966_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19247 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 55) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12399)) := by
  have c119988 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119989 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119990 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119991 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119992 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119993 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_12 := arithEq_of_rows h (row := 1966) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_12
  simp only [← c119988, ← c119989, ← c119990] at e_1966_12
  have e_1966_13 := arithEq_of_rows h (row := 1966) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_13
  simp only [← c119991, ← c119992, k2, ← c119993] at e_1966_13
  have hr := e_1966_13
  simp only [e_1966_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19248 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 63) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12400)) := by
  have c119994 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119995 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119996 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119997 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119998 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c119999 := (publicBatchWrapper89_copies3749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_14 := arithEq_of_rows h (row := 1966) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_14
  simp only [← c119994, ← c119995, ← c119996] at e_1966_14
  have e_1966_15 := arithEq_of_rows h (row := 1966) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_15
  simp only [← c119997, ← c119998, k2, ← c119999] at e_1966_15
  have hr := e_1966_15
  simp only [e_1966_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19249 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 71) = bselect (a (.wire 60 23)) (a (.virt 200)) (a (.virt 12401)) := by
  have c120000 := (publicBatchWrapper89_copies3750 a h).1
  have c120001 := (publicBatchWrapper89_copies3750 a h).2.1
  have c120002 := (publicBatchWrapper89_copies3750 a h).2.2.1
  have c120003 := (publicBatchWrapper89_copies3750 a h).2.2.2.1
  have c120004 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.1
  have c120005 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_16 := arithEq_of_rows h (row := 1966) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_16
  simp only [← c120000, ← c120001, ← c120002] at e_1966_16
  have e_1966_17 := arithEq_of_rows h (row := 1966) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_17
  simp only [← c120003, ← c120004, k2, ← c120005] at e_1966_17
  have hr := e_1966_17
  simp only [e_1966_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19250 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1966 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12570)) := by
  have c120006 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.1
  have c120007 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.1
  have c120008 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.1
  have c120009 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.1
  have c120010 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120011 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1966_18 := arithEq_of_rows h (row := 1966) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_18
  simp only [← c120006, ← c120007, ← c120008] at e_1966_18
  have e_1966_19 := arithEq_of_rows h (row := 1966) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1966_19
  simp only [← c120009, ← c120010, k2, ← c120011] at e_1966_19
  have hr := e_1966_19
  simp only [e_1966_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19251 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12571)) := by
  have c120012 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120013 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120014 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120015 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120016 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120017 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_0 := arithEq_of_rows h (row := 1967) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_0
  simp only [← c120012, ← c120013, ← c120014] at e_1967_0
  have e_1967_1 := arithEq_of_rows h (row := 1967) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_1
  simp only [← c120015, ← c120016, k2, ← c120017] at e_1967_1
  have hr := e_1967_1
  simp only [e_1967_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19252 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12572)) := by
  have c120018 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120019 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120020 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120021 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120022 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120023 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_2 := arithEq_of_rows h (row := 1967) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_2
  simp only [← c120018, ← c120019, ← c120020] at e_1967_2
  have e_1967_3 := arithEq_of_rows h (row := 1967) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_3
  simp only [← c120021, ← c120022, k2, ← c120023] at e_1967_3
  have hr := e_1967_3
  simp only [e_1967_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19253 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12573)) := by
  have c120024 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120025 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120026 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120027 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120028 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120029 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_4 := arithEq_of_rows h (row := 1967) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_4
  simp only [← c120024, ← c120025, ← c120026] at e_1967_4
  have e_1967_5 := arithEq_of_rows h (row := 1967) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_5
  simp only [← c120027, ← c120028, k2, ← c120029] at e_1967_5
  have hr := e_1967_5
  simp only [e_1967_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19254 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12574)) := by
  have c120030 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120031 := (publicBatchWrapper89_copies3750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120032 := (publicBatchWrapper89_copies3751 a h).1
  have c120033 := (publicBatchWrapper89_copies3751 a h).2.1
  have c120034 := (publicBatchWrapper89_copies3751 a h).2.2.1
  have c120035 := (publicBatchWrapper89_copies3751 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_6 := arithEq_of_rows h (row := 1967) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_6
  simp only [← c120030, ← c120031, ← c120032] at e_1967_6
  have e_1967_7 := arithEq_of_rows h (row := 1967) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_7
  simp only [← c120033, ← c120034, k2, ← c120035] at e_1967_7
  have hr := e_1967_7
  simp only [e_1967_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19255 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12575)) := by
  have c120036 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.1
  have c120037 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.1
  have c120038 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.1
  have c120039 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.1
  have c120040 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.1
  have c120041 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_8 := arithEq_of_rows h (row := 1967) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_8
  simp only [← c120036, ← c120037, ← c120038] at e_1967_8
  have e_1967_9 := arithEq_of_rows h (row := 1967) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_9
  simp only [← c120039, ← c120040, k2, ← c120041] at e_1967_9
  have hr := e_1967_9
  simp only [e_1967_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19256 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12576)) := by
  have c120042 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120043 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120044 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120045 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120046 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120047 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_10 := arithEq_of_rows h (row := 1967) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_10
  simp only [← c120042, ← c120043, ← c120044] at e_1967_10
  have e_1967_11 := arithEq_of_rows h (row := 1967) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_11
  simp only [← c120045, ← c120046, k2, ← c120047] at e_1967_11
  have hr := e_1967_11
  simp only [e_1967_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19257 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12577)) := by
  have c120048 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120049 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120050 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120051 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120052 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120053 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_12 := arithEq_of_rows h (row := 1967) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_12
  simp only [← c120048, ← c120049, ← c120050] at e_1967_12
  have e_1967_13 := arithEq_of_rows h (row := 1967) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_13
  simp only [← c120051, ← c120052, k2, ← c120053] at e_1967_13
  have hr := e_1967_13
  simp only [e_1967_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19258 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12578)) := by
  have c120054 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120055 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120056 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120057 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120058 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120059 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_14 := arithEq_of_rows h (row := 1967) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_14
  simp only [← c120054, ← c120055, ← c120056] at e_1967_14
  have e_1967_15 := arithEq_of_rows h (row := 1967) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_15
  simp only [← c120057, ← c120058, k2, ← c120059] at e_1967_15
  have hr := e_1967_15
  simp only [e_1967_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19259 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12579)) := by
  have c120060 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120061 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120062 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120063 := (publicBatchWrapper89_copies3751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120064 := (publicBatchWrapper89_copies3752 a h).1
  have c120065 := (publicBatchWrapper89_copies3752 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_16 := arithEq_of_rows h (row := 1967) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_16
  simp only [← c120060, ← c120061, ← c120062] at e_1967_16
  have e_1967_17 := arithEq_of_rows h (row := 1967) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_17
  simp only [← c120063, ← c120064, k2, ← c120065] at e_1967_17
  have hr := e_1967_17
  simp only [e_1967_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19260 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1967 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12580)) := by
  have c120066 := (publicBatchWrapper89_copies3752 a h).2.2.1
  have c120067 := (publicBatchWrapper89_copies3752 a h).2.2.2.1
  have c120068 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.1
  have c120069 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.1
  have c120070 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.1
  have c120071 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1967_18 := arithEq_of_rows h (row := 1967) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_18
  simp only [← c120066, ← c120067, ← c120068] at e_1967_18
  have e_1967_19 := arithEq_of_rows h (row := 1967) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1967_19
  simp only [← c120069, ← c120070, k2, ← c120071] at e_1967_19
  have hr := e_1967_19
  simp only [e_1967_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19261 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12581)) := by
  have c120072 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.1
  have c120073 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.1
  have c120074 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120075 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120076 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120077 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_0 := arithEq_of_rows h (row := 1968) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_0
  simp only [← c120072, ← c120073, ← c120074] at e_1968_0
  have e_1968_1 := arithEq_of_rows h (row := 1968) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_1
  simp only [← c120075, ← c120076, k2, ← c120077] at e_1968_1
  have hr := e_1968_1
  simp only [e_1968_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19262 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12582)) := by
  have c120078 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120079 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120080 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120081 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120082 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120083 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_2 := arithEq_of_rows h (row := 1968) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_2
  simp only [← c120078, ← c120079, ← c120080] at e_1968_2
  have e_1968_3 := arithEq_of_rows h (row := 1968) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_3
  simp only [← c120081, ← c120082, k2, ← c120083] at e_1968_3
  have hr := e_1968_3
  simp only [e_1968_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19263 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12583)) := by
  have c120084 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120085 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120086 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120087 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120088 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120089 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_4 := arithEq_of_rows h (row := 1968) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_4
  simp only [← c120084, ← c120085, ← c120086] at e_1968_4
  have e_1968_5 := arithEq_of_rows h (row := 1968) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_5
  simp only [← c120087, ← c120088, k2, ← c120089] at e_1968_5
  have hr := e_1968_5
  simp only [e_1968_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19264 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12584)) := by
  have c120090 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120091 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120092 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120093 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120094 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120095 := (publicBatchWrapper89_copies3752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_6 := arithEq_of_rows h (row := 1968) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_6
  simp only [← c120090, ← c120091, ← c120092] at e_1968_6
  have e_1968_7 := arithEq_of_rows h (row := 1968) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_7
  simp only [← c120093, ← c120094, k2, ← c120095] at e_1968_7
  have hr := e_1968_7
  simp only [e_1968_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19265 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12585)) := by
  have c120096 := (publicBatchWrapper89_copies3753 a h).1
  have c120097 := (publicBatchWrapper89_copies3753 a h).2.1
  have c120098 := (publicBatchWrapper89_copies3753 a h).2.2.1
  have c120099 := (publicBatchWrapper89_copies3753 a h).2.2.2.1
  have c120100 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.1
  have c120101 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_8 := arithEq_of_rows h (row := 1968) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_8
  simp only [← c120096, ← c120097, ← c120098] at e_1968_8
  have e_1968_9 := arithEq_of_rows h (row := 1968) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_9
  simp only [← c120099, ← c120100, k2, ← c120101] at e_1968_9
  have hr := e_1968_9
  simp only [e_1968_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19266 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12586)) := by
  have c120102 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.1
  have c120103 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.1
  have c120104 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.1
  have c120105 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.1
  have c120106 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120107 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_10 := arithEq_of_rows h (row := 1968) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_10
  simp only [← c120102, ← c120103, ← c120104] at e_1968_10
  have e_1968_11 := arithEq_of_rows h (row := 1968) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_11
  simp only [← c120105, ← c120106, k2, ← c120107] at e_1968_11
  have hr := e_1968_11
  simp only [e_1968_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19267 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12587)) := by
  have c120108 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120109 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120110 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120111 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120112 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120113 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_12 := arithEq_of_rows h (row := 1968) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_12
  simp only [← c120108, ← c120109, ← c120110] at e_1968_12
  have e_1968_13 := arithEq_of_rows h (row := 1968) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_13
  simp only [← c120111, ← c120112, k2, ← c120113] at e_1968_13
  have hr := e_1968_13
  simp only [e_1968_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19268 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12588)) := by
  have c120114 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120115 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120116 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120117 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120118 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120119 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_14 := arithEq_of_rows h (row := 1968) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_14
  simp only [← c120114, ← c120115, ← c120116] at e_1968_14
  have e_1968_15 := arithEq_of_rows h (row := 1968) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_15
  simp only [← c120117, ← c120118, k2, ← c120119] at e_1968_15
  have hr := e_1968_15
  simp only [e_1968_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19269 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12589)) := by
  have c120120 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120121 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120122 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120123 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120124 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120125 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_16 := arithEq_of_rows h (row := 1968) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_16
  simp only [← c120120, ← c120121, ← c120122] at e_1968_16
  have e_1968_17 := arithEq_of_rows h (row := 1968) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_17
  simp only [← c120123, ← c120124, k2, ← c120125] at e_1968_17
  have hr := e_1968_17
  simp only [e_1968_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19270 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1968 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12590)) := by
  have c120126 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120127 := (publicBatchWrapper89_copies3753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120128 := (publicBatchWrapper89_copies3754 a h).1
  have c120129 := (publicBatchWrapper89_copies3754 a h).2.1
  have c120130 := (publicBatchWrapper89_copies3754 a h).2.2.1
  have c120131 := (publicBatchWrapper89_copies3754 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1968_18 := arithEq_of_rows h (row := 1968) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_18
  simp only [← c120126, ← c120127, ← c120128] at e_1968_18
  have e_1968_19 := arithEq_of_rows h (row := 1968) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1968_19
  simp only [← c120129, ← c120130, k2, ← c120131] at e_1968_19
  have hr := e_1968_19
  simp only [e_1968_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19271 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12591)) := by
  have c120132 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.1
  have c120133 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.1
  have c120134 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.1
  have c120135 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.1
  have c120136 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.1
  have c120137 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_0 := arithEq_of_rows h (row := 1969) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_0
  simp only [← c120132, ← c120133, ← c120134] at e_1969_0
  have e_1969_1 := arithEq_of_rows h (row := 1969) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_1
  simp only [← c120135, ← c120136, k2, ← c120137] at e_1969_1
  have hr := e_1969_1
  simp only [e_1969_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19272 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 15) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12592)) := by
  have c120138 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120139 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120140 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120141 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120142 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120143 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_2 := arithEq_of_rows h (row := 1969) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_2
  simp only [← c120138, ← c120139, ← c120140] at e_1969_2
  have e_1969_3 := arithEq_of_rows h (row := 1969) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_3
  simp only [← c120141, ← c120142, k2, ← c120143] at e_1969_3
  have hr := e_1969_3
  simp only [e_1969_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19273 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 23) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12593)) := by
  have c120144 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120145 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120146 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120147 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120148 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120149 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_4 := arithEq_of_rows h (row := 1969) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_4
  simp only [← c120144, ← c120145, ← c120146] at e_1969_4
  have e_1969_5 := arithEq_of_rows h (row := 1969) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_5
  simp only [← c120147, ← c120148, k2, ← c120149] at e_1969_5
  have hr := e_1969_5
  simp only [e_1969_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19274 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 31) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12594)) := by
  have c120150 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120151 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120152 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120153 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120154 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120155 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_6 := arithEq_of_rows h (row := 1969) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_6
  simp only [← c120150, ← c120151, ← c120152] at e_1969_6
  have e_1969_7 := arithEq_of_rows h (row := 1969) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_7
  simp only [← c120153, ← c120154, k2, ← c120155] at e_1969_7
  have hr := e_1969_7
  simp only [e_1969_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19275 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 39) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12595)) := by
  have c120156 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120157 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120158 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120159 := (publicBatchWrapper89_copies3754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120160 := (publicBatchWrapper89_copies3755 a h).1
  have c120161 := (publicBatchWrapper89_copies3755 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_8 := arithEq_of_rows h (row := 1969) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_8
  simp only [← c120156, ← c120157, ← c120158] at e_1969_8
  have e_1969_9 := arithEq_of_rows h (row := 1969) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_9
  simp only [← c120159, ← c120160, k2, ← c120161] at e_1969_9
  have hr := e_1969_9
  simp only [e_1969_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19276 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 47) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12596)) := by
  have c120162 := (publicBatchWrapper89_copies3755 a h).2.2.1
  have c120163 := (publicBatchWrapper89_copies3755 a h).2.2.2.1
  have c120164 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.1
  have c120165 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.1
  have c120166 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.1
  have c120167 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_10 := arithEq_of_rows h (row := 1969) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_10
  simp only [← c120162, ← c120163, ← c120164] at e_1969_10
  have e_1969_11 := arithEq_of_rows h (row := 1969) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_11
  simp only [← c120165, ← c120166, k2, ← c120167] at e_1969_11
  have hr := e_1969_11
  simp only [e_1969_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19277 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 55) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12597)) := by
  have c120168 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.1
  have c120169 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.1
  have c120170 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120171 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120172 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120173 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_12 := arithEq_of_rows h (row := 1969) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_12
  simp only [← c120168, ← c120169, ← c120170] at e_1969_12
  have e_1969_13 := arithEq_of_rows h (row := 1969) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_13
  simp only [← c120171, ← c120172, k2, ← c120173] at e_1969_13
  have hr := e_1969_13
  simp only [e_1969_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19278 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 63) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12598)) := by
  have c120174 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120175 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120176 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120177 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120178 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120179 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_14 := arithEq_of_rows h (row := 1969) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_14
  simp only [← c120174, ← c120175, ← c120176] at e_1969_14
  have e_1969_15 := arithEq_of_rows h (row := 1969) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_15
  simp only [← c120177, ← c120178, k2, ← c120179] at e_1969_15
  have hr := e_1969_15
  simp only [e_1969_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19279 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 71) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12599)) := by
  have c120180 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120181 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120182 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120183 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120184 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120185 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_16 := arithEq_of_rows h (row := 1969) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_16
  simp only [← c120180, ← c120181, ← c120182] at e_1969_16
  have e_1969_17 := arithEq_of_rows h (row := 1969) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_17
  simp only [← c120183, ← c120184, k2, ← c120185] at e_1969_17
  have hr := e_1969_17
  simp only [e_1969_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19280 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1969 79) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12600)) := by
  have c120186 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120187 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120188 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120189 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120190 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120191 := (publicBatchWrapper89_copies3755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1969_18 := arithEq_of_rows h (row := 1969) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_18
  simp only [← c120186, ← c120187, ← c120188] at e_1969_18
  have e_1969_19 := arithEq_of_rows h (row := 1969) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1969_19
  simp only [← c120189, ← c120190, k2, ← c120191] at e_1969_19
  have hr := e_1969_19
  simp only [e_1969_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19281 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 7) = bselect (a (.wire 60 67)) (a (.virt 200)) (a (.virt 12601)) := by
  have c120192 := (publicBatchWrapper89_copies3756 a h).1
  have c120193 := (publicBatchWrapper89_copies3756 a h).2.1
  have c120194 := (publicBatchWrapper89_copies3756 a h).2.2.1
  have c120195 := (publicBatchWrapper89_copies3756 a h).2.2.2.1
  have c120196 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.1
  have c120197 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_0 := arithEq_of_rows h (row := 1970) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_0
  simp only [← c120192, ← c120193, ← c120194] at e_1970_0
  have e_1970_1 := arithEq_of_rows h (row := 1970) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_1
  simp only [← c120195, ← c120196, k2, ← c120197] at e_1970_1
  have hr := e_1970_1
  simp only [e_1970_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19282 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12770)) := by
  have c120198 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.1
  have c120199 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.1
  have c120200 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.1
  have c120201 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.1
  have c120202 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120203 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_2 := arithEq_of_rows h (row := 1970) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_2
  simp only [← c120198, ← c120199, ← c120200] at e_1970_2
  have e_1970_3 := arithEq_of_rows h (row := 1970) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_3
  simp only [← c120201, ← c120202, k2, ← c120203] at e_1970_3
  have hr := e_1970_3
  simp only [e_1970_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19283 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12771)) := by
  have c120204 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120205 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120206 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120207 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120208 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120209 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_4 := arithEq_of_rows h (row := 1970) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_4
  simp only [← c120204, ← c120205, ← c120206] at e_1970_4
  have e_1970_5 := arithEq_of_rows h (row := 1970) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_5
  simp only [← c120207, ← c120208, k2, ← c120209] at e_1970_5
  have hr := e_1970_5
  simp only [e_1970_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19284 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12772)) := by
  have c120210 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120211 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120212 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120213 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120214 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120215 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_6 := arithEq_of_rows h (row := 1970) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_6
  simp only [← c120210, ← c120211, ← c120212] at e_1970_6
  have e_1970_7 := arithEq_of_rows h (row := 1970) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_7
  simp only [← c120213, ← c120214, k2, ← c120215] at e_1970_7
  have hr := e_1970_7
  simp only [e_1970_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19285 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12773)) := by
  have c120216 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120217 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120218 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120219 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120220 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120221 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_8 := arithEq_of_rows h (row := 1970) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_8
  simp only [← c120216, ← c120217, ← c120218] at e_1970_8
  have e_1970_9 := arithEq_of_rows h (row := 1970) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_9
  simp only [← c120219, ← c120220, k2, ← c120221] at e_1970_9
  have hr := e_1970_9
  simp only [e_1970_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19286 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12774)) := by
  have c120222 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120223 := (publicBatchWrapper89_copies3756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120224 := (publicBatchWrapper89_copies3757 a h).1
  have c120225 := (publicBatchWrapper89_copies3757 a h).2.1
  have c120226 := (publicBatchWrapper89_copies3757 a h).2.2.1
  have c120227 := (publicBatchWrapper89_copies3757 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_10 := arithEq_of_rows h (row := 1970) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_10
  simp only [← c120222, ← c120223, ← c120224] at e_1970_10
  have e_1970_11 := arithEq_of_rows h (row := 1970) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_11
  simp only [← c120225, ← c120226, k2, ← c120227] at e_1970_11
  have hr := e_1970_11
  simp only [e_1970_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19287 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12775)) := by
  have c120228 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.1
  have c120229 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.1
  have c120230 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.1
  have c120231 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.1
  have c120232 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.1
  have c120233 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_12 := arithEq_of_rows h (row := 1970) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_12
  simp only [← c120228, ← c120229, ← c120230] at e_1970_12
  have e_1970_13 := arithEq_of_rows h (row := 1970) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_13
  simp only [← c120231, ← c120232, k2, ← c120233] at e_1970_13
  have hr := e_1970_13
  simp only [e_1970_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19288 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12776)) := by
  have c120234 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120235 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120236 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120237 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120238 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120239 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_14 := arithEq_of_rows h (row := 1970) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_14
  simp only [← c120234, ← c120235, ← c120236] at e_1970_14
  have e_1970_15 := arithEq_of_rows h (row := 1970) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_15
  simp only [← c120237, ← c120238, k2, ← c120239] at e_1970_15
  have hr := e_1970_15
  simp only [e_1970_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19289 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12777)) := by
  have c120240 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120241 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120242 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120243 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120244 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120245 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_16 := arithEq_of_rows h (row := 1970) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_16
  simp only [← c120240, ← c120241, ← c120242] at e_1970_16
  have e_1970_17 := arithEq_of_rows h (row := 1970) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_17
  simp only [← c120243, ← c120244, k2, ← c120245] at e_1970_17
  have hr := e_1970_17
  simp only [e_1970_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19290 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1970 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12778)) := by
  have c120246 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120247 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120248 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120249 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120250 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120251 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1970_18 := arithEq_of_rows h (row := 1970) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_18
  simp only [← c120246, ← c120247, ← c120248] at e_1970_18
  have e_1970_19 := arithEq_of_rows h (row := 1970) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1970_19
  simp only [← c120249, ← c120250, k2, ← c120251] at e_1970_19
  have hr := e_1970_19
  simp only [e_1970_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19291 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12779)) := by
  have c120252 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120253 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120254 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120255 := (publicBatchWrapper89_copies3757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120256 := (publicBatchWrapper89_copies3758 a h).1
  have c120257 := (publicBatchWrapper89_copies3758 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_0 := arithEq_of_rows h (row := 1971) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_0
  simp only [← c120252, ← c120253, ← c120254] at e_1971_0
  have e_1971_1 := arithEq_of_rows h (row := 1971) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_1
  simp only [← c120255, ← c120256, k2, ← c120257] at e_1971_1
  have hr := e_1971_1
  simp only [e_1971_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19292 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12780)) := by
  have c120258 := (publicBatchWrapper89_copies3758 a h).2.2.1
  have c120259 := (publicBatchWrapper89_copies3758 a h).2.2.2.1
  have c120260 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.1
  have c120261 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.1
  have c120262 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.1
  have c120263 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_2 := arithEq_of_rows h (row := 1971) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_2
  simp only [← c120258, ← c120259, ← c120260] at e_1971_2
  have e_1971_3 := arithEq_of_rows h (row := 1971) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_3
  simp only [← c120261, ← c120262, k2, ← c120263] at e_1971_3
  have hr := e_1971_3
  simp only [e_1971_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19293 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12781)) := by
  have c120264 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.1
  have c120265 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.1
  have c120266 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120267 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120268 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120269 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_4 := arithEq_of_rows h (row := 1971) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_4
  simp only [← c120264, ← c120265, ← c120266] at e_1971_4
  have e_1971_5 := arithEq_of_rows h (row := 1971) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_5
  simp only [← c120267, ← c120268, k2, ← c120269] at e_1971_5
  have hr := e_1971_5
  simp only [e_1971_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19294 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12782)) := by
  have c120270 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120271 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120272 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120273 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120274 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120275 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_6 := arithEq_of_rows h (row := 1971) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_6
  simp only [← c120270, ← c120271, ← c120272] at e_1971_6
  have e_1971_7 := arithEq_of_rows h (row := 1971) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_7
  simp only [← c120273, ← c120274, k2, ← c120275] at e_1971_7
  have hr := e_1971_7
  simp only [e_1971_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19295 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12783)) := by
  have c120276 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120277 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120278 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120279 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120280 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120281 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_8 := arithEq_of_rows h (row := 1971) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_8
  simp only [← c120276, ← c120277, ← c120278] at e_1971_8
  have e_1971_9 := arithEq_of_rows h (row := 1971) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_9
  simp only [← c120279, ← c120280, k2, ← c120281] at e_1971_9
  have hr := e_1971_9
  simp only [e_1971_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19296 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12784)) := by
  have c120282 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120283 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120284 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120285 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120286 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120287 := (publicBatchWrapper89_copies3758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_10 := arithEq_of_rows h (row := 1971) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_10
  simp only [← c120282, ← c120283, ← c120284] at e_1971_10
  have e_1971_11 := arithEq_of_rows h (row := 1971) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_11
  simp only [← c120285, ← c120286, k2, ← c120287] at e_1971_11
  have hr := e_1971_11
  simp only [e_1971_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19297 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12785)) := by
  have c120288 := (publicBatchWrapper89_copies3759 a h).1
  have c120289 := (publicBatchWrapper89_copies3759 a h).2.1
  have c120290 := (publicBatchWrapper89_copies3759 a h).2.2.1
  have c120291 := (publicBatchWrapper89_copies3759 a h).2.2.2.1
  have c120292 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.1
  have c120293 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_12 := arithEq_of_rows h (row := 1971) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_12
  simp only [← c120288, ← c120289, ← c120290] at e_1971_12
  have e_1971_13 := arithEq_of_rows h (row := 1971) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_13
  simp only [← c120291, ← c120292, k2, ← c120293] at e_1971_13
  have hr := e_1971_13
  simp only [e_1971_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19298 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12786)) := by
  have c120294 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.1
  have c120295 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.1
  have c120296 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.1
  have c120297 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.1
  have c120298 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120299 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_14 := arithEq_of_rows h (row := 1971) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_14
  simp only [← c120294, ← c120295, ← c120296] at e_1971_14
  have e_1971_15 := arithEq_of_rows h (row := 1971) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_15
  simp only [← c120297, ← c120298, k2, ← c120299] at e_1971_15
  have hr := e_1971_15
  simp only [e_1971_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19299 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12787)) := by
  have c120300 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120301 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120302 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120303 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120304 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120305 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_16 := arithEq_of_rows h (row := 1971) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_16
  simp only [← c120300, ← c120301, ← c120302] at e_1971_16
  have e_1971_17 := arithEq_of_rows h (row := 1971) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_17
  simp only [← c120303, ← c120304, k2, ← c120305] at e_1971_17
  have hr := e_1971_17
  simp only [e_1971_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19300 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1971 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12788)) := by
  have c120306 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120307 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120308 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120309 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120310 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120311 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1971_18 := arithEq_of_rows h (row := 1971) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_18
  simp only [← c120306, ← c120307, ← c120308] at e_1971_18
  have e_1971_19 := arithEq_of_rows h (row := 1971) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1971_19
  simp only [← c120309, ← c120310, k2, ← c120311] at e_1971_19
  have hr := e_1971_19
  simp only [e_1971_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19301 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12789)) := by
  have c120312 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120313 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120314 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120315 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120316 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120317 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_0 := arithEq_of_rows h (row := 1972) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_0
  simp only [← c120312, ← c120313, ← c120314] at e_1972_0
  have e_1972_1 := arithEq_of_rows h (row := 1972) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_1
  simp only [← c120315, ← c120316, k2, ← c120317] at e_1972_1
  have hr := e_1972_1
  simp only [e_1972_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19302 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12790)) := by
  have c120318 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120319 := (publicBatchWrapper89_copies3759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120320 := (publicBatchWrapper89_copies3760 a h).1
  have c120321 := (publicBatchWrapper89_copies3760 a h).2.1
  have c120322 := (publicBatchWrapper89_copies3760 a h).2.2.1
  have c120323 := (publicBatchWrapper89_copies3760 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_2 := arithEq_of_rows h (row := 1972) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_2
  simp only [← c120318, ← c120319, ← c120320] at e_1972_2
  have e_1972_3 := arithEq_of_rows h (row := 1972) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_3
  simp only [← c120321, ← c120322, k2, ← c120323] at e_1972_3
  have hr := e_1972_3
  simp only [e_1972_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19303 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12791)) := by
  have c120324 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.1
  have c120325 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.1
  have c120326 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.1
  have c120327 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.1
  have c120328 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.1
  have c120329 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_4 := arithEq_of_rows h (row := 1972) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_4
  simp only [← c120324, ← c120325, ← c120326] at e_1972_4
  have e_1972_5 := arithEq_of_rows h (row := 1972) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_5
  simp only [← c120327, ← c120328, k2, ← c120329] at e_1972_5
  have hr := e_1972_5
  simp only [e_1972_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19304 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 31) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12792)) := by
  have c120330 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120331 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120332 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120333 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120334 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120335 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_6 := arithEq_of_rows h (row := 1972) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_6
  simp only [← c120330, ← c120331, ← c120332] at e_1972_6
  have e_1972_7 := arithEq_of_rows h (row := 1972) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_7
  simp only [← c120333, ← c120334, k2, ← c120335] at e_1972_7
  have hr := e_1972_7
  simp only [e_1972_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19305 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 39) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12793)) := by
  have c120336 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120337 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120338 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120339 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120340 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120341 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_8 := arithEq_of_rows h (row := 1972) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_8
  simp only [← c120336, ← c120337, ← c120338] at e_1972_8
  have e_1972_9 := arithEq_of_rows h (row := 1972) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_9
  simp only [← c120339, ← c120340, k2, ← c120341] at e_1972_9
  have hr := e_1972_9
  simp only [e_1972_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19306 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 47) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12794)) := by
  have c120342 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120343 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120344 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120345 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120346 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120347 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_10 := arithEq_of_rows h (row := 1972) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_10
  simp only [← c120342, ← c120343, ← c120344] at e_1972_10
  have e_1972_11 := arithEq_of_rows h (row := 1972) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_11
  simp only [← c120345, ← c120346, k2, ← c120347] at e_1972_11
  have hr := e_1972_11
  simp only [e_1972_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19307 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 55) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12795)) := by
  have c120348 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120349 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120350 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120351 := (publicBatchWrapper89_copies3760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c120352 := (publicBatchWrapper89_copies3761 a h).1
  have c120353 := (publicBatchWrapper89_copies3761 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_12 := arithEq_of_rows h (row := 1972) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_12
  simp only [← c120348, ← c120349, ← c120350] at e_1972_12
  have e_1972_13 := arithEq_of_rows h (row := 1972) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_13
  simp only [← c120351, ← c120352, k2, ← c120353] at e_1972_13
  have hr := e_1972_13
  simp only [e_1972_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19308 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 63) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12796)) := by
  have c120354 := (publicBatchWrapper89_copies3761 a h).2.2.1
  have c120355 := (publicBatchWrapper89_copies3761 a h).2.2.2.1
  have c120356 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.1
  have c120357 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.1
  have c120358 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.1
  have c120359 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_14 := arithEq_of_rows h (row := 1972) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_14
  simp only [← c120354, ← c120355, ← c120356] at e_1972_14
  have e_1972_15 := arithEq_of_rows h (row := 1972) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_15
  simp only [← c120357, ← c120358, k2, ← c120359] at e_1972_15
  have hr := e_1972_15
  simp only [e_1972_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19309 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 71) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12797)) := by
  have c120360 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.1
  have c120361 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.1
  have c120362 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120363 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c120364 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120365 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_16 := arithEq_of_rows h (row := 1972) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_16
  simp only [← c120360, ← c120361, ← c120362] at e_1972_16
  have e_1972_17 := arithEq_of_rows h (row := 1972) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_17
  simp only [← c120363, ← c120364, k2, ← c120365] at e_1972_17
  have hr := e_1972_17
  simp only [e_1972_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19310 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1972 79) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12798)) := by
  have c120366 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120367 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120368 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120369 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120370 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120371 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1972_18 := arithEq_of_rows h (row := 1972) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_18
  simp only [← c120366, ← c120367, ← c120368] at e_1972_18
  have e_1972_19 := arithEq_of_rows h (row := 1972) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1972_19
  simp only [← c120369, ← c120370, k2, ← c120371] at e_1972_19
  have hr := e_1972_19
  simp only [e_1972_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19311 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 7) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12799)) := by
  have c120372 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120373 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120374 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120375 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120376 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120377 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_0 := arithEq_of_rows h (row := 1973) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_0
  simp only [← c120372, ← c120373, ← c120374] at e_1973_0
  have e_1973_1 := arithEq_of_rows h (row := 1973) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_1
  simp only [← c120375, ← c120376, k2, ← c120377] at e_1973_1
  have hr := e_1973_1
  simp only [e_1973_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19312 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 15) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12800)) := by
  have c120378 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120379 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120380 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120381 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120382 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120383 := (publicBatchWrapper89_copies3761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_2 := arithEq_of_rows h (row := 1973) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_2
  simp only [← c120378, ← c120379, ← c120380] at e_1973_2
  have e_1973_3 := arithEq_of_rows h (row := 1973) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_3
  simp only [← c120381, ← c120382, k2, ← c120383] at e_1973_3
  have hr := e_1973_3
  simp only [e_1973_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19313 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 23) = bselect (a (.wire 62 31)) (a (.virt 200)) (a (.virt 12801)) := by
  have c120384 := (publicBatchWrapper89_copies3762 a h).1
  have c120385 := (publicBatchWrapper89_copies3762 a h).2.1
  have c120386 := (publicBatchWrapper89_copies3762 a h).2.2.1
  have c120387 := (publicBatchWrapper89_copies3762 a h).2.2.2.1
  have c120388 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.1
  have c120389 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_4 := arithEq_of_rows h (row := 1973) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_4
  simp only [← c120384, ← c120385, ← c120386] at e_1973_4
  have e_1973_5 := arithEq_of_rows h (row := 1973) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_5
  simp only [← c120387, ← c120388, k2, ← c120389] at e_1973_5
  have hr := e_1973_5
  simp only [e_1973_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19314 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 31) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12970)) := by
  have c120390 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.1
  have c120391 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.1
  have c120392 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.1
  have c120393 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.1
  have c120394 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.1
  have c120395 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_6 := arithEq_of_rows h (row := 1973) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_6
  simp only [← c120390, ← c120391, ← c120392] at e_1973_6
  have e_1973_7 := arithEq_of_rows h (row := 1973) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_7
  simp only [← c120393, ← c120394, k2, ← c120395] at e_1973_7
  have hr := e_1973_7
  simp only [e_1973_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19315 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 39) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12971)) := by
  have c120396 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120397 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120398 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120399 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120400 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120401 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_8 := arithEq_of_rows h (row := 1973) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_8
  simp only [← c120396, ← c120397, ← c120398] at e_1973_8
  have e_1973_9 := arithEq_of_rows h (row := 1973) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_9
  simp only [← c120399, ← c120400, k2, ← c120401] at e_1973_9
  have hr := e_1973_9
  simp only [e_1973_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19316 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 47) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12972)) := by
  have c120402 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120403 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120404 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120405 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120406 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120407 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_10 := arithEq_of_rows h (row := 1973) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_10
  simp only [← c120402, ← c120403, ← c120404] at e_1973_10
  have e_1973_11 := arithEq_of_rows h (row := 1973) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_11
  simp only [← c120405, ← c120406, k2, ← c120407] at e_1973_11
  have hr := e_1973_11
  simp only [e_1973_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper89_f19317 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 1973 55) = bselect (a (.wire 62 75)) (a (.virt 200)) (a (.virt 12973)) := by
  have c120408 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120409 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120410 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120411 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120412 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c120413 := (publicBatchWrapper89_copies3762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_1973_12 := arithEq_of_rows h (row := 1973) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_12
  simp only [← c120408, ← c120409, ← c120410] at e_1973_12
  have e_1973_13 := arithEq_of_rows h (row := 1973) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1973_13
  simp only [← c120411, ← c120412, k2, ← c120413] at e_1973_13
  have hr := e_1973_13
  simp only [e_1973_12] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
