/-
  AUTO-GENERATED — do not edit by hand.

  Part of `Plonky2Spec.Generated.PublicBatchWrapper89`, split out of the recorded wrapper
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

set_option linter.all false

variable {p : ℕ} [Fact p.Prime]

theorem publicBatchWrapper89_f20041 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 7) = bselect (a (.wire 82 67)) (a (.virt 604760)) (a (.virt 591156)) := by
  have c124752 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124753 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124754 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124755 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124756 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124757 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_0 := arithEq_of_rows h (row := 2045) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_0
  simp only [← c124752, ← c124753, ← c124754] at e_2045_0
  have e_2045_1 := arithEq_of_rows h (row := 2045) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_1
  simp only [← c124755, ← c124756, k1, ← c124757] at e_2045_1
  have hr := e_2045_1
  simp only [e_2045_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20042 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 15) = bselect (a (.wire 82 67)) (a (.virt 604760)) (a (.virt 591157)) := by
  have c124758 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124759 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124760 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124761 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124762 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124763 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_2 := arithEq_of_rows h (row := 2045) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_2
  simp only [← c124758, ← c124759, ← c124760] at e_2045_2
  have e_2045_3 := arithEq_of_rows h (row := 2045) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_3
  simp only [← c124761, ← c124762, k1, ← c124763] at e_2045_3
  have hr := e_2045_3
  simp only [e_2045_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20043 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 23) = bselect (a (.wire 82 67)) (a (.virt 604760)) (a (.virt 591158)) := by
  have c124764 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124765 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124766 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124767 := (publicBatchWrapper89_copies3898 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124768 := (publicBatchWrapper89_copies3899 a h).1
  have c124769 := (publicBatchWrapper89_copies3899 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_4 := arithEq_of_rows h (row := 2045) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_4
  simp only [← c124764, ← c124765, ← c124766] at e_2045_4
  have e_2045_5 := arithEq_of_rows h (row := 2045) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_5
  simp only [← c124767, ← c124768, k1, ← c124769] at e_2045_5
  have hr := e_2045_5
  simp only [e_2045_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20044 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 31) = bselect (a (.wire 82 67)) (a (.virt 604760)) (a (.virt 591159)) := by
  have c124770 := (publicBatchWrapper89_copies3899 a h).2.2.1
  have c124771 := (publicBatchWrapper89_copies3899 a h).2.2.2.1
  have c124772 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.1
  have c124773 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.1
  have c124774 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.1
  have c124775 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_6 := arithEq_of_rows h (row := 2045) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_6
  simp only [← c124770, ← c124771, ← c124772] at e_2045_6
  have e_2045_7 := arithEq_of_rows h (row := 2045) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_7
  simp only [← c124773, ← c124774, k1, ← c124775] at e_2045_7
  have hr := e_2045_7
  simp only [e_2045_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20045 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 39) = bselect (a (.wire 82 67)) (a (.virt 604760)) (a (.virt 591160)) := by
  have c124776 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.1
  have c124777 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.1
  have c124778 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124779 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124780 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124781 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_8 := arithEq_of_rows h (row := 2045) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_8
  simp only [← c124776, ← c124777, ← c124778] at e_2045_8
  have e_2045_9 := arithEq_of_rows h (row := 2045) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_9
  simp only [← c124779, ← c124780, k1, ← c124781] at e_2045_9
  have hr := e_2045_9
  simp only [e_2045_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20046 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 47) = bselect (a (.wire 82 67)) (a (.virt 604760)) (a (.virt 591161)) := by
  have c124782 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124783 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124784 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124785 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124786 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124787 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_10 := arithEq_of_rows h (row := 2045) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_10
  simp only [← c124782, ← c124783, ← c124784] at e_2045_10
  have e_2045_11 := arithEq_of_rows h (row := 2045) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_11
  simp only [← c124785, ← c124786, k1, ← c124787] at e_2045_11
  have hr := e_2045_11
  simp only [e_2045_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20047 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 55) = bselect (a (.wire 82 67)) (a (.virt 604760)) (a (.virt 591162)) := by
  have c124788 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124789 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124790 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124791 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124792 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124793 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_12 := arithEq_of_rows h (row := 2045) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_12
  simp only [← c124788, ← c124789, ← c124790] at e_2045_12
  have e_2045_13 := arithEq_of_rows h (row := 2045) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_13
  simp only [← c124791, ← c124792, k1, ← c124793] at e_2045_13
  have hr := e_2045_13
  simp only [e_2045_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20048 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 63) = bselect (a (.wire 82 67)) (a (.virt 604760)) (a (.virt 591163)) := by
  have c124794 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124795 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124796 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124797 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124798 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124799 := (publicBatchWrapper89_copies3899 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_14 := arithEq_of_rows h (row := 2045) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_14
  simp only [← c124794, ← c124795, ← c124796] at e_2045_14
  have e_2045_15 := arithEq_of_rows h (row := 2045) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_15
  simp only [← c124797, ← c124798, k1, ← c124799] at e_2045_15
  have hr := e_2045_15
  simp only [e_2045_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20049 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 71) = bselect (a (.wire 82 67)) (a (.virt 604760)) (a (.virt 591164)) := by
  have c124800 := (publicBatchWrapper89_copies3900 a h).1
  have c124801 := (publicBatchWrapper89_copies3900 a h).2.1
  have c124802 := (publicBatchWrapper89_copies3900 a h).2.2.1
  have c124803 := (publicBatchWrapper89_copies3900 a h).2.2.2.1
  have c124804 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.1
  have c124805 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_16 := arithEq_of_rows h (row := 2045) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_16
  simp only [← c124800, ← c124801, ← c124802] at e_2045_16
  have e_2045_17 := arithEq_of_rows h (row := 2045) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_17
  simp only [← c124803, ← c124804, k1, ← c124805] at e_2045_17
  have hr := e_2045_17
  simp only [e_2045_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20050 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2045 79) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597928)) := by
  have c124806 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.1
  have c124807 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.1
  have c124808 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.1
  have c124809 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.1
  have c124810 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124811 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2045_18 := arithEq_of_rows h (row := 2045) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_18
  simp only [← c124806, ← c124807, ← c124808] at e_2045_18
  have e_2045_19 := arithEq_of_rows h (row := 2045) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2045_19
  simp only [← c124809, ← c124810, k1, ← c124811] at e_2045_19
  have hr := e_2045_19
  simp only [e_2045_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20051 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 7) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597929)) := by
  have c124812 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124813 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124814 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124815 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124816 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124817 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_0 := arithEq_of_rows h (row := 2046) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_0
  simp only [← c124812, ← c124813, ← c124814] at e_2046_0
  have e_2046_1 := arithEq_of_rows h (row := 2046) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_1
  simp only [← c124815, ← c124816, k1, ← c124817] at e_2046_1
  have hr := e_2046_1
  simp only [e_2046_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20052 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 15) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597930)) := by
  have c124818 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124819 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124820 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124821 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124822 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124823 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_2 := arithEq_of_rows h (row := 2046) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_2
  simp only [← c124818, ← c124819, ← c124820] at e_2046_2
  have e_2046_3 := arithEq_of_rows h (row := 2046) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_3
  simp only [← c124821, ← c124822, k1, ← c124823] at e_2046_3
  have hr := e_2046_3
  simp only [e_2046_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20053 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 23) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597931)) := by
  have c124824 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124825 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124826 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124827 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124828 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124829 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_4 := arithEq_of_rows h (row := 2046) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_4
  simp only [← c124824, ← c124825, ← c124826] at e_2046_4
  have e_2046_5 := arithEq_of_rows h (row := 2046) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_5
  simp only [← c124827, ← c124828, k1, ← c124829] at e_2046_5
  have hr := e_2046_5
  simp only [e_2046_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20054 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 31) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597932)) := by
  have c124830 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124831 := (publicBatchWrapper89_copies3900 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124832 := (publicBatchWrapper89_copies3901 a h).1
  have c124833 := (publicBatchWrapper89_copies3901 a h).2.1
  have c124834 := (publicBatchWrapper89_copies3901 a h).2.2.1
  have c124835 := (publicBatchWrapper89_copies3901 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_6 := arithEq_of_rows h (row := 2046) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_6
  simp only [← c124830, ← c124831, ← c124832] at e_2046_6
  have e_2046_7 := arithEq_of_rows h (row := 2046) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_7
  simp only [← c124833, ← c124834, k1, ← c124835] at e_2046_7
  have hr := e_2046_7
  simp only [e_2046_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20055 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 39) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597933)) := by
  have c124836 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.1
  have c124837 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.1
  have c124838 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.1
  have c124839 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.1
  have c124840 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.1
  have c124841 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_8 := arithEq_of_rows h (row := 2046) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_8
  simp only [← c124836, ← c124837, ← c124838] at e_2046_8
  have e_2046_9 := arithEq_of_rows h (row := 2046) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_9
  simp only [← c124839, ← c124840, k1, ← c124841] at e_2046_9
  have hr := e_2046_9
  simp only [e_2046_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20056 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 47) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597934)) := by
  have c124842 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124843 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124844 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124845 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124846 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124847 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_10 := arithEq_of_rows h (row := 2046) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_10
  simp only [← c124842, ← c124843, ← c124844] at e_2046_10
  have e_2046_11 := arithEq_of_rows h (row := 2046) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_11
  simp only [← c124845, ← c124846, k1, ← c124847] at e_2046_11
  have hr := e_2046_11
  simp only [e_2046_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20057 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 55) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597935)) := by
  have c124848 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124849 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124850 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124851 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124852 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124853 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_12 := arithEq_of_rows h (row := 2046) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_12
  simp only [← c124848, ← c124849, ← c124850] at e_2046_12
  have e_2046_13 := arithEq_of_rows h (row := 2046) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_13
  simp only [← c124851, ← c124852, k1, ← c124853] at e_2046_13
  have hr := e_2046_13
  simp only [e_2046_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20058 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 63) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597936)) := by
  have c124854 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124855 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124856 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124857 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124858 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124859 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_14 := arithEq_of_rows h (row := 2046) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_14
  simp only [← c124854, ← c124855, ← c124856] at e_2046_14
  have e_2046_15 := arithEq_of_rows h (row := 2046) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_15
  simp only [← c124857, ← c124858, k1, ← c124859] at e_2046_15
  have hr := e_2046_15
  simp only [e_2046_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20059 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 71) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597937)) := by
  have c124860 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124861 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124862 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124863 := (publicBatchWrapper89_copies3901 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124864 := (publicBatchWrapper89_copies3902 a h).1
  have c124865 := (publicBatchWrapper89_copies3902 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_16 := arithEq_of_rows h (row := 2046) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_16
  simp only [← c124860, ← c124861, ← c124862] at e_2046_16
  have e_2046_17 := arithEq_of_rows h (row := 2046) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_17
  simp only [← c124863, ← c124864, k1, ← c124865] at e_2046_17
  have hr := e_2046_17
  simp only [e_2046_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20060 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2046 79) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597938)) := by
  have c124866 := (publicBatchWrapper89_copies3902 a h).2.2.1
  have c124867 := (publicBatchWrapper89_copies3902 a h).2.2.2.1
  have c124868 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.1
  have c124869 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.1
  have c124870 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.1
  have c124871 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2046_18 := arithEq_of_rows h (row := 2046) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_18
  simp only [← c124866, ← c124867, ← c124868] at e_2046_18
  have e_2046_19 := arithEq_of_rows h (row := 2046) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2046_19
  simp only [← c124869, ← c124870, k1, ← c124871] at e_2046_19
  have hr := e_2046_19
  simp only [e_2046_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20061 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 7) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597939)) := by
  have c124872 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.1
  have c124873 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.1
  have c124874 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124875 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124876 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124877 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_0 := arithEq_of_rows h (row := 2047) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_0
  simp only [← c124872, ← c124873, ← c124874] at e_2047_0
  have e_2047_1 := arithEq_of_rows h (row := 2047) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_1
  simp only [← c124875, ← c124876, k1, ← c124877] at e_2047_1
  have hr := e_2047_1
  simp only [e_2047_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20062 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 15) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597940)) := by
  have c124878 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124879 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124880 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124881 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124882 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124883 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_2 := arithEq_of_rows h (row := 2047) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_2
  simp only [← c124878, ← c124879, ← c124880] at e_2047_2
  have e_2047_3 := arithEq_of_rows h (row := 2047) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_3
  simp only [← c124881, ← c124882, k1, ← c124883] at e_2047_3
  have hr := e_2047_3
  simp only [e_2047_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20063 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 23) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597941)) := by
  have c124884 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124885 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124886 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124887 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124888 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124889 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_4 := arithEq_of_rows h (row := 2047) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_4
  simp only [← c124884, ← c124885, ← c124886] at e_2047_4
  have e_2047_5 := arithEq_of_rows h (row := 2047) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_5
  simp only [← c124887, ← c124888, k1, ← c124889] at e_2047_5
  have hr := e_2047_5
  simp only [e_2047_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20064 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 31) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597942)) := by
  have c124890 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124891 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124892 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124893 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124894 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124895 := (publicBatchWrapper89_copies3902 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_6 := arithEq_of_rows h (row := 2047) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_6
  simp only [← c124890, ← c124891, ← c124892] at e_2047_6
  have e_2047_7 := arithEq_of_rows h (row := 2047) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_7
  simp only [← c124893, ← c124894, k1, ← c124895] at e_2047_7
  have hr := e_2047_7
  simp only [e_2047_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20065 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 39) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597943)) := by
  have c124896 := (publicBatchWrapper89_copies3903 a h).1
  have c124897 := (publicBatchWrapper89_copies3903 a h).2.1
  have c124898 := (publicBatchWrapper89_copies3903 a h).2.2.1
  have c124899 := (publicBatchWrapper89_copies3903 a h).2.2.2.1
  have c124900 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.1
  have c124901 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_8 := arithEq_of_rows h (row := 2047) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_8
  simp only [← c124896, ← c124897, ← c124898] at e_2047_8
  have e_2047_9 := arithEq_of_rows h (row := 2047) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_9
  simp only [← c124899, ← c124900, k1, ← c124901] at e_2047_9
  have hr := e_2047_9
  simp only [e_2047_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20066 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 47) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597944)) := by
  have c124902 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.1
  have c124903 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.1
  have c124904 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.1
  have c124905 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.1
  have c124906 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124907 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_10 := arithEq_of_rows h (row := 2047) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_10
  simp only [← c124902, ← c124903, ← c124904] at e_2047_10
  have e_2047_11 := arithEq_of_rows h (row := 2047) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_11
  simp only [← c124905, ← c124906, k1, ← c124907] at e_2047_11
  have hr := e_2047_11
  simp only [e_2047_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20067 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 55) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597945)) := by
  have c124908 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124909 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124910 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124911 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124912 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124913 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_12 := arithEq_of_rows h (row := 2047) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_12
  simp only [← c124908, ← c124909, ← c124910] at e_2047_12
  have e_2047_13 := arithEq_of_rows h (row := 2047) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_13
  simp only [← c124911, ← c124912, k1, ← c124913] at e_2047_13
  have hr := e_2047_13
  simp only [e_2047_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20068 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 63) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597946)) := by
  have c124914 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124915 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124916 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124917 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124918 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124919 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_14 := arithEq_of_rows h (row := 2047) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_14
  simp only [← c124914, ← c124915, ← c124916] at e_2047_14
  have e_2047_15 := arithEq_of_rows h (row := 2047) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_15
  simp only [← c124917, ← c124918, k1, ← c124919] at e_2047_15
  have hr := e_2047_15
  simp only [e_2047_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20069 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 71) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597947)) := by
  have c124920 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124921 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124922 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124923 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124924 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124925 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_16 := arithEq_of_rows h (row := 2047) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_16
  simp only [← c124920, ← c124921, ← c124922] at e_2047_16
  have e_2047_17 := arithEq_of_rows h (row := 2047) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_17
  simp only [← c124923, ← c124924, k1, ← c124925] at e_2047_17
  have hr := e_2047_17
  simp only [e_2047_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20070 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2047 79) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597948)) := by
  have c124926 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124927 := (publicBatchWrapper89_copies3903 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124928 := (publicBatchWrapper89_copies3904 a h).1
  have c124929 := (publicBatchWrapper89_copies3904 a h).2.1
  have c124930 := (publicBatchWrapper89_copies3904 a h).2.2.1
  have c124931 := (publicBatchWrapper89_copies3904 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2047_18 := arithEq_of_rows h (row := 2047) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_18
  simp only [← c124926, ← c124927, ← c124928] at e_2047_18
  have e_2047_19 := arithEq_of_rows h (row := 2047) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2047_19
  simp only [← c124929, ← c124930, k1, ← c124931] at e_2047_19
  have hr := e_2047_19
  simp only [e_2047_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20071 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 7) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597949)) := by
  have c124932 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.1
  have c124933 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.1
  have c124934 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.1
  have c124935 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.1
  have c124936 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.1
  have c124937 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_0 := arithEq_of_rows h (row := 2048) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_0
  simp only [← c124932, ← c124933, ← c124934] at e_2048_0
  have e_2048_1 := arithEq_of_rows h (row := 2048) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_1
  simp only [← c124935, ← c124936, k1, ← c124937] at e_2048_1
  have hr := e_2048_1
  simp only [e_2048_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20072 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 15) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597950)) := by
  have c124938 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124939 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124940 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124941 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124942 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124943 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_2 := arithEq_of_rows h (row := 2048) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_2
  simp only [← c124938, ← c124939, ← c124940] at e_2048_2
  have e_2048_3 := arithEq_of_rows h (row := 2048) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_3
  simp only [← c124941, ← c124942, k1, ← c124943] at e_2048_3
  have hr := e_2048_3
  simp only [e_2048_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20073 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 23) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597951)) := by
  have c124944 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124945 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124946 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124947 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124948 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124949 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_4 := arithEq_of_rows h (row := 2048) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_4
  simp only [← c124944, ← c124945, ← c124946] at e_2048_4
  have e_2048_5 := arithEq_of_rows h (row := 2048) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_5
  simp only [← c124947, ← c124948, k1, ← c124949] at e_2048_5
  have hr := e_2048_5
  simp only [e_2048_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20074 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 31) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597952)) := by
  have c124950 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124951 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124952 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124953 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124954 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124955 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_6 := arithEq_of_rows h (row := 2048) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_6
  simp only [← c124950, ← c124951, ← c124952] at e_2048_6
  have e_2048_7 := arithEq_of_rows h (row := 2048) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_7
  simp only [← c124953, ← c124954, k1, ← c124955] at e_2048_7
  have hr := e_2048_7
  simp only [e_2048_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20075 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 39) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597953)) := by
  have c124956 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124957 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124958 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124959 := (publicBatchWrapper89_copies3904 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c124960 := (publicBatchWrapper89_copies3905 a h).1
  have c124961 := (publicBatchWrapper89_copies3905 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_8 := arithEq_of_rows h (row := 2048) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_8
  simp only [← c124956, ← c124957, ← c124958] at e_2048_8
  have e_2048_9 := arithEq_of_rows h (row := 2048) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_9
  simp only [← c124959, ← c124960, k1, ← c124961] at e_2048_9
  have hr := e_2048_9
  simp only [e_2048_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20076 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 47) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597954)) := by
  have c124962 := (publicBatchWrapper89_copies3905 a h).2.2.1
  have c124963 := (publicBatchWrapper89_copies3905 a h).2.2.2.1
  have c124964 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.1
  have c124965 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.1
  have c124966 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.1
  have c124967 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_10 := arithEq_of_rows h (row := 2048) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_10
  simp only [← c124962, ← c124963, ← c124964] at e_2048_10
  have e_2048_11 := arithEq_of_rows h (row := 2048) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_11
  simp only [← c124965, ← c124966, k1, ← c124967] at e_2048_11
  have hr := e_2048_11
  simp only [e_2048_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20077 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 55) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597955)) := by
  have c124968 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.1
  have c124969 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.1
  have c124970 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.1
  have c124971 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c124972 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124973 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_12 := arithEq_of_rows h (row := 2048) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_12
  simp only [← c124968, ← c124969, ← c124970] at e_2048_12
  have e_2048_13 := arithEq_of_rows h (row := 2048) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_13
  simp only [← c124971, ← c124972, k1, ← c124973] at e_2048_13
  have hr := e_2048_13
  simp only [e_2048_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20078 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 63) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597956)) := by
  have c124974 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124975 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124976 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124977 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124978 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124979 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_14 := arithEq_of_rows h (row := 2048) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_14
  simp only [← c124974, ← c124975, ← c124976] at e_2048_14
  have e_2048_15 := arithEq_of_rows h (row := 2048) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_15
  simp only [← c124977, ← c124978, k1, ← c124979] at e_2048_15
  have hr := e_2048_15
  simp only [e_2048_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20079 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 71) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597957)) := by
  have c124980 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124981 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124982 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124983 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124984 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124985 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_16 := arithEq_of_rows h (row := 2048) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_16
  simp only [← c124980, ← c124981, ← c124982] at e_2048_16
  have e_2048_17 := arithEq_of_rows h (row := 2048) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_17
  simp only [← c124983, ← c124984, k1, ← c124985] at e_2048_17
  have hr := e_2048_17
  simp only [e_2048_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20080 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2048 79) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597958)) := by
  have c124986 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124987 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124988 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124989 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124990 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c124991 := (publicBatchWrapper89_copies3905 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2048_18 := arithEq_of_rows h (row := 2048) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_18
  simp only [← c124986, ← c124987, ← c124988] at e_2048_18
  have e_2048_19 := arithEq_of_rows h (row := 2048) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2048_19
  simp only [← c124989, ← c124990, k1, ← c124991] at e_2048_19
  have hr := e_2048_19
  simp only [e_2048_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20081 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 7) = bselect (a (.wire 83 31)) (a (.virt 604760)) (a (.virt 597959)) := by
  have c124992 := (publicBatchWrapper89_copies3906 a h).1
  have c124993 := (publicBatchWrapper89_copies3906 a h).2.1
  have c124994 := (publicBatchWrapper89_copies3906 a h).2.2.1
  have c124995 := (publicBatchWrapper89_copies3906 a h).2.2.2.1
  have c124996 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.1
  have c124997 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_0 := arithEq_of_rows h (row := 2049) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_0
  simp only [← c124992, ← c124993, ← c124994] at e_2049_0
  have e_2049_1 := arithEq_of_rows h (row := 2049) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_1
  simp only [← c124995, ← c124996, k1, ← c124997] at e_2049_1
  have hr := e_2049_1
  simp only [e_2049_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20082 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 15) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604723)) := by
  have c124998 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.1
  have c124999 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.1
  have c125000 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.1
  have c125001 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.1
  have c125002 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125003 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_2 := arithEq_of_rows h (row := 2049) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_2
  simp only [← c124998, ← c124999, ← c125000] at e_2049_2
  have e_2049_3 := arithEq_of_rows h (row := 2049) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_3
  simp only [← c125001, ← c125002, k1, ← c125003] at e_2049_3
  have hr := e_2049_3
  simp only [e_2049_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20083 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 23) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604724)) := by
  have c125004 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125005 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125006 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125007 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125008 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125009 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_4 := arithEq_of_rows h (row := 2049) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_4
  simp only [← c125004, ← c125005, ← c125006] at e_2049_4
  have e_2049_5 := arithEq_of_rows h (row := 2049) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_5
  simp only [← c125007, ← c125008, k1, ← c125009] at e_2049_5
  have hr := e_2049_5
  simp only [e_2049_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20084 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 31) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604725)) := by
  have c125010 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125011 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125012 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125013 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125014 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125015 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_6 := arithEq_of_rows h (row := 2049) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_6
  simp only [← c125010, ← c125011, ← c125012] at e_2049_6
  have e_2049_7 := arithEq_of_rows h (row := 2049) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_7
  simp only [← c125013, ← c125014, k1, ← c125015] at e_2049_7
  have hr := e_2049_7
  simp only [e_2049_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20085 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 39) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604726)) := by
  have c125016 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125017 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125018 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125019 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125020 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125021 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_8 := arithEq_of_rows h (row := 2049) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_8
  simp only [← c125016, ← c125017, ← c125018] at e_2049_8
  have e_2049_9 := arithEq_of_rows h (row := 2049) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_9
  simp only [← c125019, ← c125020, k1, ← c125021] at e_2049_9
  have hr := e_2049_9
  simp only [e_2049_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20086 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 47) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604727)) := by
  have c125022 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125023 := (publicBatchWrapper89_copies3906 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c125024 := (publicBatchWrapper89_copies3907 a h).1
  have c125025 := (publicBatchWrapper89_copies3907 a h).2.1
  have c125026 := (publicBatchWrapper89_copies3907 a h).2.2.1
  have c125027 := (publicBatchWrapper89_copies3907 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_10 := arithEq_of_rows h (row := 2049) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_10
  simp only [← c125022, ← c125023, ← c125024] at e_2049_10
  have e_2049_11 := arithEq_of_rows h (row := 2049) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_11
  simp only [← c125025, ← c125026, k1, ← c125027] at e_2049_11
  have hr := e_2049_11
  simp only [e_2049_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20087 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 55) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604728)) := by
  have c125028 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.1
  have c125029 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.1
  have c125030 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.1
  have c125031 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.1
  have c125032 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.1
  have c125033 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_12 := arithEq_of_rows h (row := 2049) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_12
  simp only [← c125028, ← c125029, ← c125030] at e_2049_12
  have e_2049_13 := arithEq_of_rows h (row := 2049) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_13
  simp only [← c125031, ← c125032, k1, ← c125033] at e_2049_13
  have hr := e_2049_13
  simp only [e_2049_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20088 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 63) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604729)) := by
  have c125034 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125035 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c125036 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125037 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125038 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125039 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_14 := arithEq_of_rows h (row := 2049) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_14
  simp only [← c125034, ← c125035, ← c125036] at e_2049_14
  have e_2049_15 := arithEq_of_rows h (row := 2049) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_15
  simp only [← c125037, ← c125038, k1, ← c125039] at e_2049_15
  have hr := e_2049_15
  simp only [e_2049_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20089 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 71) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604730)) := by
  have c125040 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125041 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125042 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125043 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125044 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125045 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_16 := arithEq_of_rows h (row := 2049) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_16
  simp only [← c125040, ← c125041, ← c125042] at e_2049_16
  have e_2049_17 := arithEq_of_rows h (row := 2049) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_17
  simp only [← c125043, ← c125044, k1, ← c125045] at e_2049_17
  have hr := e_2049_17
  simp only [e_2049_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20090 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2049 79) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604731)) := by
  have c125046 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125047 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125048 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125049 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125050 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125051 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2049_18 := arithEq_of_rows h (row := 2049) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_18
  simp only [← c125046, ← c125047, ← c125048] at e_2049_18
  have e_2049_19 := arithEq_of_rows h (row := 2049) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2049_19
  simp only [← c125049, ← c125050, k1, ← c125051] at e_2049_19
  have hr := e_2049_19
  simp only [e_2049_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20091 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 7) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604732)) := by
  have c125052 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125053 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125054 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125055 := (publicBatchWrapper89_copies3907 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c125056 := (publicBatchWrapper89_copies3908 a h).1
  have c125057 := (publicBatchWrapper89_copies3908 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_0 := arithEq_of_rows h (row := 2050) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_0
  simp only [← c125052, ← c125053, ← c125054] at e_2050_0
  have e_2050_1 := arithEq_of_rows h (row := 2050) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_1
  simp only [← c125055, ← c125056, k1, ← c125057] at e_2050_1
  have hr := e_2050_1
  simp only [e_2050_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20092 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 15) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604733)) := by
  have c125058 := (publicBatchWrapper89_copies3908 a h).2.2.1
  have c125059 := (publicBatchWrapper89_copies3908 a h).2.2.2.1
  have c125060 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.1
  have c125061 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.1
  have c125062 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.1
  have c125063 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_2 := arithEq_of_rows h (row := 2050) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_2
  simp only [← c125058, ← c125059, ← c125060] at e_2050_2
  have e_2050_3 := arithEq_of_rows h (row := 2050) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_3
  simp only [← c125061, ← c125062, k1, ← c125063] at e_2050_3
  have hr := e_2050_3
  simp only [e_2050_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20093 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 23) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604734)) := by
  have c125064 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.1
  have c125065 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.1
  have c125066 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125067 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c125068 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125069 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_4 := arithEq_of_rows h (row := 2050) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_4
  simp only [← c125064, ← c125065, ← c125066] at e_2050_4
  have e_2050_5 := arithEq_of_rows h (row := 2050) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_5
  simp only [← c125067, ← c125068, k1, ← c125069] at e_2050_5
  have hr := e_2050_5
  simp only [e_2050_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20094 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 31) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604735)) := by
  have c125070 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125071 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125072 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125073 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125074 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125075 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_6 := arithEq_of_rows h (row := 2050) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_6
  simp only [← c125070, ← c125071, ← c125072] at e_2050_6
  have e_2050_7 := arithEq_of_rows h (row := 2050) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_7
  simp only [← c125073, ← c125074, k1, ← c125075] at e_2050_7
  have hr := e_2050_7
  simp only [e_2050_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20095 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 39) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604736)) := by
  have c125076 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125077 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125078 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125079 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125080 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125081 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_8 := arithEq_of_rows h (row := 2050) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_8
  simp only [← c125076, ← c125077, ← c125078] at e_2050_8
  have e_2050_9 := arithEq_of_rows h (row := 2050) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_9
  simp only [← c125079, ← c125080, k1, ← c125081] at e_2050_9
  have hr := e_2050_9
  simp only [e_2050_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20096 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 47) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604737)) := by
  have c125082 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125083 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125084 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125085 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125086 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125087 := (publicBatchWrapper89_copies3908 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_10 := arithEq_of_rows h (row := 2050) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_10
  simp only [← c125082, ← c125083, ← c125084] at e_2050_10
  have e_2050_11 := arithEq_of_rows h (row := 2050) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_11
  simp only [← c125085, ← c125086, k1, ← c125087] at e_2050_11
  have hr := e_2050_11
  simp only [e_2050_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20097 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 55) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604738)) := by
  have c125088 := (publicBatchWrapper89_copies3909 a h).1
  have c125089 := (publicBatchWrapper89_copies3909 a h).2.1
  have c125090 := (publicBatchWrapper89_copies3909 a h).2.2.1
  have c125091 := (publicBatchWrapper89_copies3909 a h).2.2.2.1
  have c125092 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.1
  have c125093 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_12 := arithEq_of_rows h (row := 2050) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_12
  simp only [← c125088, ← c125089, ← c125090] at e_2050_12
  have e_2050_13 := arithEq_of_rows h (row := 2050) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_13
  simp only [← c125091, ← c125092, k1, ← c125093] at e_2050_13
  have hr := e_2050_13
  simp only [e_2050_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20098 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 63) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604739)) := by
  have c125094 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.1
  have c125095 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.1
  have c125096 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.1
  have c125097 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.1
  have c125098 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125099 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_14 := arithEq_of_rows h (row := 2050) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_14
  simp only [← c125094, ← c125095, ← c125096] at e_2050_14
  have e_2050_15 := arithEq_of_rows h (row := 2050) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_15
  simp only [← c125097, ← c125098, k1, ← c125099] at e_2050_15
  have hr := e_2050_15
  simp only [e_2050_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20099 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 71) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604740)) := by
  have c125100 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125101 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125102 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125103 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125104 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125105 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_16 := arithEq_of_rows h (row := 2050) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_16
  simp only [← c125100, ← c125101, ← c125102] at e_2050_16
  have e_2050_17 := arithEq_of_rows h (row := 2050) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_17
  simp only [← c125103, ← c125104, k1, ← c125105] at e_2050_17
  have hr := e_2050_17
  simp only [e_2050_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20100 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2050 79) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604741)) := by
  have c125106 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125107 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125108 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125109 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125110 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125111 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2050_18 := arithEq_of_rows h (row := 2050) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_18
  simp only [← c125106, ← c125107, ← c125108] at e_2050_18
  have e_2050_19 := arithEq_of_rows h (row := 2050) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2050_19
  simp only [← c125109, ← c125110, k1, ← c125111] at e_2050_19
  have hr := e_2050_19
  simp only [e_2050_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20101 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 7) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604742)) := by
  have c125112 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125113 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125114 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125115 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125116 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125117 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_0 := arithEq_of_rows h (row := 2051) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_0
  simp only [← c125112, ← c125113, ← c125114] at e_2051_0
  have e_2051_1 := arithEq_of_rows h (row := 2051) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_1
  simp only [← c125115, ← c125116, k1, ← c125117] at e_2051_1
  have hr := e_2051_1
  simp only [e_2051_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20102 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 15) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604743)) := by
  have c125118 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125119 := (publicBatchWrapper89_copies3909 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c125120 := (publicBatchWrapper89_copies3910 a h).1
  have c125121 := (publicBatchWrapper89_copies3910 a h).2.1
  have c125122 := (publicBatchWrapper89_copies3910 a h).2.2.1
  have c125123 := (publicBatchWrapper89_copies3910 a h).2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_2 := arithEq_of_rows h (row := 2051) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_2
  simp only [← c125118, ← c125119, ← c125120] at e_2051_2
  have e_2051_3 := arithEq_of_rows h (row := 2051) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_3
  simp only [← c125121, ← c125122, k1, ← c125123] at e_2051_3
  have hr := e_2051_3
  simp only [e_2051_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20103 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 23) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604744)) := by
  have c125124 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.1
  have c125125 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.1
  have c125126 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.1
  have c125127 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.1
  have c125128 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.1
  have c125129 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_4 := arithEq_of_rows h (row := 2051) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_4
  simp only [← c125124, ← c125125, ← c125126] at e_2051_4
  have e_2051_5 := arithEq_of_rows h (row := 2051) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_5
  simp only [← c125127, ← c125128, k1, ← c125129] at e_2051_5
  have hr := e_2051_5
  simp only [e_2051_4] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20104 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 31) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604745)) := by
  have c125130 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125131 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c125132 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125133 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125134 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125135 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_6 := arithEq_of_rows h (row := 2051) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_6
  simp only [← c125130, ← c125131, ← c125132] at e_2051_6
  have e_2051_7 := arithEq_of_rows h (row := 2051) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_7
  simp only [← c125133, ← c125134, k1, ← c125135] at e_2051_7
  have hr := e_2051_7
  simp only [e_2051_6] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20105 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 39) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604746)) := by
  have c125136 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125137 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125138 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125139 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125140 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125141 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_8 := arithEq_of_rows h (row := 2051) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_8
  simp only [← c125136, ← c125137, ← c125138] at e_2051_8
  have e_2051_9 := arithEq_of_rows h (row := 2051) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_9
  simp only [← c125139, ← c125140, k1, ← c125141] at e_2051_9
  have hr := e_2051_9
  simp only [e_2051_8] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20106 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 47) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604747)) := by
  have c125142 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125143 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125144 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125145 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125146 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125147 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_10 := arithEq_of_rows h (row := 2051) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_10
  simp only [← c125142, ← c125143, ← c125144] at e_2051_10
  have e_2051_11 := arithEq_of_rows h (row := 2051) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_11
  simp only [← c125145, ← c125146, k1, ← c125147] at e_2051_11
  have hr := e_2051_11
  simp only [e_2051_10] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20107 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 55) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604748)) := by
  have c125148 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125149 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125150 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125151 := (publicBatchWrapper89_copies3910 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c125152 := (publicBatchWrapper89_copies3911 a h).1
  have c125153 := (publicBatchWrapper89_copies3911 a h).2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_12 := arithEq_of_rows h (row := 2051) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_12
  simp only [← c125148, ← c125149, ← c125150] at e_2051_12
  have e_2051_13 := arithEq_of_rows h (row := 2051) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_13
  simp only [← c125151, ← c125152, k1, ← c125153] at e_2051_13
  have hr := e_2051_13
  simp only [e_2051_12] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20108 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 63) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604749)) := by
  have c125154 := (publicBatchWrapper89_copies3911 a h).2.2.1
  have c125155 := (publicBatchWrapper89_copies3911 a h).2.2.2.1
  have c125156 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.1
  have c125157 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.1
  have c125158 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.1
  have c125159 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_14 := arithEq_of_rows h (row := 2051) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_14
  simp only [← c125154, ← c125155, ← c125156] at e_2051_14
  have e_2051_15 := arithEq_of_rows h (row := 2051) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_15
  simp only [← c125157, ← c125158, k1, ← c125159] at e_2051_15
  have hr := e_2051_15
  simp only [e_2051_14] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20109 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 71) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604750)) := by
  have c125160 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.1
  have c125161 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.1
  have c125162 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.1
  have c125163 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c125164 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125165 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_16 := arithEq_of_rows h (row := 2051) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_16
  simp only [← c125160, ← c125161, ← c125162] at e_2051_16
  have e_2051_17 := arithEq_of_rows h (row := 2051) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_17
  simp only [← c125163, ← c125164, k1, ← c125165] at e_2051_17
  have hr := e_2051_17
  simp only [e_2051_16] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20110 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2051 79) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604751)) := by
  have c125166 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125167 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125168 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125169 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125170 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125171 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2051_18 := arithEq_of_rows h (row := 2051) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_18
  simp only [← c125166, ← c125167, ← c125168] at e_2051_18
  have e_2051_19 := arithEq_of_rows h (row := 2051) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2051_19
  simp only [← c125169, ← c125170, k1, ← c125171] at e_2051_19
  have hr := e_2051_19
  simp only [e_2051_18] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20111 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 7) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604752)) := by
  have c125172 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125173 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125174 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125175 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125176 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125177 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_0 := arithEq_of_rows h (row := 2052) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_0
  simp only [← c125172, ← c125173, ← c125174] at e_2052_0
  have e_2052_1 := arithEq_of_rows h (row := 2052) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_1
  simp only [← c125175, ← c125176, k1, ← c125177] at e_2052_1
  have hr := e_2052_1
  simp only [e_2052_0] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20112 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 15) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604753)) := by
  have c125178 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125179 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125180 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125181 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125182 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c125183 := (publicBatchWrapper89_copies3911 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_2 := arithEq_of_rows h (row := 2052) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_2
  simp only [← c125178, ← c125179, ← c125180] at e_2052_2
  have e_2052_3 := arithEq_of_rows h (row := 2052) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_3
  simp only [← c125181, ← c125182, k1, ← c125183] at e_2052_3
  have hr := e_2052_3
  simp only [e_2052_2] at hr
  simp only [bselect, k1]
  linear_combination hr

theorem publicBatchWrapper89_f20113 (a : Assignment p) (h : Satisfies (publicBatchWrapper89 p) a) :
    a (.wire 2052 23) = bselect (a (.wire 83 75)) (a (.virt 604760)) (a (.virt 604754)) := by
  have c125184 := (publicBatchWrapper89_copies3912 a h).1
  have c125185 := (publicBatchWrapper89_copies3912 a h).2.1
  have c125186 := (publicBatchWrapper89_copies3912 a h).2.2.1
  have c125187 := (publicBatchWrapper89_copies3912 a h).2.2.2.1
  have c125188 := (publicBatchWrapper89_copies3912 a h).2.2.2.2.1
  have c125189 := (publicBatchWrapper89_copies3912 a h).2.2.2.2.2
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper89_consts a h
  have e_2052_4 := arithEq_of_rows h (row := 2052) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_4
  simp only [← c125184, ← c125185, ← c125186] at e_2052_4
  have e_2052_5 := arithEq_of_rows h (row := 2052) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_2052_5
  simp only [← c125187, ← c125188, k1, ← c125189] at e_2052_5
  have hr := e_2052_5
  simp only [e_2052_4] at hr
  simp only [bselect, k1]
  linear_combination hr

end Plonky2Spec.Generated
