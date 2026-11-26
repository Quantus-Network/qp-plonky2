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

theorem publicBatchWrapper86_f13885 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1429 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13691)) := by
  have c87663 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87664 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87665 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87666 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87667 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87668 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1429_6 := arithEq_of_rows h (row := 1429) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_6
  simp only [← c87663, ← c87664, ← c87665] at e_1429_6
  have e_1429_7 := arithEq_of_rows h (row := 1429) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_7
  simp only [← c87666, ← c87667, k2, ← c87668] at e_1429_7
  have hr := e_1429_7
  simp only [e_1429_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13886 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1429 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13692)) := by
  have c87669 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87670 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87671 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87672 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87673 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87674 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1429_8 := arithEq_of_rows h (row := 1429) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_8
  simp only [← c87669, ← c87670, ← c87671] at e_1429_8
  have e_1429_9 := arithEq_of_rows h (row := 1429) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_9
  simp only [← c87672, ← c87673, k2, ← c87674] at e_1429_9
  have hr := e_1429_9
  simp only [e_1429_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13887 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1429 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13693)) := by
  have c87675 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87676 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87677 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87678 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87679 := (publicBatchWrapper86_copies2739 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87680 := (publicBatchWrapper86_copies2740 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1429_10 := arithEq_of_rows h (row := 1429) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_10
  simp only [← c87675, ← c87676, ← c87677] at e_1429_10
  have e_1429_11 := arithEq_of_rows h (row := 1429) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_11
  simp only [← c87678, ← c87679, k2, ← c87680] at e_1429_11
  have hr := e_1429_11
  simp only [e_1429_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13888 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1429 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13694)) := by
  have c87681 := (publicBatchWrapper86_copies2740 a h).2.1
  have c87682 := (publicBatchWrapper86_copies2740 a h).2.2.1
  have c87683 := (publicBatchWrapper86_copies2740 a h).2.2.2.1
  have c87684 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.1
  have c87685 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.1
  have c87686 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1429_12 := arithEq_of_rows h (row := 1429) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_12
  simp only [← c87681, ← c87682, ← c87683] at e_1429_12
  have e_1429_13 := arithEq_of_rows h (row := 1429) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_13
  simp only [← c87684, ← c87685, k2, ← c87686] at e_1429_13
  have hr := e_1429_13
  simp only [e_1429_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13889 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1429 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13695)) := by
  have c87687 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.1
  have c87688 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.1
  have c87689 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.1
  have c87690 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87691 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87692 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1429_14 := arithEq_of_rows h (row := 1429) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_14
  simp only [← c87687, ← c87688, ← c87689] at e_1429_14
  have e_1429_15 := arithEq_of_rows h (row := 1429) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_15
  simp only [← c87690, ← c87691, k2, ← c87692] at e_1429_15
  have hr := e_1429_15
  simp only [e_1429_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13890 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1429 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13696)) := by
  have c87693 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87694 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87695 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87696 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87697 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87698 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1429_16 := arithEq_of_rows h (row := 1429) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_16
  simp only [← c87693, ← c87694, ← c87695] at e_1429_16
  have e_1429_17 := arithEq_of_rows h (row := 1429) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_17
  simp only [← c87696, ← c87697, k2, ← c87698] at e_1429_17
  have hr := e_1429_17
  simp only [e_1429_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13891 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1429 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13697)) := by
  have c87699 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87700 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87701 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87702 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87703 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87704 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1429_18 := arithEq_of_rows h (row := 1429) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_18
  simp only [← c87699, ← c87700, ← c87701] at e_1429_18
  have e_1429_19 := arithEq_of_rows h (row := 1429) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1429_19
  simp only [← c87702, ← c87703, k2, ← c87704] at e_1429_19
  have hr := e_1429_19
  simp only [e_1429_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13892 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1430 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13698)) := by
  have c87705 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87706 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87707 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87708 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87709 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87710 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1430_0 := arithEq_of_rows h (row := 1430) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_0
  simp only [← c87705, ← c87706, ← c87707] at e_1430_0
  have e_1430_1 := arithEq_of_rows h (row := 1430) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_1
  simp only [← c87708, ← c87709, k2, ← c87710] at e_1430_1
  have hr := e_1430_1
  simp only [e_1430_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13893 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1430 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13699)) := by
  have c87711 := (publicBatchWrapper86_copies2740 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87712 := (publicBatchWrapper86_copies2741 a h).1
  have c87713 := (publicBatchWrapper86_copies2741 a h).2.1
  have c87714 := (publicBatchWrapper86_copies2741 a h).2.2.1
  have c87715 := (publicBatchWrapper86_copies2741 a h).2.2.2.1
  have c87716 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1430_2 := arithEq_of_rows h (row := 1430) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_2
  simp only [← c87711, ← c87712, ← c87713] at e_1430_2
  have e_1430_3 := arithEq_of_rows h (row := 1430) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_3
  simp only [← c87714, ← c87715, k2, ← c87716] at e_1430_3
  have hr := e_1430_3
  simp only [e_1430_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13894 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1430 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13700)) := by
  have c87717 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.1
  have c87718 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.1
  have c87719 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.1
  have c87720 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.1
  have c87721 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.1
  have c87722 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1430_4 := arithEq_of_rows h (row := 1430) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_4
  simp only [← c87717, ← c87718, ← c87719] at e_1430_4
  have e_1430_5 := arithEq_of_rows h (row := 1430) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_5
  simp only [← c87720, ← c87721, k2, ← c87722] at e_1430_5
  have hr := e_1430_5
  simp only [e_1430_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13895 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1430 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13701)) := by
  have c87723 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87724 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87725 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87726 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87727 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87728 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1430_6 := arithEq_of_rows h (row := 1430) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_6
  simp only [← c87723, ← c87724, ← c87725] at e_1430_6
  have e_1430_7 := arithEq_of_rows h (row := 1430) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_7
  simp only [← c87726, ← c87727, k2, ← c87728] at e_1430_7
  have hr := e_1430_7
  simp only [e_1430_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13896 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1430 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13702)) := by
  have c87729 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87730 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87731 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87732 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87733 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87734 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1430_8 := arithEq_of_rows h (row := 1430) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_8
  simp only [← c87729, ← c87730, ← c87731] at e_1430_8
  have e_1430_9 := arithEq_of_rows h (row := 1430) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_9
  simp only [← c87732, ← c87733, k2, ← c87734] at e_1430_9
  have hr := e_1430_9
  simp only [e_1430_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13897 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1430 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13703)) := by
  have c87735 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87736 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87737 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87738 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87739 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87740 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1430_10 := arithEq_of_rows h (row := 1430) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_10
  simp only [← c87735, ← c87736, ← c87737] at e_1430_10
  have e_1430_11 := arithEq_of_rows h (row := 1430) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_11
  simp only [← c87738, ← c87739, k2, ← c87740] at e_1430_11
  have hr := e_1430_11
  simp only [e_1430_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13898 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1430 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13704)) := by
  have c87741 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87742 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87743 := (publicBatchWrapper86_copies2741 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87744 := (publicBatchWrapper86_copies2742 a h).1
  have c87745 := (publicBatchWrapper86_copies2742 a h).2.1
  have c87746 := (publicBatchWrapper86_copies2742 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1430_12 := arithEq_of_rows h (row := 1430) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_12
  simp only [← c87741, ← c87742, ← c87743] at e_1430_12
  have e_1430_13 := arithEq_of_rows h (row := 1430) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_13
  simp only [← c87744, ← c87745, k2, ← c87746] at e_1430_13
  have hr := e_1430_13
  simp only [e_1430_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13899 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1430 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13705)) := by
  have c87747 := (publicBatchWrapper86_copies2742 a h).2.2.2.1
  have c87748 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.1
  have c87749 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.1
  have c87750 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.1
  have c87751 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.1
  have c87752 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1430_14 := arithEq_of_rows h (row := 1430) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_14
  simp only [← c87747, ← c87748, ← c87749] at e_1430_14
  have e_1430_15 := arithEq_of_rows h (row := 1430) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_15
  simp only [← c87750, ← c87751, k2, ← c87752] at e_1430_15
  have hr := e_1430_15
  simp only [e_1430_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13900 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1430 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13706)) := by
  have c87753 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.1
  have c87754 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87755 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87756 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87757 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87758 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1430_16 := arithEq_of_rows h (row := 1430) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_16
  simp only [← c87753, ← c87754, ← c87755] at e_1430_16
  have e_1430_17 := arithEq_of_rows h (row := 1430) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_17
  simp only [← c87756, ← c87757, k2, ← c87758] at e_1430_17
  have hr := e_1430_17
  simp only [e_1430_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13901 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1430 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13707)) := by
  have c87759 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87760 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87761 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87762 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87763 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87764 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1430_18 := arithEq_of_rows h (row := 1430) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_18
  simp only [← c87759, ← c87760, ← c87761] at e_1430_18
  have e_1430_19 := arithEq_of_rows h (row := 1430) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1430_19
  simp only [← c87762, ← c87763, k2, ← c87764] at e_1430_19
  have hr := e_1430_19
  simp only [e_1430_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13902 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1431 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13708)) := by
  have c87765 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87766 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87767 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87768 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87769 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87770 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1431_0 := arithEq_of_rows h (row := 1431) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_0
  simp only [← c87765, ← c87766, ← c87767] at e_1431_0
  have e_1431_1 := arithEq_of_rows h (row := 1431) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_1
  simp only [← c87768, ← c87769, k2, ← c87770] at e_1431_1
  have hr := e_1431_1
  simp only [e_1431_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13903 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1431 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13709)) := by
  have c87771 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87772 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87773 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87774 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87775 := (publicBatchWrapper86_copies2742 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87776 := (publicBatchWrapper86_copies2743 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1431_2 := arithEq_of_rows h (row := 1431) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_2
  simp only [← c87771, ← c87772, ← c87773] at e_1431_2
  have e_1431_3 := arithEq_of_rows h (row := 1431) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_3
  simp only [← c87774, ← c87775, k2, ← c87776] at e_1431_3
  have hr := e_1431_3
  simp only [e_1431_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13904 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1431 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13710)) := by
  have c87777 := (publicBatchWrapper86_copies2743 a h).2.1
  have c87778 := (publicBatchWrapper86_copies2743 a h).2.2.1
  have c87779 := (publicBatchWrapper86_copies2743 a h).2.2.2.1
  have c87780 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.1
  have c87781 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.1
  have c87782 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1431_4 := arithEq_of_rows h (row := 1431) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_4
  simp only [← c87777, ← c87778, ← c87779] at e_1431_4
  have e_1431_5 := arithEq_of_rows h (row := 1431) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_5
  simp only [← c87780, ← c87781, k2, ← c87782] at e_1431_5
  have hr := e_1431_5
  simp only [e_1431_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13905 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1431 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13711)) := by
  have c87783 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.1
  have c87784 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.1
  have c87785 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.1
  have c87786 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87787 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87788 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1431_6 := arithEq_of_rows h (row := 1431) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_6
  simp only [← c87783, ← c87784, ← c87785] at e_1431_6
  have e_1431_7 := arithEq_of_rows h (row := 1431) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_7
  simp only [← c87786, ← c87787, k2, ← c87788] at e_1431_7
  have hr := e_1431_7
  simp only [e_1431_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13906 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1431 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13712)) := by
  have c87789 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87790 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87791 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87792 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87793 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87794 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1431_8 := arithEq_of_rows h (row := 1431) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_8
  simp only [← c87789, ← c87790, ← c87791] at e_1431_8
  have e_1431_9 := arithEq_of_rows h (row := 1431) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_9
  simp only [← c87792, ← c87793, k2, ← c87794] at e_1431_9
  have hr := e_1431_9
  simp only [e_1431_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13907 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1431 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13713)) := by
  have c87795 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87796 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87797 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87798 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87799 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87800 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1431_10 := arithEq_of_rows h (row := 1431) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_10
  simp only [← c87795, ← c87796, ← c87797] at e_1431_10
  have e_1431_11 := arithEq_of_rows h (row := 1431) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_11
  simp only [← c87798, ← c87799, k2, ← c87800] at e_1431_11
  have hr := e_1431_11
  simp only [e_1431_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13908 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1431 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13714)) := by
  have c87801 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87802 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87803 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87804 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87805 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87806 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1431_12 := arithEq_of_rows h (row := 1431) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_12
  simp only [← c87801, ← c87802, ← c87803] at e_1431_12
  have e_1431_13 := arithEq_of_rows h (row := 1431) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_13
  simp only [← c87804, ← c87805, k2, ← c87806] at e_1431_13
  have hr := e_1431_13
  simp only [e_1431_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13909 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1431 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13715)) := by
  have c87807 := (publicBatchWrapper86_copies2743 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87808 := (publicBatchWrapper86_copies2744 a h).1
  have c87809 := (publicBatchWrapper86_copies2744 a h).2.1
  have c87810 := (publicBatchWrapper86_copies2744 a h).2.2.1
  have c87811 := (publicBatchWrapper86_copies2744 a h).2.2.2.1
  have c87812 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1431_14 := arithEq_of_rows h (row := 1431) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_14
  simp only [← c87807, ← c87808, ← c87809] at e_1431_14
  have e_1431_15 := arithEq_of_rows h (row := 1431) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_15
  simp only [← c87810, ← c87811, k2, ← c87812] at e_1431_15
  have hr := e_1431_15
  simp only [e_1431_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13910 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1431 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13716)) := by
  have c87813 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.1
  have c87814 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.1
  have c87815 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.1
  have c87816 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.1
  have c87817 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.1
  have c87818 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1431_16 := arithEq_of_rows h (row := 1431) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_16
  simp only [← c87813, ← c87814, ← c87815] at e_1431_16
  have e_1431_17 := arithEq_of_rows h (row := 1431) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_17
  simp only [← c87816, ← c87817, k2, ← c87818] at e_1431_17
  have hr := e_1431_17
  simp only [e_1431_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13911 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1431 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13717)) := by
  have c87819 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87820 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87821 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87822 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87823 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87824 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1431_18 := arithEq_of_rows h (row := 1431) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_18
  simp only [← c87819, ← c87820, ← c87821] at e_1431_18
  have e_1431_19 := arithEq_of_rows h (row := 1431) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1431_19
  simp only [← c87822, ← c87823, k2, ← c87824] at e_1431_19
  have hr := e_1431_19
  simp only [e_1431_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13912 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1432 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13718)) := by
  have c87825 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87826 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87827 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87828 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87829 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87830 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1432_0 := arithEq_of_rows h (row := 1432) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_0
  simp only [← c87825, ← c87826, ← c87827] at e_1432_0
  have e_1432_1 := arithEq_of_rows h (row := 1432) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_1
  simp only [← c87828, ← c87829, k2, ← c87830] at e_1432_1
  have hr := e_1432_1
  simp only [e_1432_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13913 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1432 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13719)) := by
  have c87831 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87832 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87833 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87834 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87835 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87836 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1432_2 := arithEq_of_rows h (row := 1432) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_2
  simp only [← c87831, ← c87832, ← c87833] at e_1432_2
  have e_1432_3 := arithEq_of_rows h (row := 1432) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_3
  simp only [← c87834, ← c87835, k2, ← c87836] at e_1432_3
  have hr := e_1432_3
  simp only [e_1432_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13914 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1432 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13720)) := by
  have c87837 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87838 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87839 := (publicBatchWrapper86_copies2744 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87840 := (publicBatchWrapper86_copies2745 a h).1
  have c87841 := (publicBatchWrapper86_copies2745 a h).2.1
  have c87842 := (publicBatchWrapper86_copies2745 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1432_4 := arithEq_of_rows h (row := 1432) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_4
  simp only [← c87837, ← c87838, ← c87839] at e_1432_4
  have e_1432_5 := arithEq_of_rows h (row := 1432) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_5
  simp only [← c87840, ← c87841, k2, ← c87842] at e_1432_5
  have hr := e_1432_5
  simp only [e_1432_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13915 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1432 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13721)) := by
  have c87843 := (publicBatchWrapper86_copies2745 a h).2.2.2.1
  have c87844 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.1
  have c87845 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.1
  have c87846 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.1
  have c87847 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.1
  have c87848 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1432_6 := arithEq_of_rows h (row := 1432) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_6
  simp only [← c87843, ← c87844, ← c87845] at e_1432_6
  have e_1432_7 := arithEq_of_rows h (row := 1432) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_7
  simp only [← c87846, ← c87847, k2, ← c87848] at e_1432_7
  have hr := e_1432_7
  simp only [e_1432_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13916 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1432 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13722)) := by
  have c87849 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.1
  have c87850 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87851 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87852 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87853 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87854 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1432_8 := arithEq_of_rows h (row := 1432) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_8
  simp only [← c87849, ← c87850, ← c87851] at e_1432_8
  have e_1432_9 := arithEq_of_rows h (row := 1432) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_9
  simp only [← c87852, ← c87853, k2, ← c87854] at e_1432_9
  have hr := e_1432_9
  simp only [e_1432_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13917 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1432 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13723)) := by
  have c87855 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87856 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87857 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87858 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87859 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87860 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1432_10 := arithEq_of_rows h (row := 1432) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_10
  simp only [← c87855, ← c87856, ← c87857] at e_1432_10
  have e_1432_11 := arithEq_of_rows h (row := 1432) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_11
  simp only [← c87858, ← c87859, k2, ← c87860] at e_1432_11
  have hr := e_1432_11
  simp only [e_1432_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13918 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1432 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13724)) := by
  have c87861 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87862 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87863 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87864 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87865 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87866 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1432_12 := arithEq_of_rows h (row := 1432) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_12
  simp only [← c87861, ← c87862, ← c87863] at e_1432_12
  have e_1432_13 := arithEq_of_rows h (row := 1432) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_13
  simp only [← c87864, ← c87865, k2, ← c87866] at e_1432_13
  have hr := e_1432_13
  simp only [e_1432_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13919 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1432 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13725)) := by
  have c87867 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87868 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87869 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87870 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87871 := (publicBatchWrapper86_copies2745 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87872 := (publicBatchWrapper86_copies2746 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1432_14 := arithEq_of_rows h (row := 1432) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_14
  simp only [← c87867, ← c87868, ← c87869] at e_1432_14
  have e_1432_15 := arithEq_of_rows h (row := 1432) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_15
  simp only [← c87870, ← c87871, k2, ← c87872] at e_1432_15
  have hr := e_1432_15
  simp only [e_1432_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13920 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1432 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13726)) := by
  have c87873 := (publicBatchWrapper86_copies2746 a h).2.1
  have c87874 := (publicBatchWrapper86_copies2746 a h).2.2.1
  have c87875 := (publicBatchWrapper86_copies2746 a h).2.2.2.1
  have c87876 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.1
  have c87877 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.1
  have c87878 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1432_16 := arithEq_of_rows h (row := 1432) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_16
  simp only [← c87873, ← c87874, ← c87875] at e_1432_16
  have e_1432_17 := arithEq_of_rows h (row := 1432) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_17
  simp only [← c87876, ← c87877, k2, ← c87878] at e_1432_17
  have hr := e_1432_17
  simp only [e_1432_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13921 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1432 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13727)) := by
  have c87879 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.1
  have c87880 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.1
  have c87881 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.1
  have c87882 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87883 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87884 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1432_18 := arithEq_of_rows h (row := 1432) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_18
  simp only [← c87879, ← c87880, ← c87881] at e_1432_18
  have e_1432_19 := arithEq_of_rows h (row := 1432) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1432_19
  simp only [← c87882, ← c87883, k2, ← c87884] at e_1432_19
  have hr := e_1432_19
  simp only [e_1432_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13922 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1433 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13728)) := by
  have c87885 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87886 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87887 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87888 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87889 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87890 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1433_0 := arithEq_of_rows h (row := 1433) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_0
  simp only [← c87885, ← c87886, ← c87887] at e_1433_0
  have e_1433_1 := arithEq_of_rows h (row := 1433) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_1
  simp only [← c87888, ← c87889, k2, ← c87890] at e_1433_1
  have hr := e_1433_1
  simp only [e_1433_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13923 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1433 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13729)) := by
  have c87891 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87892 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87893 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87894 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87895 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87896 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1433_2 := arithEq_of_rows h (row := 1433) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_2
  simp only [← c87891, ← c87892, ← c87893] at e_1433_2
  have e_1433_3 := arithEq_of_rows h (row := 1433) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_3
  simp only [← c87894, ← c87895, k2, ← c87896] at e_1433_3
  have hr := e_1433_3
  simp only [e_1433_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13924 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1433 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13730)) := by
  have c87897 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87898 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87899 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87900 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87901 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87902 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1433_4 := arithEq_of_rows h (row := 1433) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_4
  simp only [← c87897, ← c87898, ← c87899] at e_1433_4
  have e_1433_5 := arithEq_of_rows h (row := 1433) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_5
  simp only [← c87900, ← c87901, k2, ← c87902] at e_1433_5
  have hr := e_1433_5
  simp only [e_1433_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13925 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1433 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13731)) := by
  have c87903 := (publicBatchWrapper86_copies2746 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87904 := (publicBatchWrapper86_copies2747 a h).1
  have c87905 := (publicBatchWrapper86_copies2747 a h).2.1
  have c87906 := (publicBatchWrapper86_copies2747 a h).2.2.1
  have c87907 := (publicBatchWrapper86_copies2747 a h).2.2.2.1
  have c87908 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1433_6 := arithEq_of_rows h (row := 1433) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_6
  simp only [← c87903, ← c87904, ← c87905] at e_1433_6
  have e_1433_7 := arithEq_of_rows h (row := 1433) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_7
  simp only [← c87906, ← c87907, k2, ← c87908] at e_1433_7
  have hr := e_1433_7
  simp only [e_1433_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13926 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1433 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13732)) := by
  have c87909 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.1
  have c87910 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.1
  have c87911 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.1
  have c87912 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.1
  have c87913 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.1
  have c87914 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1433_8 := arithEq_of_rows h (row := 1433) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_8
  simp only [← c87909, ← c87910, ← c87911] at e_1433_8
  have e_1433_9 := arithEq_of_rows h (row := 1433) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_9
  simp only [← c87912, ← c87913, k2, ← c87914] at e_1433_9
  have hr := e_1433_9
  simp only [e_1433_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13927 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1433 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13733)) := by
  have c87915 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87916 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87917 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87918 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87919 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87920 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1433_10 := arithEq_of_rows h (row := 1433) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_10
  simp only [← c87915, ← c87916, ← c87917] at e_1433_10
  have e_1433_11 := arithEq_of_rows h (row := 1433) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_11
  simp only [← c87918, ← c87919, k2, ← c87920] at e_1433_11
  have hr := e_1433_11
  simp only [e_1433_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13928 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1433 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13734)) := by
  have c87921 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87922 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87923 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87924 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87925 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87926 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1433_12 := arithEq_of_rows h (row := 1433) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_12
  simp only [← c87921, ← c87922, ← c87923] at e_1433_12
  have e_1433_13 := arithEq_of_rows h (row := 1433) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_13
  simp only [← c87924, ← c87925, k2, ← c87926] at e_1433_13
  have hr := e_1433_13
  simp only [e_1433_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13929 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1433 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13735)) := by
  have c87927 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87928 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87929 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87930 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87931 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87932 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1433_14 := arithEq_of_rows h (row := 1433) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_14
  simp only [← c87927, ← c87928, ← c87929] at e_1433_14
  have e_1433_15 := arithEq_of_rows h (row := 1433) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_15
  simp only [← c87930, ← c87931, k2, ← c87932] at e_1433_15
  have hr := e_1433_15
  simp only [e_1433_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13930 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1433 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13736)) := by
  have c87933 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87934 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87935 := (publicBatchWrapper86_copies2747 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87936 := (publicBatchWrapper86_copies2748 a h).1
  have c87937 := (publicBatchWrapper86_copies2748 a h).2.1
  have c87938 := (publicBatchWrapper86_copies2748 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1433_16 := arithEq_of_rows h (row := 1433) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_16
  simp only [← c87933, ← c87934, ← c87935] at e_1433_16
  have e_1433_17 := arithEq_of_rows h (row := 1433) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_17
  simp only [← c87936, ← c87937, k2, ← c87938] at e_1433_17
  have hr := e_1433_17
  simp only [e_1433_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13931 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1433 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13737)) := by
  have c87939 := (publicBatchWrapper86_copies2748 a h).2.2.2.1
  have c87940 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.1
  have c87941 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.1
  have c87942 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.1
  have c87943 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.1
  have c87944 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1433_18 := arithEq_of_rows h (row := 1433) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_18
  simp only [← c87939, ← c87940, ← c87941] at e_1433_18
  have e_1433_19 := arithEq_of_rows h (row := 1433) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1433_19
  simp only [← c87942, ← c87943, k2, ← c87944] at e_1433_19
  have hr := e_1433_19
  simp only [e_1433_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13932 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1434 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13738)) := by
  have c87945 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.1
  have c87946 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87947 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87948 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87949 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87950 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1434_0 := arithEq_of_rows h (row := 1434) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_0
  simp only [← c87945, ← c87946, ← c87947] at e_1434_0
  have e_1434_1 := arithEq_of_rows h (row := 1434) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_1
  simp only [← c87948, ← c87949, k2, ← c87950] at e_1434_1
  have hr := e_1434_1
  simp only [e_1434_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13933 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1434 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13739)) := by
  have c87951 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87952 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87953 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87954 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87955 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87956 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1434_2 := arithEq_of_rows h (row := 1434) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_2
  simp only [← c87951, ← c87952, ← c87953] at e_1434_2
  have e_1434_3 := arithEq_of_rows h (row := 1434) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_3
  simp only [← c87954, ← c87955, k2, ← c87956] at e_1434_3
  have hr := e_1434_3
  simp only [e_1434_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13934 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1434 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13740)) := by
  have c87957 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87958 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87959 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87960 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87961 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87962 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1434_4 := arithEq_of_rows h (row := 1434) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_4
  simp only [← c87957, ← c87958, ← c87959] at e_1434_4
  have e_1434_5 := arithEq_of_rows h (row := 1434) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_5
  simp only [← c87960, ← c87961, k2, ← c87962] at e_1434_5
  have hr := e_1434_5
  simp only [e_1434_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13935 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1434 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13741)) := by
  have c87963 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87964 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87965 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87966 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87967 := (publicBatchWrapper86_copies2748 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c87968 := (publicBatchWrapper86_copies2749 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1434_6 := arithEq_of_rows h (row := 1434) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_6
  simp only [← c87963, ← c87964, ← c87965] at e_1434_6
  have e_1434_7 := arithEq_of_rows h (row := 1434) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_7
  simp only [← c87966, ← c87967, k2, ← c87968] at e_1434_7
  have hr := e_1434_7
  simp only [e_1434_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13936 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1434 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13742)) := by
  have c87969 := (publicBatchWrapper86_copies2749 a h).2.1
  have c87970 := (publicBatchWrapper86_copies2749 a h).2.2.1
  have c87971 := (publicBatchWrapper86_copies2749 a h).2.2.2.1
  have c87972 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.1
  have c87973 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.1
  have c87974 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1434_8 := arithEq_of_rows h (row := 1434) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_8
  simp only [← c87969, ← c87970, ← c87971] at e_1434_8
  have e_1434_9 := arithEq_of_rows h (row := 1434) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_9
  simp only [← c87972, ← c87973, k2, ← c87974] at e_1434_9
  have hr := e_1434_9
  simp only [e_1434_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13937 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1434 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13743)) := by
  have c87975 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.1
  have c87976 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.1
  have c87977 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.1
  have c87978 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.1
  have c87979 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c87980 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1434_10 := arithEq_of_rows h (row := 1434) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_10
  simp only [← c87975, ← c87976, ← c87977] at e_1434_10
  have e_1434_11 := arithEq_of_rows h (row := 1434) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_11
  simp only [← c87978, ← c87979, k2, ← c87980] at e_1434_11
  have hr := e_1434_11
  simp only [e_1434_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13938 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1434 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13744)) := by
  have c87981 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87982 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87983 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87984 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87985 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87986 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1434_12 := arithEq_of_rows h (row := 1434) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_12
  simp only [← c87981, ← c87982, ← c87983] at e_1434_12
  have e_1434_13 := arithEq_of_rows h (row := 1434) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_13
  simp only [← c87984, ← c87985, k2, ← c87986] at e_1434_13
  have hr := e_1434_13
  simp only [e_1434_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13939 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1434 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13745)) := by
  have c87987 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87988 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87989 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87990 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87991 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87992 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1434_14 := arithEq_of_rows h (row := 1434) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_14
  simp only [← c87987, ← c87988, ← c87989] at e_1434_14
  have e_1434_15 := arithEq_of_rows h (row := 1434) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_15
  simp only [← c87990, ← c87991, k2, ← c87992] at e_1434_15
  have hr := e_1434_15
  simp only [e_1434_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13940 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1434 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13746)) := by
  have c87993 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87994 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87995 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87996 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87997 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c87998 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1434_16 := arithEq_of_rows h (row := 1434) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_16
  simp only [← c87993, ← c87994, ← c87995] at e_1434_16
  have e_1434_17 := arithEq_of_rows h (row := 1434) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_17
  simp only [← c87996, ← c87997, k2, ← c87998] at e_1434_17
  have hr := e_1434_17
  simp only [e_1434_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13941 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1434 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13747)) := by
  have c87999 := (publicBatchWrapper86_copies2749 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88000 := (publicBatchWrapper86_copies2750 a h).1
  have c88001 := (publicBatchWrapper86_copies2750 a h).2.1
  have c88002 := (publicBatchWrapper86_copies2750 a h).2.2.1
  have c88003 := (publicBatchWrapper86_copies2750 a h).2.2.2.1
  have c88004 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1434_18 := arithEq_of_rows h (row := 1434) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_18
  simp only [← c87999, ← c88000, ← c88001] at e_1434_18
  have e_1434_19 := arithEq_of_rows h (row := 1434) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1434_19
  simp only [← c88002, ← c88003, k2, ← c88004] at e_1434_19
  have hr := e_1434_19
  simp only [e_1434_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13942 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1435 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13748)) := by
  have c88005 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.1
  have c88006 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.1
  have c88007 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.1
  have c88008 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.1
  have c88009 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.1
  have c88010 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1435_0 := arithEq_of_rows h (row := 1435) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_0
  simp only [← c88005, ← c88006, ← c88007] at e_1435_0
  have e_1435_1 := arithEq_of_rows h (row := 1435) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_1
  simp only [← c88008, ← c88009, k2, ← c88010] at e_1435_1
  have hr := e_1435_1
  simp only [e_1435_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13943 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1435 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13749)) := by
  have c88011 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88012 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88013 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88014 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88015 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88016 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1435_2 := arithEq_of_rows h (row := 1435) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_2
  simp only [← c88011, ← c88012, ← c88013] at e_1435_2
  have e_1435_3 := arithEq_of_rows h (row := 1435) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_3
  simp only [← c88014, ← c88015, k2, ← c88016] at e_1435_3
  have hr := e_1435_3
  simp only [e_1435_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13944 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1435 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13750)) := by
  have c88017 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88018 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88019 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88020 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88021 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88022 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1435_4 := arithEq_of_rows h (row := 1435) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_4
  simp only [← c88017, ← c88018, ← c88019] at e_1435_4
  have e_1435_5 := arithEq_of_rows h (row := 1435) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_5
  simp only [← c88020, ← c88021, k2, ← c88022] at e_1435_5
  have hr := e_1435_5
  simp only [e_1435_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13945 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1435 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13751)) := by
  have c88023 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88024 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88025 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88026 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88027 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88028 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1435_6 := arithEq_of_rows h (row := 1435) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_6
  simp only [← c88023, ← c88024, ← c88025] at e_1435_6
  have e_1435_7 := arithEq_of_rows h (row := 1435) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_7
  simp only [← c88026, ← c88027, k2, ← c88028] at e_1435_7
  have hr := e_1435_7
  simp only [e_1435_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13946 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1435 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13752)) := by
  have c88029 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88030 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88031 := (publicBatchWrapper86_copies2750 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88032 := (publicBatchWrapper86_copies2751 a h).1
  have c88033 := (publicBatchWrapper86_copies2751 a h).2.1
  have c88034 := (publicBatchWrapper86_copies2751 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1435_8 := arithEq_of_rows h (row := 1435) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_8
  simp only [← c88029, ← c88030, ← c88031] at e_1435_8
  have e_1435_9 := arithEq_of_rows h (row := 1435) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_9
  simp only [← c88032, ← c88033, k2, ← c88034] at e_1435_9
  have hr := e_1435_9
  simp only [e_1435_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13947 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1435 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13753)) := by
  have c88035 := (publicBatchWrapper86_copies2751 a h).2.2.2.1
  have c88036 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.1
  have c88037 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.1
  have c88038 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.1
  have c88039 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.1
  have c88040 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1435_10 := arithEq_of_rows h (row := 1435) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_10
  simp only [← c88035, ← c88036, ← c88037] at e_1435_10
  have e_1435_11 := arithEq_of_rows h (row := 1435) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_11
  simp only [← c88038, ← c88039, k2, ← c88040] at e_1435_11
  have hr := e_1435_11
  simp only [e_1435_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13948 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1435 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13754)) := by
  have c88041 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.1
  have c88042 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88043 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88044 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88045 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88046 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1435_12 := arithEq_of_rows h (row := 1435) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_12
  simp only [← c88041, ← c88042, ← c88043] at e_1435_12
  have e_1435_13 := arithEq_of_rows h (row := 1435) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_13
  simp only [← c88044, ← c88045, k2, ← c88046] at e_1435_13
  have hr := e_1435_13
  simp only [e_1435_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13949 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1435 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13755)) := by
  have c88047 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88048 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88049 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88050 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88051 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88052 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1435_14 := arithEq_of_rows h (row := 1435) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_14
  simp only [← c88047, ← c88048, ← c88049] at e_1435_14
  have e_1435_15 := arithEq_of_rows h (row := 1435) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_15
  simp only [← c88050, ← c88051, k2, ← c88052] at e_1435_15
  have hr := e_1435_15
  simp only [e_1435_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13950 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1435 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13756)) := by
  have c88053 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88054 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88055 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88056 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88057 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88058 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1435_16 := arithEq_of_rows h (row := 1435) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_16
  simp only [← c88053, ← c88054, ← c88055] at e_1435_16
  have e_1435_17 := arithEq_of_rows h (row := 1435) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_17
  simp only [← c88056, ← c88057, k2, ← c88058] at e_1435_17
  have hr := e_1435_17
  simp only [e_1435_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13951 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1435 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13757)) := by
  have c88059 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88060 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88061 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88062 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88063 := (publicBatchWrapper86_copies2751 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88064 := (publicBatchWrapper86_copies2752 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1435_18 := arithEq_of_rows h (row := 1435) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_18
  simp only [← c88059, ← c88060, ← c88061] at e_1435_18
  have e_1435_19 := arithEq_of_rows h (row := 1435) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1435_19
  simp only [← c88062, ← c88063, k2, ← c88064] at e_1435_19
  have hr := e_1435_19
  simp only [e_1435_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13952 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1436 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13758)) := by
  have c88065 := (publicBatchWrapper86_copies2752 a h).2.1
  have c88066 := (publicBatchWrapper86_copies2752 a h).2.2.1
  have c88067 := (publicBatchWrapper86_copies2752 a h).2.2.2.1
  have c88068 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.1
  have c88069 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.1
  have c88070 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1436_0 := arithEq_of_rows h (row := 1436) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_0
  simp only [← c88065, ← c88066, ← c88067] at e_1436_0
  have e_1436_1 := arithEq_of_rows h (row := 1436) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_1
  simp only [← c88068, ← c88069, k2, ← c88070] at e_1436_1
  have hr := e_1436_1
  simp only [e_1436_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13953 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1436 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13759)) := by
  have c88071 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.1
  have c88072 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.1
  have c88073 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.1
  have c88074 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88075 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88076 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1436_2 := arithEq_of_rows h (row := 1436) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_2
  simp only [← c88071, ← c88072, ← c88073] at e_1436_2
  have e_1436_3 := arithEq_of_rows h (row := 1436) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_3
  simp only [← c88074, ← c88075, k2, ← c88076] at e_1436_3
  have hr := e_1436_3
  simp only [e_1436_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13954 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1436 23) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13760)) := by
  have c88077 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88078 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88079 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88080 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88081 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88082 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1436_4 := arithEq_of_rows h (row := 1436) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_4
  simp only [← c88077, ← c88078, ← c88079] at e_1436_4
  have e_1436_5 := arithEq_of_rows h (row := 1436) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_5
  simp only [← c88080, ← c88081, k2, ← c88082] at e_1436_5
  have hr := e_1436_5
  simp only [e_1436_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13955 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1436 31) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13761)) := by
  have c88083 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88084 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88085 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88086 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88087 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88088 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1436_6 := arithEq_of_rows h (row := 1436) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_6
  simp only [← c88083, ← c88084, ← c88085] at e_1436_6
  have e_1436_7 := arithEq_of_rows h (row := 1436) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_7
  simp only [← c88086, ← c88087, k2, ← c88088] at e_1436_7
  have hr := e_1436_7
  simp only [e_1436_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13956 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1436 39) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13762)) := by
  have c88089 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88090 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88091 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88092 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88093 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88094 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1436_8 := arithEq_of_rows h (row := 1436) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_8
  simp only [← c88089, ← c88090, ← c88091] at e_1436_8
  have e_1436_9 := arithEq_of_rows h (row := 1436) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_9
  simp only [← c88092, ← c88093, k2, ← c88094] at e_1436_9
  have hr := e_1436_9
  simp only [e_1436_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13957 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1436 47) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13763)) := by
  have c88095 := (publicBatchWrapper86_copies2752 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88096 := (publicBatchWrapper86_copies2753 a h).1
  have c88097 := (publicBatchWrapper86_copies2753 a h).2.1
  have c88098 := (publicBatchWrapper86_copies2753 a h).2.2.1
  have c88099 := (publicBatchWrapper86_copies2753 a h).2.2.2.1
  have c88100 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1436_10 := arithEq_of_rows h (row := 1436) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_10
  simp only [← c88095, ← c88096, ← c88097] at e_1436_10
  have e_1436_11 := arithEq_of_rows h (row := 1436) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_11
  simp only [← c88098, ← c88099, k2, ← c88100] at e_1436_11
  have hr := e_1436_11
  simp only [e_1436_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13958 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1436 55) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13764)) := by
  have c88101 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.1
  have c88102 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.1
  have c88103 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.1
  have c88104 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.1
  have c88105 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.1
  have c88106 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1436_12 := arithEq_of_rows h (row := 1436) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_12
  simp only [← c88101, ← c88102, ← c88103] at e_1436_12
  have e_1436_13 := arithEq_of_rows h (row := 1436) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_13
  simp only [← c88104, ← c88105, k2, ← c88106] at e_1436_13
  have hr := e_1436_13
  simp only [e_1436_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13959 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1436 63) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13765)) := by
  have c88107 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88108 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88109 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88110 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88111 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88112 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1436_14 := arithEq_of_rows h (row := 1436) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_14
  simp only [← c88107, ← c88108, ← c88109] at e_1436_14
  have e_1436_15 := arithEq_of_rows h (row := 1436) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_15
  simp only [← c88110, ← c88111, k2, ← c88112] at e_1436_15
  have hr := e_1436_15
  simp only [e_1436_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13960 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1436 71) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13766)) := by
  have c88113 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88114 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88115 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88116 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88117 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88118 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1436_16 := arithEq_of_rows h (row := 1436) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_16
  simp only [← c88113, ← c88114, ← c88115] at e_1436_16
  have e_1436_17 := arithEq_of_rows h (row := 1436) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_17
  simp only [← c88116, ← c88117, k2, ← c88118] at e_1436_17
  have hr := e_1436_17
  simp only [e_1436_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13961 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1436 79) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13767)) := by
  have c88119 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88120 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88121 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88122 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88123 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88124 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1436_18 := arithEq_of_rows h (row := 1436) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_18
  simp only [← c88119, ← c88120, ← c88121] at e_1436_18
  have e_1436_19 := arithEq_of_rows h (row := 1436) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1436_19
  simp only [← c88122, ← c88123, k2, ← c88124] at e_1436_19
  have hr := e_1436_19
  simp only [e_1436_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13962 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1437 7) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13768)) := by
  have c88125 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88126 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88127 := (publicBatchWrapper86_copies2753 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88128 := (publicBatchWrapper86_copies2754 a h).1
  have c88129 := (publicBatchWrapper86_copies2754 a h).2.1
  have c88130 := (publicBatchWrapper86_copies2754 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1437_0 := arithEq_of_rows h (row := 1437) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_0
  simp only [← c88125, ← c88126, ← c88127] at e_1437_0
  have e_1437_1 := arithEq_of_rows h (row := 1437) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_1
  simp only [← c88128, ← c88129, k2, ← c88130] at e_1437_1
  have hr := e_1437_1
  simp only [e_1437_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13963 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1437 15) = bselect (a (.wire 67 59)) (a (.virt 200)) (a (.virt 13769)) := by
  have c88131 := (publicBatchWrapper86_copies2754 a h).2.2.2.1
  have c88132 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.1
  have c88133 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.1
  have c88134 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.1
  have c88135 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.1
  have c88136 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1437_2 := arithEq_of_rows h (row := 1437) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_2
  simp only [← c88131, ← c88132, ← c88133] at e_1437_2
  have e_1437_3 := arithEq_of_rows h (row := 1437) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_3
  simp only [← c88134, ← c88135, k2, ← c88136] at e_1437_3
  have hr := e_1437_3
  simp only [e_1437_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13964 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1437 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13810)) := by
  have c88137 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.1
  have c88138 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88139 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88140 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88141 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88142 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1437_4 := arithEq_of_rows h (row := 1437) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_4
  simp only [← c88137, ← c88138, ← c88139] at e_1437_4
  have e_1437_5 := arithEq_of_rows h (row := 1437) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_5
  simp only [← c88140, ← c88141, k2, ← c88142] at e_1437_5
  have hr := e_1437_5
  simp only [e_1437_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13965 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1437 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13811)) := by
  have c88143 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88144 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88145 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88146 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88147 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88148 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1437_6 := arithEq_of_rows h (row := 1437) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_6
  simp only [← c88143, ← c88144, ← c88145] at e_1437_6
  have e_1437_7 := arithEq_of_rows h (row := 1437) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_7
  simp only [← c88146, ← c88147, k2, ← c88148] at e_1437_7
  have hr := e_1437_7
  simp only [e_1437_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13966 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1437 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13812)) := by
  have c88149 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88150 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88151 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88152 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88153 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88154 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1437_8 := arithEq_of_rows h (row := 1437) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_8
  simp only [← c88149, ← c88150, ← c88151] at e_1437_8
  have e_1437_9 := arithEq_of_rows h (row := 1437) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_9
  simp only [← c88152, ← c88153, k2, ← c88154] at e_1437_9
  have hr := e_1437_9
  simp only [e_1437_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13967 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1437 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13813)) := by
  have c88155 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88156 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88157 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88158 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88159 := (publicBatchWrapper86_copies2754 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88160 := (publicBatchWrapper86_copies2755 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1437_10 := arithEq_of_rows h (row := 1437) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_10
  simp only [← c88155, ← c88156, ← c88157] at e_1437_10
  have e_1437_11 := arithEq_of_rows h (row := 1437) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_11
  simp only [← c88158, ← c88159, k2, ← c88160] at e_1437_11
  have hr := e_1437_11
  simp only [e_1437_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13968 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1437 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13814)) := by
  have c88161 := (publicBatchWrapper86_copies2755 a h).2.1
  have c88162 := (publicBatchWrapper86_copies2755 a h).2.2.1
  have c88163 := (publicBatchWrapper86_copies2755 a h).2.2.2.1
  have c88164 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.1
  have c88165 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.1
  have c88166 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1437_12 := arithEq_of_rows h (row := 1437) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_12
  simp only [← c88161, ← c88162, ← c88163] at e_1437_12
  have e_1437_13 := arithEq_of_rows h (row := 1437) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_13
  simp only [← c88164, ← c88165, k2, ← c88166] at e_1437_13
  have hr := e_1437_13
  simp only [e_1437_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13969 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1437 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13815)) := by
  have c88167 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.1
  have c88168 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.1
  have c88169 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.1
  have c88170 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88171 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88172 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1437_14 := arithEq_of_rows h (row := 1437) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_14
  simp only [← c88167, ← c88168, ← c88169] at e_1437_14
  have e_1437_15 := arithEq_of_rows h (row := 1437) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_15
  simp only [← c88170, ← c88171, k2, ← c88172] at e_1437_15
  have hr := e_1437_15
  simp only [e_1437_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13970 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1437 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13816)) := by
  have c88173 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88174 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88175 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88176 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88177 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88178 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1437_16 := arithEq_of_rows h (row := 1437) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_16
  simp only [← c88173, ← c88174, ← c88175] at e_1437_16
  have e_1437_17 := arithEq_of_rows h (row := 1437) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_17
  simp only [← c88176, ← c88177, k2, ← c88178] at e_1437_17
  have hr := e_1437_17
  simp only [e_1437_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13971 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1437 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13817)) := by
  have c88179 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88180 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88181 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88182 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88183 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88184 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1437_18 := arithEq_of_rows h (row := 1437) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_18
  simp only [← c88179, ← c88180, ← c88181] at e_1437_18
  have e_1437_19 := arithEq_of_rows h (row := 1437) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1437_19
  simp only [← c88182, ← c88183, k2, ← c88184] at e_1437_19
  have hr := e_1437_19
  simp only [e_1437_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13972 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1438 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13818)) := by
  have c88185 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88186 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88187 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88188 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88189 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88190 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1438_0 := arithEq_of_rows h (row := 1438) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_0
  simp only [← c88185, ← c88186, ← c88187] at e_1438_0
  have e_1438_1 := arithEq_of_rows h (row := 1438) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_1
  simp only [← c88188, ← c88189, k2, ← c88190] at e_1438_1
  have hr := e_1438_1
  simp only [e_1438_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13973 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1438 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13819)) := by
  have c88191 := (publicBatchWrapper86_copies2755 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88192 := (publicBatchWrapper86_copies2756 a h).1
  have c88193 := (publicBatchWrapper86_copies2756 a h).2.1
  have c88194 := (publicBatchWrapper86_copies2756 a h).2.2.1
  have c88195 := (publicBatchWrapper86_copies2756 a h).2.2.2.1
  have c88196 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1438_2 := arithEq_of_rows h (row := 1438) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_2
  simp only [← c88191, ← c88192, ← c88193] at e_1438_2
  have e_1438_3 := arithEq_of_rows h (row := 1438) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_3
  simp only [← c88194, ← c88195, k2, ← c88196] at e_1438_3
  have hr := e_1438_3
  simp only [e_1438_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13974 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1438 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13820)) := by
  have c88197 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.1
  have c88198 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.1
  have c88199 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.1
  have c88200 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.1
  have c88201 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.1
  have c88202 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1438_4 := arithEq_of_rows h (row := 1438) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_4
  simp only [← c88197, ← c88198, ← c88199] at e_1438_4
  have e_1438_5 := arithEq_of_rows h (row := 1438) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_5
  simp only [← c88200, ← c88201, k2, ← c88202] at e_1438_5
  have hr := e_1438_5
  simp only [e_1438_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13975 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1438 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13821)) := by
  have c88203 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88204 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88205 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88206 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88207 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88208 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1438_6 := arithEq_of_rows h (row := 1438) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_6
  simp only [← c88203, ← c88204, ← c88205] at e_1438_6
  have e_1438_7 := arithEq_of_rows h (row := 1438) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_7
  simp only [← c88206, ← c88207, k2, ← c88208] at e_1438_7
  have hr := e_1438_7
  simp only [e_1438_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13976 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1438 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13822)) := by
  have c88209 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88210 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88211 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88212 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88213 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88214 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1438_8 := arithEq_of_rows h (row := 1438) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_8
  simp only [← c88209, ← c88210, ← c88211] at e_1438_8
  have e_1438_9 := arithEq_of_rows h (row := 1438) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_9
  simp only [← c88212, ← c88213, k2, ← c88214] at e_1438_9
  have hr := e_1438_9
  simp only [e_1438_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13977 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1438 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13823)) := by
  have c88215 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88216 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88217 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88218 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88219 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88220 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1438_10 := arithEq_of_rows h (row := 1438) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_10
  simp only [← c88215, ← c88216, ← c88217] at e_1438_10
  have e_1438_11 := arithEq_of_rows h (row := 1438) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_11
  simp only [← c88218, ← c88219, k2, ← c88220] at e_1438_11
  have hr := e_1438_11
  simp only [e_1438_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13978 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1438 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13824)) := by
  have c88221 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88222 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88223 := (publicBatchWrapper86_copies2756 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88224 := (publicBatchWrapper86_copies2757 a h).1
  have c88225 := (publicBatchWrapper86_copies2757 a h).2.1
  have c88226 := (publicBatchWrapper86_copies2757 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1438_12 := arithEq_of_rows h (row := 1438) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_12
  simp only [← c88221, ← c88222, ← c88223] at e_1438_12
  have e_1438_13 := arithEq_of_rows h (row := 1438) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_13
  simp only [← c88224, ← c88225, k2, ← c88226] at e_1438_13
  have hr := e_1438_13
  simp only [e_1438_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13979 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1438 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13825)) := by
  have c88227 := (publicBatchWrapper86_copies2757 a h).2.2.2.1
  have c88228 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.1
  have c88229 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.1
  have c88230 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.1
  have c88231 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.1
  have c88232 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1438_14 := arithEq_of_rows h (row := 1438) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_14
  simp only [← c88227, ← c88228, ← c88229] at e_1438_14
  have e_1438_15 := arithEq_of_rows h (row := 1438) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_15
  simp only [← c88230, ← c88231, k2, ← c88232] at e_1438_15
  have hr := e_1438_15
  simp only [e_1438_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13980 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1438 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13826)) := by
  have c88233 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.1
  have c88234 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88235 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88236 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88237 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88238 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1438_16 := arithEq_of_rows h (row := 1438) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_16
  simp only [← c88233, ← c88234, ← c88235] at e_1438_16
  have e_1438_17 := arithEq_of_rows h (row := 1438) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_17
  simp only [← c88236, ← c88237, k2, ← c88238] at e_1438_17
  have hr := e_1438_17
  simp only [e_1438_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13981 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1438 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13827)) := by
  have c88239 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88240 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88241 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88242 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88243 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88244 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1438_18 := arithEq_of_rows h (row := 1438) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_18
  simp only [← c88239, ← c88240, ← c88241] at e_1438_18
  have e_1438_19 := arithEq_of_rows h (row := 1438) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1438_19
  simp only [← c88242, ← c88243, k2, ← c88244] at e_1438_19
  have hr := e_1438_19
  simp only [e_1438_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13982 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1439 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13828)) := by
  have c88245 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88246 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88247 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88248 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88249 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88250 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1439_0 := arithEq_of_rows h (row := 1439) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_0
  simp only [← c88245, ← c88246, ← c88247] at e_1439_0
  have e_1439_1 := arithEq_of_rows h (row := 1439) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_1
  simp only [← c88248, ← c88249, k2, ← c88250] at e_1439_1
  have hr := e_1439_1
  simp only [e_1439_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13983 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1439 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13829)) := by
  have c88251 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88252 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88253 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88254 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88255 := (publicBatchWrapper86_copies2757 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88256 := (publicBatchWrapper86_copies2758 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1439_2 := arithEq_of_rows h (row := 1439) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_2
  simp only [← c88251, ← c88252, ← c88253] at e_1439_2
  have e_1439_3 := arithEq_of_rows h (row := 1439) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_3
  simp only [← c88254, ← c88255, k2, ← c88256] at e_1439_3
  have hr := e_1439_3
  simp only [e_1439_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13984 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1439 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13830)) := by
  have c88257 := (publicBatchWrapper86_copies2758 a h).2.1
  have c88258 := (publicBatchWrapper86_copies2758 a h).2.2.1
  have c88259 := (publicBatchWrapper86_copies2758 a h).2.2.2.1
  have c88260 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.1
  have c88261 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.1
  have c88262 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1439_4 := arithEq_of_rows h (row := 1439) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_4
  simp only [← c88257, ← c88258, ← c88259] at e_1439_4
  have e_1439_5 := arithEq_of_rows h (row := 1439) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_5
  simp only [← c88260, ← c88261, k2, ← c88262] at e_1439_5
  have hr := e_1439_5
  simp only [e_1439_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13985 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1439 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13831)) := by
  have c88263 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.1
  have c88264 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.1
  have c88265 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.1
  have c88266 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88267 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88268 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1439_6 := arithEq_of_rows h (row := 1439) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_6
  simp only [← c88263, ← c88264, ← c88265] at e_1439_6
  have e_1439_7 := arithEq_of_rows h (row := 1439) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_7
  simp only [← c88266, ← c88267, k2, ← c88268] at e_1439_7
  have hr := e_1439_7
  simp only [e_1439_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13986 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1439 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13832)) := by
  have c88269 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88270 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88271 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88272 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88273 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88274 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1439_8 := arithEq_of_rows h (row := 1439) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_8
  simp only [← c88269, ← c88270, ← c88271] at e_1439_8
  have e_1439_9 := arithEq_of_rows h (row := 1439) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_9
  simp only [← c88272, ← c88273, k2, ← c88274] at e_1439_9
  have hr := e_1439_9
  simp only [e_1439_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13987 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1439 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13833)) := by
  have c88275 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88276 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88277 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88278 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88279 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88280 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1439_10 := arithEq_of_rows h (row := 1439) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_10
  simp only [← c88275, ← c88276, ← c88277] at e_1439_10
  have e_1439_11 := arithEq_of_rows h (row := 1439) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_11
  simp only [← c88278, ← c88279, k2, ← c88280] at e_1439_11
  have hr := e_1439_11
  simp only [e_1439_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13988 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1439 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13834)) := by
  have c88281 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88282 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88283 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88284 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88285 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88286 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1439_12 := arithEq_of_rows h (row := 1439) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_12
  simp only [← c88281, ← c88282, ← c88283] at e_1439_12
  have e_1439_13 := arithEq_of_rows h (row := 1439) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_13
  simp only [← c88284, ← c88285, k2, ← c88286] at e_1439_13
  have hr := e_1439_13
  simp only [e_1439_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13989 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1439 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13835)) := by
  have c88287 := (publicBatchWrapper86_copies2758 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88288 := (publicBatchWrapper86_copies2759 a h).1
  have c88289 := (publicBatchWrapper86_copies2759 a h).2.1
  have c88290 := (publicBatchWrapper86_copies2759 a h).2.2.1
  have c88291 := (publicBatchWrapper86_copies2759 a h).2.2.2.1
  have c88292 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1439_14 := arithEq_of_rows h (row := 1439) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_14
  simp only [← c88287, ← c88288, ← c88289] at e_1439_14
  have e_1439_15 := arithEq_of_rows h (row := 1439) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_15
  simp only [← c88290, ← c88291, k2, ← c88292] at e_1439_15
  have hr := e_1439_15
  simp only [e_1439_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13990 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1439 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13836)) := by
  have c88293 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.1
  have c88294 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.1
  have c88295 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.1
  have c88296 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.1
  have c88297 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.1
  have c88298 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1439_16 := arithEq_of_rows h (row := 1439) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_16
  simp only [← c88293, ← c88294, ← c88295] at e_1439_16
  have e_1439_17 := arithEq_of_rows h (row := 1439) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_17
  simp only [← c88296, ← c88297, k2, ← c88298] at e_1439_17
  have hr := e_1439_17
  simp only [e_1439_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13991 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1439 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13837)) := by
  have c88299 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88300 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88301 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88302 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88303 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88304 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1439_18 := arithEq_of_rows h (row := 1439) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_18
  simp only [← c88299, ← c88300, ← c88301] at e_1439_18
  have e_1439_19 := arithEq_of_rows h (row := 1439) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1439_19
  simp only [← c88302, ← c88303, k2, ← c88304] at e_1439_19
  have hr := e_1439_19
  simp only [e_1439_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13992 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1440 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13838)) := by
  have c88305 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88306 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88307 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88308 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88309 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88310 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1440_0 := arithEq_of_rows h (row := 1440) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_0
  simp only [← c88305, ← c88306, ← c88307] at e_1440_0
  have e_1440_1 := arithEq_of_rows h (row := 1440) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_1
  simp only [← c88308, ← c88309, k2, ← c88310] at e_1440_1
  have hr := e_1440_1
  simp only [e_1440_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13993 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1440 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13839)) := by
  have c88311 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88312 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88313 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88314 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88315 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88316 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1440_2 := arithEq_of_rows h (row := 1440) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_2
  simp only [← c88311, ← c88312, ← c88313] at e_1440_2
  have e_1440_3 := arithEq_of_rows h (row := 1440) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_3
  simp only [← c88314, ← c88315, k2, ← c88316] at e_1440_3
  have hr := e_1440_3
  simp only [e_1440_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13994 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1440 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13840)) := by
  have c88317 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88318 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88319 := (publicBatchWrapper86_copies2759 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88320 := (publicBatchWrapper86_copies2760 a h).1
  have c88321 := (publicBatchWrapper86_copies2760 a h).2.1
  have c88322 := (publicBatchWrapper86_copies2760 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1440_4 := arithEq_of_rows h (row := 1440) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_4
  simp only [← c88317, ← c88318, ← c88319] at e_1440_4
  have e_1440_5 := arithEq_of_rows h (row := 1440) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_5
  simp only [← c88320, ← c88321, k2, ← c88322] at e_1440_5
  have hr := e_1440_5
  simp only [e_1440_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13995 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1440 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13841)) := by
  have c88323 := (publicBatchWrapper86_copies2760 a h).2.2.2.1
  have c88324 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.1
  have c88325 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.1
  have c88326 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.1
  have c88327 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.1
  have c88328 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1440_6 := arithEq_of_rows h (row := 1440) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_6
  simp only [← c88323, ← c88324, ← c88325] at e_1440_6
  have e_1440_7 := arithEq_of_rows h (row := 1440) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_7
  simp only [← c88326, ← c88327, k2, ← c88328] at e_1440_7
  have hr := e_1440_7
  simp only [e_1440_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13996 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1440 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13842)) := by
  have c88329 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.1
  have c88330 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88331 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88332 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88333 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88334 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1440_8 := arithEq_of_rows h (row := 1440) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_8
  simp only [← c88329, ← c88330, ← c88331] at e_1440_8
  have e_1440_9 := arithEq_of_rows h (row := 1440) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_9
  simp only [← c88332, ← c88333, k2, ← c88334] at e_1440_9
  have hr := e_1440_9
  simp only [e_1440_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13997 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1440 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13843)) := by
  have c88335 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88336 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88337 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88338 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88339 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88340 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1440_10 := arithEq_of_rows h (row := 1440) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_10
  simp only [← c88335, ← c88336, ← c88337] at e_1440_10
  have e_1440_11 := arithEq_of_rows h (row := 1440) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_11
  simp only [← c88338, ← c88339, k2, ← c88340] at e_1440_11
  have hr := e_1440_11
  simp only [e_1440_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13998 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1440 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13844)) := by
  have c88341 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88342 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88343 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88344 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88345 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88346 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1440_12 := arithEq_of_rows h (row := 1440) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_12
  simp only [← c88341, ← c88342, ← c88343] at e_1440_12
  have e_1440_13 := arithEq_of_rows h (row := 1440) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_13
  simp only [← c88344, ← c88345, k2, ← c88346] at e_1440_13
  have hr := e_1440_13
  simp only [e_1440_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f13999 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1440 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13845)) := by
  have c88347 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88348 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88349 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88350 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88351 := (publicBatchWrapper86_copies2760 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88352 := (publicBatchWrapper86_copies2761 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1440_14 := arithEq_of_rows h (row := 1440) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_14
  simp only [← c88347, ← c88348, ← c88349] at e_1440_14
  have e_1440_15 := arithEq_of_rows h (row := 1440) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_15
  simp only [← c88350, ← c88351, k2, ← c88352] at e_1440_15
  have hr := e_1440_15
  simp only [e_1440_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14000 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1440 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13846)) := by
  have c88353 := (publicBatchWrapper86_copies2761 a h).2.1
  have c88354 := (publicBatchWrapper86_copies2761 a h).2.2.1
  have c88355 := (publicBatchWrapper86_copies2761 a h).2.2.2.1
  have c88356 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.1
  have c88357 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.1
  have c88358 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1440_16 := arithEq_of_rows h (row := 1440) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_16
  simp only [← c88353, ← c88354, ← c88355] at e_1440_16
  have e_1440_17 := arithEq_of_rows h (row := 1440) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_17
  simp only [← c88356, ← c88357, k2, ← c88358] at e_1440_17
  have hr := e_1440_17
  simp only [e_1440_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14001 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1440 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13847)) := by
  have c88359 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.1
  have c88360 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.1
  have c88361 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.1
  have c88362 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88363 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88364 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1440_18 := arithEq_of_rows h (row := 1440) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_18
  simp only [← c88359, ← c88360, ← c88361] at e_1440_18
  have e_1440_19 := arithEq_of_rows h (row := 1440) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1440_19
  simp only [← c88362, ← c88363, k2, ← c88364] at e_1440_19
  have hr := e_1440_19
  simp only [e_1440_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14002 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1441 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13848)) := by
  have c88365 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88366 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88367 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88368 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88369 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88370 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1441_0 := arithEq_of_rows h (row := 1441) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_0
  simp only [← c88365, ← c88366, ← c88367] at e_1441_0
  have e_1441_1 := arithEq_of_rows h (row := 1441) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_1
  simp only [← c88368, ← c88369, k2, ← c88370] at e_1441_1
  have hr := e_1441_1
  simp only [e_1441_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14003 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1441 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13849)) := by
  have c88371 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88372 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88373 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88374 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88375 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88376 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1441_2 := arithEq_of_rows h (row := 1441) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_2
  simp only [← c88371, ← c88372, ← c88373] at e_1441_2
  have e_1441_3 := arithEq_of_rows h (row := 1441) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_3
  simp only [← c88374, ← c88375, k2, ← c88376] at e_1441_3
  have hr := e_1441_3
  simp only [e_1441_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14004 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1441 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13850)) := by
  have c88377 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88378 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88379 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88380 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88381 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88382 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1441_4 := arithEq_of_rows h (row := 1441) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_4
  simp only [← c88377, ← c88378, ← c88379] at e_1441_4
  have e_1441_5 := arithEq_of_rows h (row := 1441) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_5
  simp only [← c88380, ← c88381, k2, ← c88382] at e_1441_5
  have hr := e_1441_5
  simp only [e_1441_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14005 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1441 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13851)) := by
  have c88383 := (publicBatchWrapper86_copies2761 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88384 := (publicBatchWrapper86_copies2762 a h).1
  have c88385 := (publicBatchWrapper86_copies2762 a h).2.1
  have c88386 := (publicBatchWrapper86_copies2762 a h).2.2.1
  have c88387 := (publicBatchWrapper86_copies2762 a h).2.2.2.1
  have c88388 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1441_6 := arithEq_of_rows h (row := 1441) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_6
  simp only [← c88383, ← c88384, ← c88385] at e_1441_6
  have e_1441_7 := arithEq_of_rows h (row := 1441) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_7
  simp only [← c88386, ← c88387, k2, ← c88388] at e_1441_7
  have hr := e_1441_7
  simp only [e_1441_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14006 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1441 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13852)) := by
  have c88389 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.1
  have c88390 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.1
  have c88391 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.1
  have c88392 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.1
  have c88393 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.1
  have c88394 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1441_8 := arithEq_of_rows h (row := 1441) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_8
  simp only [← c88389, ← c88390, ← c88391] at e_1441_8
  have e_1441_9 := arithEq_of_rows h (row := 1441) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_9
  simp only [← c88392, ← c88393, k2, ← c88394] at e_1441_9
  have hr := e_1441_9
  simp only [e_1441_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14007 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1441 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13853)) := by
  have c88395 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88396 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88397 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88398 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88399 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88400 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1441_10 := arithEq_of_rows h (row := 1441) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_10
  simp only [← c88395, ← c88396, ← c88397] at e_1441_10
  have e_1441_11 := arithEq_of_rows h (row := 1441) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_11
  simp only [← c88398, ← c88399, k2, ← c88400] at e_1441_11
  have hr := e_1441_11
  simp only [e_1441_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14008 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1441 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13854)) := by
  have c88401 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88402 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88403 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88404 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88405 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88406 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1441_12 := arithEq_of_rows h (row := 1441) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_12
  simp only [← c88401, ← c88402, ← c88403] at e_1441_12
  have e_1441_13 := arithEq_of_rows h (row := 1441) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_13
  simp only [← c88404, ← c88405, k2, ← c88406] at e_1441_13
  have hr := e_1441_13
  simp only [e_1441_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14009 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1441 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13855)) := by
  have c88407 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88408 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88409 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88410 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88411 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88412 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1441_14 := arithEq_of_rows h (row := 1441) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_14
  simp only [← c88407, ← c88408, ← c88409] at e_1441_14
  have e_1441_15 := arithEq_of_rows h (row := 1441) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_15
  simp only [← c88410, ← c88411, k2, ← c88412] at e_1441_15
  have hr := e_1441_15
  simp only [e_1441_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14010 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1441 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13856)) := by
  have c88413 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88414 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88415 := (publicBatchWrapper86_copies2762 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88416 := (publicBatchWrapper86_copies2763 a h).1
  have c88417 := (publicBatchWrapper86_copies2763 a h).2.1
  have c88418 := (publicBatchWrapper86_copies2763 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1441_16 := arithEq_of_rows h (row := 1441) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_16
  simp only [← c88413, ← c88414, ← c88415] at e_1441_16
  have e_1441_17 := arithEq_of_rows h (row := 1441) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_17
  simp only [← c88416, ← c88417, k2, ← c88418] at e_1441_17
  have hr := e_1441_17
  simp only [e_1441_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14011 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1441 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13857)) := by
  have c88419 := (publicBatchWrapper86_copies2763 a h).2.2.2.1
  have c88420 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.1
  have c88421 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.1
  have c88422 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.1
  have c88423 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.1
  have c88424 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1441_18 := arithEq_of_rows h (row := 1441) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_18
  simp only [← c88419, ← c88420, ← c88421] at e_1441_18
  have e_1441_19 := arithEq_of_rows h (row := 1441) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1441_19
  simp only [← c88422, ← c88423, k2, ← c88424] at e_1441_19
  have hr := e_1441_19
  simp only [e_1441_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14012 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1442 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13858)) := by
  have c88425 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.1
  have c88426 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88427 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88428 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88429 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88430 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1442_0 := arithEq_of_rows h (row := 1442) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_0
  simp only [← c88425, ← c88426, ← c88427] at e_1442_0
  have e_1442_1 := arithEq_of_rows h (row := 1442) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_1
  simp only [← c88428, ← c88429, k2, ← c88430] at e_1442_1
  have hr := e_1442_1
  simp only [e_1442_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14013 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1442 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13859)) := by
  have c88431 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88432 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88433 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88434 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88435 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88436 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1442_2 := arithEq_of_rows h (row := 1442) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_2
  simp only [← c88431, ← c88432, ← c88433] at e_1442_2
  have e_1442_3 := arithEq_of_rows h (row := 1442) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_3
  simp only [← c88434, ← c88435, k2, ← c88436] at e_1442_3
  have hr := e_1442_3
  simp only [e_1442_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14014 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1442 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13860)) := by
  have c88437 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88438 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88439 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88440 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88441 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88442 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1442_4 := arithEq_of_rows h (row := 1442) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_4
  simp only [← c88437, ← c88438, ← c88439] at e_1442_4
  have e_1442_5 := arithEq_of_rows h (row := 1442) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_5
  simp only [← c88440, ← c88441, k2, ← c88442] at e_1442_5
  have hr := e_1442_5
  simp only [e_1442_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14015 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1442 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13861)) := by
  have c88443 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88444 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88445 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88446 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88447 := (publicBatchWrapper86_copies2763 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88448 := (publicBatchWrapper86_copies2764 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1442_6 := arithEq_of_rows h (row := 1442) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_6
  simp only [← c88443, ← c88444, ← c88445] at e_1442_6
  have e_1442_7 := arithEq_of_rows h (row := 1442) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_7
  simp only [← c88446, ← c88447, k2, ← c88448] at e_1442_7
  have hr := e_1442_7
  simp only [e_1442_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14016 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1442 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13862)) := by
  have c88449 := (publicBatchWrapper86_copies2764 a h).2.1
  have c88450 := (publicBatchWrapper86_copies2764 a h).2.2.1
  have c88451 := (publicBatchWrapper86_copies2764 a h).2.2.2.1
  have c88452 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.1
  have c88453 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.1
  have c88454 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1442_8 := arithEq_of_rows h (row := 1442) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_8
  simp only [← c88449, ← c88450, ← c88451] at e_1442_8
  have e_1442_9 := arithEq_of_rows h (row := 1442) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_9
  simp only [← c88452, ← c88453, k2, ← c88454] at e_1442_9
  have hr := e_1442_9
  simp only [e_1442_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14017 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1442 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13863)) := by
  have c88455 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.1
  have c88456 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.1
  have c88457 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.1
  have c88458 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88459 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88460 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1442_10 := arithEq_of_rows h (row := 1442) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_10
  simp only [← c88455, ← c88456, ← c88457] at e_1442_10
  have e_1442_11 := arithEq_of_rows h (row := 1442) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_11
  simp only [← c88458, ← c88459, k2, ← c88460] at e_1442_11
  have hr := e_1442_11
  simp only [e_1442_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14018 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1442 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13864)) := by
  have c88461 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88462 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88463 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88464 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88465 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88466 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1442_12 := arithEq_of_rows h (row := 1442) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_12
  simp only [← c88461, ← c88462, ← c88463] at e_1442_12
  have e_1442_13 := arithEq_of_rows h (row := 1442) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_13
  simp only [← c88464, ← c88465, k2, ← c88466] at e_1442_13
  have hr := e_1442_13
  simp only [e_1442_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14019 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1442 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13865)) := by
  have c88467 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88468 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88469 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88470 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88471 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88472 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1442_14 := arithEq_of_rows h (row := 1442) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_14
  simp only [← c88467, ← c88468, ← c88469] at e_1442_14
  have e_1442_15 := arithEq_of_rows h (row := 1442) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_15
  simp only [← c88470, ← c88471, k2, ← c88472] at e_1442_15
  have hr := e_1442_15
  simp only [e_1442_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14020 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1442 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13866)) := by
  have c88473 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88474 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88475 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88476 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88477 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88478 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1442_16 := arithEq_of_rows h (row := 1442) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_16
  simp only [← c88473, ← c88474, ← c88475] at e_1442_16
  have e_1442_17 := arithEq_of_rows h (row := 1442) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_17
  simp only [← c88476, ← c88477, k2, ← c88478] at e_1442_17
  have hr := e_1442_17
  simp only [e_1442_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14021 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1442 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13867)) := by
  have c88479 := (publicBatchWrapper86_copies2764 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88480 := (publicBatchWrapper86_copies2765 a h).1
  have c88481 := (publicBatchWrapper86_copies2765 a h).2.1
  have c88482 := (publicBatchWrapper86_copies2765 a h).2.2.1
  have c88483 := (publicBatchWrapper86_copies2765 a h).2.2.2.1
  have c88484 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1442_18 := arithEq_of_rows h (row := 1442) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_18
  simp only [← c88479, ← c88480, ← c88481] at e_1442_18
  have e_1442_19 := arithEq_of_rows h (row := 1442) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1442_19
  simp only [← c88482, ← c88483, k2, ← c88484] at e_1442_19
  have hr := e_1442_19
  simp only [e_1442_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14022 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1443 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13868)) := by
  have c88485 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.1
  have c88486 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.1
  have c88487 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.1
  have c88488 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.1
  have c88489 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.1
  have c88490 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1443_0 := arithEq_of_rows h (row := 1443) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_0
  simp only [← c88485, ← c88486, ← c88487] at e_1443_0
  have e_1443_1 := arithEq_of_rows h (row := 1443) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_1
  simp only [← c88488, ← c88489, k2, ← c88490] at e_1443_1
  have hr := e_1443_1
  simp only [e_1443_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14023 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1443 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13869)) := by
  have c88491 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88492 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88493 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88494 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88495 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88496 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1443_2 := arithEq_of_rows h (row := 1443) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_2
  simp only [← c88491, ← c88492, ← c88493] at e_1443_2
  have e_1443_3 := arithEq_of_rows h (row := 1443) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_3
  simp only [← c88494, ← c88495, k2, ← c88496] at e_1443_3
  have hr := e_1443_3
  simp only [e_1443_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14024 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1443 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13870)) := by
  have c88497 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88498 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88499 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88500 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88501 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88502 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1443_4 := arithEq_of_rows h (row := 1443) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_4
  simp only [← c88497, ← c88498, ← c88499] at e_1443_4
  have e_1443_5 := arithEq_of_rows h (row := 1443) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_5
  simp only [← c88500, ← c88501, k2, ← c88502] at e_1443_5
  have hr := e_1443_5
  simp only [e_1443_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14025 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1443 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13871)) := by
  have c88503 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88504 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88505 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88506 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88507 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88508 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1443_6 := arithEq_of_rows h (row := 1443) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_6
  simp only [← c88503, ← c88504, ← c88505] at e_1443_6
  have e_1443_7 := arithEq_of_rows h (row := 1443) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_7
  simp only [← c88506, ← c88507, k2, ← c88508] at e_1443_7
  have hr := e_1443_7
  simp only [e_1443_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14026 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1443 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13872)) := by
  have c88509 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88510 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88511 := (publicBatchWrapper86_copies2765 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88512 := (publicBatchWrapper86_copies2766 a h).1
  have c88513 := (publicBatchWrapper86_copies2766 a h).2.1
  have c88514 := (publicBatchWrapper86_copies2766 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1443_8 := arithEq_of_rows h (row := 1443) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_8
  simp only [← c88509, ← c88510, ← c88511] at e_1443_8
  have e_1443_9 := arithEq_of_rows h (row := 1443) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_9
  simp only [← c88512, ← c88513, k2, ← c88514] at e_1443_9
  have hr := e_1443_9
  simp only [e_1443_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14027 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1443 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13873)) := by
  have c88515 := (publicBatchWrapper86_copies2766 a h).2.2.2.1
  have c88516 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.1
  have c88517 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.1
  have c88518 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.1
  have c88519 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.1
  have c88520 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1443_10 := arithEq_of_rows h (row := 1443) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_10
  simp only [← c88515, ← c88516, ← c88517] at e_1443_10
  have e_1443_11 := arithEq_of_rows h (row := 1443) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_11
  simp only [← c88518, ← c88519, k2, ← c88520] at e_1443_11
  have hr := e_1443_11
  simp only [e_1443_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14028 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1443 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13874)) := by
  have c88521 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.1
  have c88522 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88523 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88524 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88525 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88526 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1443_12 := arithEq_of_rows h (row := 1443) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_12
  simp only [← c88521, ← c88522, ← c88523] at e_1443_12
  have e_1443_13 := arithEq_of_rows h (row := 1443) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_13
  simp only [← c88524, ← c88525, k2, ← c88526] at e_1443_13
  have hr := e_1443_13
  simp only [e_1443_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14029 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1443 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13875)) := by
  have c88527 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88528 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88529 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88530 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88531 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88532 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1443_14 := arithEq_of_rows h (row := 1443) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_14
  simp only [← c88527, ← c88528, ← c88529] at e_1443_14
  have e_1443_15 := arithEq_of_rows h (row := 1443) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_15
  simp only [← c88530, ← c88531, k2, ← c88532] at e_1443_15
  have hr := e_1443_15
  simp only [e_1443_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14030 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1443 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13876)) := by
  have c88533 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88534 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88535 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88536 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88537 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88538 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1443_16 := arithEq_of_rows h (row := 1443) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_16
  simp only [← c88533, ← c88534, ← c88535] at e_1443_16
  have e_1443_17 := arithEq_of_rows h (row := 1443) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_17
  simp only [← c88536, ← c88537, k2, ← c88538] at e_1443_17
  have hr := e_1443_17
  simp only [e_1443_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14031 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1443 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13877)) := by
  have c88539 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88540 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88541 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88542 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88543 := (publicBatchWrapper86_copies2766 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88544 := (publicBatchWrapper86_copies2767 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1443_18 := arithEq_of_rows h (row := 1443) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_18
  simp only [← c88539, ← c88540, ← c88541] at e_1443_18
  have e_1443_19 := arithEq_of_rows h (row := 1443) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1443_19
  simp only [← c88542, ← c88543, k2, ← c88544] at e_1443_19
  have hr := e_1443_19
  simp only [e_1443_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14032 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1444 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13878)) := by
  have c88545 := (publicBatchWrapper86_copies2767 a h).2.1
  have c88546 := (publicBatchWrapper86_copies2767 a h).2.2.1
  have c88547 := (publicBatchWrapper86_copies2767 a h).2.2.2.1
  have c88548 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.1
  have c88549 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.1
  have c88550 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1444_0 := arithEq_of_rows h (row := 1444) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_0
  simp only [← c88545, ← c88546, ← c88547] at e_1444_0
  have e_1444_1 := arithEq_of_rows h (row := 1444) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_1
  simp only [← c88548, ← c88549, k2, ← c88550] at e_1444_1
  have hr := e_1444_1
  simp only [e_1444_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14033 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1444 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13879)) := by
  have c88551 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.1
  have c88552 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.1
  have c88553 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.1
  have c88554 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88555 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88556 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1444_2 := arithEq_of_rows h (row := 1444) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_2
  simp only [← c88551, ← c88552, ← c88553] at e_1444_2
  have e_1444_3 := arithEq_of_rows h (row := 1444) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_3
  simp only [← c88554, ← c88555, k2, ← c88556] at e_1444_3
  have hr := e_1444_3
  simp only [e_1444_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14034 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1444 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13880)) := by
  have c88557 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88558 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88559 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88560 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88561 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88562 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1444_4 := arithEq_of_rows h (row := 1444) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_4
  simp only [← c88557, ← c88558, ← c88559] at e_1444_4
  have e_1444_5 := arithEq_of_rows h (row := 1444) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_5
  simp only [← c88560, ← c88561, k2, ← c88562] at e_1444_5
  have hr := e_1444_5
  simp only [e_1444_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14035 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1444 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13881)) := by
  have c88563 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88564 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88565 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88566 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88567 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88568 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1444_6 := arithEq_of_rows h (row := 1444) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_6
  simp only [← c88563, ← c88564, ← c88565] at e_1444_6
  have e_1444_7 := arithEq_of_rows h (row := 1444) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_7
  simp only [← c88566, ← c88567, k2, ← c88568] at e_1444_7
  have hr := e_1444_7
  simp only [e_1444_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14036 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1444 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13882)) := by
  have c88569 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88570 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88571 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88572 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88573 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88574 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1444_8 := arithEq_of_rows h (row := 1444) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_8
  simp only [← c88569, ← c88570, ← c88571] at e_1444_8
  have e_1444_9 := arithEq_of_rows h (row := 1444) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_9
  simp only [← c88572, ← c88573, k2, ← c88574] at e_1444_9
  have hr := e_1444_9
  simp only [e_1444_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14037 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1444 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13883)) := by
  have c88575 := (publicBatchWrapper86_copies2767 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88576 := (publicBatchWrapper86_copies2768 a h).1
  have c88577 := (publicBatchWrapper86_copies2768 a h).2.1
  have c88578 := (publicBatchWrapper86_copies2768 a h).2.2.1
  have c88579 := (publicBatchWrapper86_copies2768 a h).2.2.2.1
  have c88580 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1444_10 := arithEq_of_rows h (row := 1444) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_10
  simp only [← c88575, ← c88576, ← c88577] at e_1444_10
  have e_1444_11 := arithEq_of_rows h (row := 1444) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_11
  simp only [← c88578, ← c88579, k2, ← c88580] at e_1444_11
  have hr := e_1444_11
  simp only [e_1444_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14038 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1444 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13884)) := by
  have c88581 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.1
  have c88582 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.1
  have c88583 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.1
  have c88584 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.1
  have c88585 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.1
  have c88586 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1444_12 := arithEq_of_rows h (row := 1444) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_12
  simp only [← c88581, ← c88582, ← c88583] at e_1444_12
  have e_1444_13 := arithEq_of_rows h (row := 1444) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_13
  simp only [← c88584, ← c88585, k2, ← c88586] at e_1444_13
  have hr := e_1444_13
  simp only [e_1444_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14039 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1444 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13885)) := by
  have c88587 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88588 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88589 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88590 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88591 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88592 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1444_14 := arithEq_of_rows h (row := 1444) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_14
  simp only [← c88587, ← c88588, ← c88589] at e_1444_14
  have e_1444_15 := arithEq_of_rows h (row := 1444) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_15
  simp only [← c88590, ← c88591, k2, ← c88592] at e_1444_15
  have hr := e_1444_15
  simp only [e_1444_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14040 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1444 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13886)) := by
  have c88593 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88594 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88595 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88596 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88597 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88598 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1444_16 := arithEq_of_rows h (row := 1444) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_16
  simp only [← c88593, ← c88594, ← c88595] at e_1444_16
  have e_1444_17 := arithEq_of_rows h (row := 1444) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_17
  simp only [← c88596, ← c88597, k2, ← c88598] at e_1444_17
  have hr := e_1444_17
  simp only [e_1444_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14041 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1444 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13887)) := by
  have c88599 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88600 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88601 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88602 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88603 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88604 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1444_18 := arithEq_of_rows h (row := 1444) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_18
  simp only [← c88599, ← c88600, ← c88601] at e_1444_18
  have e_1444_19 := arithEq_of_rows h (row := 1444) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1444_19
  simp only [← c88602, ← c88603, k2, ← c88604] at e_1444_19
  have hr := e_1444_19
  simp only [e_1444_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14042 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1445 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13888)) := by
  have c88605 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88606 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88607 := (publicBatchWrapper86_copies2768 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88608 := (publicBatchWrapper86_copies2769 a h).1
  have c88609 := (publicBatchWrapper86_copies2769 a h).2.1
  have c88610 := (publicBatchWrapper86_copies2769 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1445_0 := arithEq_of_rows h (row := 1445) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_0
  simp only [← c88605, ← c88606, ← c88607] at e_1445_0
  have e_1445_1 := arithEq_of_rows h (row := 1445) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_1
  simp only [← c88608, ← c88609, k2, ← c88610] at e_1445_1
  have hr := e_1445_1
  simp only [e_1445_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14043 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1445 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13889)) := by
  have c88611 := (publicBatchWrapper86_copies2769 a h).2.2.2.1
  have c88612 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.1
  have c88613 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.1
  have c88614 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.1
  have c88615 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.1
  have c88616 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1445_2 := arithEq_of_rows h (row := 1445) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_2
  simp only [← c88611, ← c88612, ← c88613] at e_1445_2
  have e_1445_3 := arithEq_of_rows h (row := 1445) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_3
  simp only [← c88614, ← c88615, k2, ← c88616] at e_1445_3
  have hr := e_1445_3
  simp only [e_1445_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14044 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1445 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13890)) := by
  have c88617 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.1
  have c88618 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88619 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88620 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88621 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88622 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1445_4 := arithEq_of_rows h (row := 1445) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_4
  simp only [← c88617, ← c88618, ← c88619] at e_1445_4
  have e_1445_5 := arithEq_of_rows h (row := 1445) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_5
  simp only [← c88620, ← c88621, k2, ← c88622] at e_1445_5
  have hr := e_1445_5
  simp only [e_1445_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14045 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1445 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13891)) := by
  have c88623 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88624 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88625 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88626 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88627 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88628 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1445_6 := arithEq_of_rows h (row := 1445) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_6
  simp only [← c88623, ← c88624, ← c88625] at e_1445_6
  have e_1445_7 := arithEq_of_rows h (row := 1445) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_7
  simp only [← c88626, ← c88627, k2, ← c88628] at e_1445_7
  have hr := e_1445_7
  simp only [e_1445_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14046 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1445 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13892)) := by
  have c88629 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88630 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88631 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88632 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88633 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88634 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1445_8 := arithEq_of_rows h (row := 1445) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_8
  simp only [← c88629, ← c88630, ← c88631] at e_1445_8
  have e_1445_9 := arithEq_of_rows h (row := 1445) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_9
  simp only [← c88632, ← c88633, k2, ← c88634] at e_1445_9
  have hr := e_1445_9
  simp only [e_1445_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14047 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1445 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13893)) := by
  have c88635 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88636 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88637 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88638 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88639 := (publicBatchWrapper86_copies2769 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88640 := (publicBatchWrapper86_copies2770 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1445_10 := arithEq_of_rows h (row := 1445) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_10
  simp only [← c88635, ← c88636, ← c88637] at e_1445_10
  have e_1445_11 := arithEq_of_rows h (row := 1445) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_11
  simp only [← c88638, ← c88639, k2, ← c88640] at e_1445_11
  have hr := e_1445_11
  simp only [e_1445_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14048 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1445 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13894)) := by
  have c88641 := (publicBatchWrapper86_copies2770 a h).2.1
  have c88642 := (publicBatchWrapper86_copies2770 a h).2.2.1
  have c88643 := (publicBatchWrapper86_copies2770 a h).2.2.2.1
  have c88644 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.1
  have c88645 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.1
  have c88646 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1445_12 := arithEq_of_rows h (row := 1445) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_12
  simp only [← c88641, ← c88642, ← c88643] at e_1445_12
  have e_1445_13 := arithEq_of_rows h (row := 1445) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_13
  simp only [← c88644, ← c88645, k2, ← c88646] at e_1445_13
  have hr := e_1445_13
  simp only [e_1445_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14049 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1445 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13895)) := by
  have c88647 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.1
  have c88648 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.1
  have c88649 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.1
  have c88650 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88651 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88652 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1445_14 := arithEq_of_rows h (row := 1445) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_14
  simp only [← c88647, ← c88648, ← c88649] at e_1445_14
  have e_1445_15 := arithEq_of_rows h (row := 1445) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_15
  simp only [← c88650, ← c88651, k2, ← c88652] at e_1445_15
  have hr := e_1445_15
  simp only [e_1445_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14050 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1445 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13896)) := by
  have c88653 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88654 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88655 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88656 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88657 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88658 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1445_16 := arithEq_of_rows h (row := 1445) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_16
  simp only [← c88653, ← c88654, ← c88655] at e_1445_16
  have e_1445_17 := arithEq_of_rows h (row := 1445) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_17
  simp only [← c88656, ← c88657, k2, ← c88658] at e_1445_17
  have hr := e_1445_17
  simp only [e_1445_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14051 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1445 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13897)) := by
  have c88659 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88660 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88661 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88662 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88663 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88664 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1445_18 := arithEq_of_rows h (row := 1445) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_18
  simp only [← c88659, ← c88660, ← c88661] at e_1445_18
  have e_1445_19 := arithEq_of_rows h (row := 1445) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1445_19
  simp only [← c88662, ← c88663, k2, ← c88664] at e_1445_19
  have hr := e_1445_19
  simp only [e_1445_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14052 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1446 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13898)) := by
  have c88665 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88666 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88667 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88668 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88669 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88670 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1446_0 := arithEq_of_rows h (row := 1446) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_0
  simp only [← c88665, ← c88666, ← c88667] at e_1446_0
  have e_1446_1 := arithEq_of_rows h (row := 1446) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_1
  simp only [← c88668, ← c88669, k2, ← c88670] at e_1446_1
  have hr := e_1446_1
  simp only [e_1446_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14053 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1446 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13899)) := by
  have c88671 := (publicBatchWrapper86_copies2770 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88672 := (publicBatchWrapper86_copies2771 a h).1
  have c88673 := (publicBatchWrapper86_copies2771 a h).2.1
  have c88674 := (publicBatchWrapper86_copies2771 a h).2.2.1
  have c88675 := (publicBatchWrapper86_copies2771 a h).2.2.2.1
  have c88676 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1446_2 := arithEq_of_rows h (row := 1446) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_2
  simp only [← c88671, ← c88672, ← c88673] at e_1446_2
  have e_1446_3 := arithEq_of_rows h (row := 1446) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_3
  simp only [← c88674, ← c88675, k2, ← c88676] at e_1446_3
  have hr := e_1446_3
  simp only [e_1446_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14054 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1446 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13900)) := by
  have c88677 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.1
  have c88678 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.1
  have c88679 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.1
  have c88680 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.1
  have c88681 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.1
  have c88682 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1446_4 := arithEq_of_rows h (row := 1446) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_4
  simp only [← c88677, ← c88678, ← c88679] at e_1446_4
  have e_1446_5 := arithEq_of_rows h (row := 1446) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_5
  simp only [← c88680, ← c88681, k2, ← c88682] at e_1446_5
  have hr := e_1446_5
  simp only [e_1446_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14055 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1446 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13901)) := by
  have c88683 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88684 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88685 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88686 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88687 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88688 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1446_6 := arithEq_of_rows h (row := 1446) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_6
  simp only [← c88683, ← c88684, ← c88685] at e_1446_6
  have e_1446_7 := arithEq_of_rows h (row := 1446) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_7
  simp only [← c88686, ← c88687, k2, ← c88688] at e_1446_7
  have hr := e_1446_7
  simp only [e_1446_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14056 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1446 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13902)) := by
  have c88689 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88690 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88691 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88692 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88693 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88694 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1446_8 := arithEq_of_rows h (row := 1446) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_8
  simp only [← c88689, ← c88690, ← c88691] at e_1446_8
  have e_1446_9 := arithEq_of_rows h (row := 1446) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_9
  simp only [← c88692, ← c88693, k2, ← c88694] at e_1446_9
  have hr := e_1446_9
  simp only [e_1446_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14057 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1446 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13903)) := by
  have c88695 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88696 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88697 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88698 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88699 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88700 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1446_10 := arithEq_of_rows h (row := 1446) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_10
  simp only [← c88695, ← c88696, ← c88697] at e_1446_10
  have e_1446_11 := arithEq_of_rows h (row := 1446) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_11
  simp only [← c88698, ← c88699, k2, ← c88700] at e_1446_11
  have hr := e_1446_11
  simp only [e_1446_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14058 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1446 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13904)) := by
  have c88701 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88702 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88703 := (publicBatchWrapper86_copies2771 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88704 := (publicBatchWrapper86_copies2772 a h).1
  have c88705 := (publicBatchWrapper86_copies2772 a h).2.1
  have c88706 := (publicBatchWrapper86_copies2772 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1446_12 := arithEq_of_rows h (row := 1446) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_12
  simp only [← c88701, ← c88702, ← c88703] at e_1446_12
  have e_1446_13 := arithEq_of_rows h (row := 1446) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_13
  simp only [← c88704, ← c88705, k2, ← c88706] at e_1446_13
  have hr := e_1446_13
  simp only [e_1446_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14059 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1446 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13905)) := by
  have c88707 := (publicBatchWrapper86_copies2772 a h).2.2.2.1
  have c88708 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.1
  have c88709 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.1
  have c88710 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.1
  have c88711 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.1
  have c88712 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1446_14 := arithEq_of_rows h (row := 1446) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_14
  simp only [← c88707, ← c88708, ← c88709] at e_1446_14
  have e_1446_15 := arithEq_of_rows h (row := 1446) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_15
  simp only [← c88710, ← c88711, k2, ← c88712] at e_1446_15
  have hr := e_1446_15
  simp only [e_1446_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14060 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1446 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13906)) := by
  have c88713 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.1
  have c88714 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88715 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88716 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88717 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88718 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1446_16 := arithEq_of_rows h (row := 1446) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_16
  simp only [← c88713, ← c88714, ← c88715] at e_1446_16
  have e_1446_17 := arithEq_of_rows h (row := 1446) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_17
  simp only [← c88716, ← c88717, k2, ← c88718] at e_1446_17
  have hr := e_1446_17
  simp only [e_1446_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14061 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1446 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13907)) := by
  have c88719 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88720 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88721 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88722 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88723 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88724 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1446_18 := arithEq_of_rows h (row := 1446) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_18
  simp only [← c88719, ← c88720, ← c88721] at e_1446_18
  have e_1446_19 := arithEq_of_rows h (row := 1446) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1446_19
  simp only [← c88722, ← c88723, k2, ← c88724] at e_1446_19
  have hr := e_1446_19
  simp only [e_1446_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14062 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1447 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13908)) := by
  have c88725 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88726 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88727 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88728 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88729 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88730 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1447_0 := arithEq_of_rows h (row := 1447) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_0
  simp only [← c88725, ← c88726, ← c88727] at e_1447_0
  have e_1447_1 := arithEq_of_rows h (row := 1447) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_1
  simp only [← c88728, ← c88729, k2, ← c88730] at e_1447_1
  have hr := e_1447_1
  simp only [e_1447_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14063 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1447 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13909)) := by
  have c88731 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88732 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88733 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88734 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88735 := (publicBatchWrapper86_copies2772 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88736 := (publicBatchWrapper86_copies2773 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1447_2 := arithEq_of_rows h (row := 1447) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_2
  simp only [← c88731, ← c88732, ← c88733] at e_1447_2
  have e_1447_3 := arithEq_of_rows h (row := 1447) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_3
  simp only [← c88734, ← c88735, k2, ← c88736] at e_1447_3
  have hr := e_1447_3
  simp only [e_1447_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14064 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1447 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13910)) := by
  have c88737 := (publicBatchWrapper86_copies2773 a h).2.1
  have c88738 := (publicBatchWrapper86_copies2773 a h).2.2.1
  have c88739 := (publicBatchWrapper86_copies2773 a h).2.2.2.1
  have c88740 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.1
  have c88741 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.1
  have c88742 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1447_4 := arithEq_of_rows h (row := 1447) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_4
  simp only [← c88737, ← c88738, ← c88739] at e_1447_4
  have e_1447_5 := arithEq_of_rows h (row := 1447) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_5
  simp only [← c88740, ← c88741, k2, ← c88742] at e_1447_5
  have hr := e_1447_5
  simp only [e_1447_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14065 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1447 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13911)) := by
  have c88743 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.1
  have c88744 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.1
  have c88745 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.1
  have c88746 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88747 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88748 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1447_6 := arithEq_of_rows h (row := 1447) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_6
  simp only [← c88743, ← c88744, ← c88745] at e_1447_6
  have e_1447_7 := arithEq_of_rows h (row := 1447) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_7
  simp only [← c88746, ← c88747, k2, ← c88748] at e_1447_7
  have hr := e_1447_7
  simp only [e_1447_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14066 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1447 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13912)) := by
  have c88749 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88750 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88751 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88752 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88753 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88754 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1447_8 := arithEq_of_rows h (row := 1447) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_8
  simp only [← c88749, ← c88750, ← c88751] at e_1447_8
  have e_1447_9 := arithEq_of_rows h (row := 1447) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_9
  simp only [← c88752, ← c88753, k2, ← c88754] at e_1447_9
  have hr := e_1447_9
  simp only [e_1447_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14067 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1447 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13913)) := by
  have c88755 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88756 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88757 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88758 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88759 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88760 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1447_10 := arithEq_of_rows h (row := 1447) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_10
  simp only [← c88755, ← c88756, ← c88757] at e_1447_10
  have e_1447_11 := arithEq_of_rows h (row := 1447) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_11
  simp only [← c88758, ← c88759, k2, ← c88760] at e_1447_11
  have hr := e_1447_11
  simp only [e_1447_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14068 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1447 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13914)) := by
  have c88761 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88762 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88763 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88764 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88765 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88766 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1447_12 := arithEq_of_rows h (row := 1447) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_12
  simp only [← c88761, ← c88762, ← c88763] at e_1447_12
  have e_1447_13 := arithEq_of_rows h (row := 1447) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_13
  simp only [← c88764, ← c88765, k2, ← c88766] at e_1447_13
  have hr := e_1447_13
  simp only [e_1447_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14069 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1447 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13915)) := by
  have c88767 := (publicBatchWrapper86_copies2773 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88768 := (publicBatchWrapper86_copies2774 a h).1
  have c88769 := (publicBatchWrapper86_copies2774 a h).2.1
  have c88770 := (publicBatchWrapper86_copies2774 a h).2.2.1
  have c88771 := (publicBatchWrapper86_copies2774 a h).2.2.2.1
  have c88772 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1447_14 := arithEq_of_rows h (row := 1447) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_14
  simp only [← c88767, ← c88768, ← c88769] at e_1447_14
  have e_1447_15 := arithEq_of_rows h (row := 1447) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_15
  simp only [← c88770, ← c88771, k2, ← c88772] at e_1447_15
  have hr := e_1447_15
  simp only [e_1447_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14070 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1447 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13916)) := by
  have c88773 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.1
  have c88774 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.1
  have c88775 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.1
  have c88776 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.1
  have c88777 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.1
  have c88778 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1447_16 := arithEq_of_rows h (row := 1447) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_16
  simp only [← c88773, ← c88774, ← c88775] at e_1447_16
  have e_1447_17 := arithEq_of_rows h (row := 1447) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_17
  simp only [← c88776, ← c88777, k2, ← c88778] at e_1447_17
  have hr := e_1447_17
  simp only [e_1447_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14071 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1447 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13917)) := by
  have c88779 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88780 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88781 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88782 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88783 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88784 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1447_18 := arithEq_of_rows h (row := 1447) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_18
  simp only [← c88779, ← c88780, ← c88781] at e_1447_18
  have e_1447_19 := arithEq_of_rows h (row := 1447) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1447_19
  simp only [← c88782, ← c88783, k2, ← c88784] at e_1447_19
  have hr := e_1447_19
  simp only [e_1447_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14072 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1448 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13918)) := by
  have c88785 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88786 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88787 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88788 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88789 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88790 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1448_0 := arithEq_of_rows h (row := 1448) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_0
  simp only [← c88785, ← c88786, ← c88787] at e_1448_0
  have e_1448_1 := arithEq_of_rows h (row := 1448) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_1
  simp only [← c88788, ← c88789, k2, ← c88790] at e_1448_1
  have hr := e_1448_1
  simp only [e_1448_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14073 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1448 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13919)) := by
  have c88791 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88792 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88793 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88794 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88795 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88796 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1448_2 := arithEq_of_rows h (row := 1448) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_2
  simp only [← c88791, ← c88792, ← c88793] at e_1448_2
  have e_1448_3 := arithEq_of_rows h (row := 1448) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_3
  simp only [← c88794, ← c88795, k2, ← c88796] at e_1448_3
  have hr := e_1448_3
  simp only [e_1448_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14074 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1448 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13920)) := by
  have c88797 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88798 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88799 := (publicBatchWrapper86_copies2774 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88800 := (publicBatchWrapper86_copies2775 a h).1
  have c88801 := (publicBatchWrapper86_copies2775 a h).2.1
  have c88802 := (publicBatchWrapper86_copies2775 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1448_4 := arithEq_of_rows h (row := 1448) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_4
  simp only [← c88797, ← c88798, ← c88799] at e_1448_4
  have e_1448_5 := arithEq_of_rows h (row := 1448) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_5
  simp only [← c88800, ← c88801, k2, ← c88802] at e_1448_5
  have hr := e_1448_5
  simp only [e_1448_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14075 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1448 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13921)) := by
  have c88803 := (publicBatchWrapper86_copies2775 a h).2.2.2.1
  have c88804 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.1
  have c88805 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.1
  have c88806 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.1
  have c88807 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.1
  have c88808 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1448_6 := arithEq_of_rows h (row := 1448) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_6
  simp only [← c88803, ← c88804, ← c88805] at e_1448_6
  have e_1448_7 := arithEq_of_rows h (row := 1448) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_7
  simp only [← c88806, ← c88807, k2, ← c88808] at e_1448_7
  have hr := e_1448_7
  simp only [e_1448_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14076 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1448 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13922)) := by
  have c88809 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.1
  have c88810 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88811 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88812 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88813 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88814 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1448_8 := arithEq_of_rows h (row := 1448) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_8
  simp only [← c88809, ← c88810, ← c88811] at e_1448_8
  have e_1448_9 := arithEq_of_rows h (row := 1448) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_9
  simp only [← c88812, ← c88813, k2, ← c88814] at e_1448_9
  have hr := e_1448_9
  simp only [e_1448_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14077 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1448 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13923)) := by
  have c88815 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88816 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88817 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88818 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88819 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88820 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1448_10 := arithEq_of_rows h (row := 1448) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_10
  simp only [← c88815, ← c88816, ← c88817] at e_1448_10
  have e_1448_11 := arithEq_of_rows h (row := 1448) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_11
  simp only [← c88818, ← c88819, k2, ← c88820] at e_1448_11
  have hr := e_1448_11
  simp only [e_1448_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14078 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1448 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13924)) := by
  have c88821 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88822 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88823 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88824 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88825 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88826 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1448_12 := arithEq_of_rows h (row := 1448) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_12
  simp only [← c88821, ← c88822, ← c88823] at e_1448_12
  have e_1448_13 := arithEq_of_rows h (row := 1448) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_13
  simp only [← c88824, ← c88825, k2, ← c88826] at e_1448_13
  have hr := e_1448_13
  simp only [e_1448_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14079 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1448 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13925)) := by
  have c88827 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88828 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88829 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88830 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88831 := (publicBatchWrapper86_copies2775 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88832 := (publicBatchWrapper86_copies2776 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1448_14 := arithEq_of_rows h (row := 1448) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_14
  simp only [← c88827, ← c88828, ← c88829] at e_1448_14
  have e_1448_15 := arithEq_of_rows h (row := 1448) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_15
  simp only [← c88830, ← c88831, k2, ← c88832] at e_1448_15
  have hr := e_1448_15
  simp only [e_1448_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14080 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1448 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13926)) := by
  have c88833 := (publicBatchWrapper86_copies2776 a h).2.1
  have c88834 := (publicBatchWrapper86_copies2776 a h).2.2.1
  have c88835 := (publicBatchWrapper86_copies2776 a h).2.2.2.1
  have c88836 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.1
  have c88837 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.1
  have c88838 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1448_16 := arithEq_of_rows h (row := 1448) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_16
  simp only [← c88833, ← c88834, ← c88835] at e_1448_16
  have e_1448_17 := arithEq_of_rows h (row := 1448) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_17
  simp only [← c88836, ← c88837, k2, ← c88838] at e_1448_17
  have hr := e_1448_17
  simp only [e_1448_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14081 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1448 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13927)) := by
  have c88839 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.1
  have c88840 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.1
  have c88841 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.1
  have c88842 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88843 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88844 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1448_18 := arithEq_of_rows h (row := 1448) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_18
  simp only [← c88839, ← c88840, ← c88841] at e_1448_18
  have e_1448_19 := arithEq_of_rows h (row := 1448) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1448_19
  simp only [← c88842, ← c88843, k2, ← c88844] at e_1448_19
  have hr := e_1448_19
  simp only [e_1448_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14082 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1449 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13928)) := by
  have c88845 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88846 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88847 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88848 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88849 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88850 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1449_0 := arithEq_of_rows h (row := 1449) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_0
  simp only [← c88845, ← c88846, ← c88847] at e_1449_0
  have e_1449_1 := arithEq_of_rows h (row := 1449) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_1
  simp only [← c88848, ← c88849, k2, ← c88850] at e_1449_1
  have hr := e_1449_1
  simp only [e_1449_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14083 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1449 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13929)) := by
  have c88851 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88852 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88853 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88854 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88855 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88856 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1449_2 := arithEq_of_rows h (row := 1449) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_2
  simp only [← c88851, ← c88852, ← c88853] at e_1449_2
  have e_1449_3 := arithEq_of_rows h (row := 1449) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_3
  simp only [← c88854, ← c88855, k2, ← c88856] at e_1449_3
  have hr := e_1449_3
  simp only [e_1449_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14084 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1449 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13930)) := by
  have c88857 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88858 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88859 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88860 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88861 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88862 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1449_4 := arithEq_of_rows h (row := 1449) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_4
  simp only [← c88857, ← c88858, ← c88859] at e_1449_4
  have e_1449_5 := arithEq_of_rows h (row := 1449) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_5
  simp only [← c88860, ← c88861, k2, ← c88862] at e_1449_5
  have hr := e_1449_5
  simp only [e_1449_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14085 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1449 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13931)) := by
  have c88863 := (publicBatchWrapper86_copies2776 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88864 := (publicBatchWrapper86_copies2777 a h).1
  have c88865 := (publicBatchWrapper86_copies2777 a h).2.1
  have c88866 := (publicBatchWrapper86_copies2777 a h).2.2.1
  have c88867 := (publicBatchWrapper86_copies2777 a h).2.2.2.1
  have c88868 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1449_6 := arithEq_of_rows h (row := 1449) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_6
  simp only [← c88863, ← c88864, ← c88865] at e_1449_6
  have e_1449_7 := arithEq_of_rows h (row := 1449) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_7
  simp only [← c88866, ← c88867, k2, ← c88868] at e_1449_7
  have hr := e_1449_7
  simp only [e_1449_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14086 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1449 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13932)) := by
  have c88869 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.1
  have c88870 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.1
  have c88871 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.1
  have c88872 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.1
  have c88873 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.1
  have c88874 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1449_8 := arithEq_of_rows h (row := 1449) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_8
  simp only [← c88869, ← c88870, ← c88871] at e_1449_8
  have e_1449_9 := arithEq_of_rows h (row := 1449) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_9
  simp only [← c88872, ← c88873, k2, ← c88874] at e_1449_9
  have hr := e_1449_9
  simp only [e_1449_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14087 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1449 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13933)) := by
  have c88875 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88876 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88877 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88878 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88879 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88880 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1449_10 := arithEq_of_rows h (row := 1449) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_10
  simp only [← c88875, ← c88876, ← c88877] at e_1449_10
  have e_1449_11 := arithEq_of_rows h (row := 1449) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_11
  simp only [← c88878, ← c88879, k2, ← c88880] at e_1449_11
  have hr := e_1449_11
  simp only [e_1449_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14088 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1449 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13934)) := by
  have c88881 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88882 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88883 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88884 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88885 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88886 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1449_12 := arithEq_of_rows h (row := 1449) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_12
  simp only [← c88881, ← c88882, ← c88883] at e_1449_12
  have e_1449_13 := arithEq_of_rows h (row := 1449) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_13
  simp only [← c88884, ← c88885, k2, ← c88886] at e_1449_13
  have hr := e_1449_13
  simp only [e_1449_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14089 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1449 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13935)) := by
  have c88887 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88888 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88889 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88890 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88891 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88892 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1449_14 := arithEq_of_rows h (row := 1449) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_14
  simp only [← c88887, ← c88888, ← c88889] at e_1449_14
  have e_1449_15 := arithEq_of_rows h (row := 1449) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_15
  simp only [← c88890, ← c88891, k2, ← c88892] at e_1449_15
  have hr := e_1449_15
  simp only [e_1449_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14090 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1449 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13936)) := by
  have c88893 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88894 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88895 := (publicBatchWrapper86_copies2777 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88896 := (publicBatchWrapper86_copies2778 a h).1
  have c88897 := (publicBatchWrapper86_copies2778 a h).2.1
  have c88898 := (publicBatchWrapper86_copies2778 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1449_16 := arithEq_of_rows h (row := 1449) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_16
  simp only [← c88893, ← c88894, ← c88895] at e_1449_16
  have e_1449_17 := arithEq_of_rows h (row := 1449) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_17
  simp only [← c88896, ← c88897, k2, ← c88898] at e_1449_17
  have hr := e_1449_17
  simp only [e_1449_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14091 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1449 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13937)) := by
  have c88899 := (publicBatchWrapper86_copies2778 a h).2.2.2.1
  have c88900 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.1
  have c88901 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.1
  have c88902 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.1
  have c88903 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.1
  have c88904 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1449_18 := arithEq_of_rows h (row := 1449) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_18
  simp only [← c88899, ← c88900, ← c88901] at e_1449_18
  have e_1449_19 := arithEq_of_rows h (row := 1449) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1449_19
  simp only [← c88902, ← c88903, k2, ← c88904] at e_1449_19
  have hr := e_1449_19
  simp only [e_1449_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14092 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1450 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13938)) := by
  have c88905 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.1
  have c88906 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88907 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88908 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88909 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88910 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1450_0 := arithEq_of_rows h (row := 1450) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_0
  simp only [← c88905, ← c88906, ← c88907] at e_1450_0
  have e_1450_1 := arithEq_of_rows h (row := 1450) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_1
  simp only [← c88908, ← c88909, k2, ← c88910] at e_1450_1
  have hr := e_1450_1
  simp only [e_1450_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14093 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1450 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13939)) := by
  have c88911 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88912 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88913 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88914 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88915 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88916 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1450_2 := arithEq_of_rows h (row := 1450) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_2
  simp only [← c88911, ← c88912, ← c88913] at e_1450_2
  have e_1450_3 := arithEq_of_rows h (row := 1450) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_3
  simp only [← c88914, ← c88915, k2, ← c88916] at e_1450_3
  have hr := e_1450_3
  simp only [e_1450_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14094 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1450 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13940)) := by
  have c88917 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88918 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88919 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88920 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88921 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88922 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1450_4 := arithEq_of_rows h (row := 1450) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_4
  simp only [← c88917, ← c88918, ← c88919] at e_1450_4
  have e_1450_5 := arithEq_of_rows h (row := 1450) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_5
  simp only [← c88920, ← c88921, k2, ← c88922] at e_1450_5
  have hr := e_1450_5
  simp only [e_1450_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14095 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1450 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13941)) := by
  have c88923 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88924 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88925 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88926 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88927 := (publicBatchWrapper86_copies2778 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88928 := (publicBatchWrapper86_copies2779 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1450_6 := arithEq_of_rows h (row := 1450) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_6
  simp only [← c88923, ← c88924, ← c88925] at e_1450_6
  have e_1450_7 := arithEq_of_rows h (row := 1450) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_7
  simp only [← c88926, ← c88927, k2, ← c88928] at e_1450_7
  have hr := e_1450_7
  simp only [e_1450_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14096 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1450 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13942)) := by
  have c88929 := (publicBatchWrapper86_copies2779 a h).2.1
  have c88930 := (publicBatchWrapper86_copies2779 a h).2.2.1
  have c88931 := (publicBatchWrapper86_copies2779 a h).2.2.2.1
  have c88932 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.1
  have c88933 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.1
  have c88934 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1450_8 := arithEq_of_rows h (row := 1450) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_8
  simp only [← c88929, ← c88930, ← c88931] at e_1450_8
  have e_1450_9 := arithEq_of_rows h (row := 1450) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_9
  simp only [← c88932, ← c88933, k2, ← c88934] at e_1450_9
  have hr := e_1450_9
  simp only [e_1450_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14097 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1450 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13943)) := by
  have c88935 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.1
  have c88936 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.1
  have c88937 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.1
  have c88938 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.1
  have c88939 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88940 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1450_10 := arithEq_of_rows h (row := 1450) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_10
  simp only [← c88935, ← c88936, ← c88937] at e_1450_10
  have e_1450_11 := arithEq_of_rows h (row := 1450) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_11
  simp only [← c88938, ← c88939, k2, ← c88940] at e_1450_11
  have hr := e_1450_11
  simp only [e_1450_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14098 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1450 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13944)) := by
  have c88941 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88942 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88943 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88944 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88945 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88946 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1450_12 := arithEq_of_rows h (row := 1450) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_12
  simp only [← c88941, ← c88942, ← c88943] at e_1450_12
  have e_1450_13 := arithEq_of_rows h (row := 1450) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_13
  simp only [← c88944, ← c88945, k2, ← c88946] at e_1450_13
  have hr := e_1450_13
  simp only [e_1450_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14099 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1450 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13945)) := by
  have c88947 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88948 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88949 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88950 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88951 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88952 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1450_14 := arithEq_of_rows h (row := 1450) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_14
  simp only [← c88947, ← c88948, ← c88949] at e_1450_14
  have e_1450_15 := arithEq_of_rows h (row := 1450) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_15
  simp only [← c88950, ← c88951, k2, ← c88952] at e_1450_15
  have hr := e_1450_15
  simp only [e_1450_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14100 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1450 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13946)) := by
  have c88953 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88954 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88955 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88956 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88957 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88958 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1450_16 := arithEq_of_rows h (row := 1450) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_16
  simp only [← c88953, ← c88954, ← c88955] at e_1450_16
  have e_1450_17 := arithEq_of_rows h (row := 1450) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_17
  simp only [← c88956, ← c88957, k2, ← c88958] at e_1450_17
  have hr := e_1450_17
  simp only [e_1450_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14101 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1450 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13947)) := by
  have c88959 := (publicBatchWrapper86_copies2779 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88960 := (publicBatchWrapper86_copies2780 a h).1
  have c88961 := (publicBatchWrapper86_copies2780 a h).2.1
  have c88962 := (publicBatchWrapper86_copies2780 a h).2.2.1
  have c88963 := (publicBatchWrapper86_copies2780 a h).2.2.2.1
  have c88964 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1450_18 := arithEq_of_rows h (row := 1450) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_18
  simp only [← c88959, ← c88960, ← c88961] at e_1450_18
  have e_1450_19 := arithEq_of_rows h (row := 1450) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1450_19
  simp only [← c88962, ← c88963, k2, ← c88964] at e_1450_19
  have hr := e_1450_19
  simp only [e_1450_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14102 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1451 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13948)) := by
  have c88965 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.1
  have c88966 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.1
  have c88967 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.1
  have c88968 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.1
  have c88969 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.1
  have c88970 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1451_0 := arithEq_of_rows h (row := 1451) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_0
  simp only [← c88965, ← c88966, ← c88967] at e_1451_0
  have e_1451_1 := arithEq_of_rows h (row := 1451) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_1
  simp only [← c88968, ← c88969, k2, ← c88970] at e_1451_1
  have hr := e_1451_1
  simp only [e_1451_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14103 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1451 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13949)) := by
  have c88971 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c88972 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88973 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88974 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88975 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88976 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1451_2 := arithEq_of_rows h (row := 1451) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_2
  simp only [← c88971, ← c88972, ← c88973] at e_1451_2
  have e_1451_3 := arithEq_of_rows h (row := 1451) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_3
  simp only [← c88974, ← c88975, k2, ← c88976] at e_1451_3
  have hr := e_1451_3
  simp only [e_1451_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14104 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1451 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13950)) := by
  have c88977 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88978 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88979 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88980 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88981 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88982 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1451_4 := arithEq_of_rows h (row := 1451) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_4
  simp only [← c88977, ← c88978, ← c88979] at e_1451_4
  have e_1451_5 := arithEq_of_rows h (row := 1451) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_5
  simp only [← c88980, ← c88981, k2, ← c88982] at e_1451_5
  have hr := e_1451_5
  simp only [e_1451_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14105 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1451 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13951)) := by
  have c88983 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88984 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88985 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88986 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88987 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88988 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1451_6 := arithEq_of_rows h (row := 1451) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_6
  simp only [← c88983, ← c88984, ← c88985] at e_1451_6
  have e_1451_7 := arithEq_of_rows h (row := 1451) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_7
  simp only [← c88986, ← c88987, k2, ← c88988] at e_1451_7
  have hr := e_1451_7
  simp only [e_1451_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14106 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1451 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13952)) := by
  have c88989 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88990 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c88991 := (publicBatchWrapper86_copies2780 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c88992 := (publicBatchWrapper86_copies2781 a h).1
  have c88993 := (publicBatchWrapper86_copies2781 a h).2.1
  have c88994 := (publicBatchWrapper86_copies2781 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1451_8 := arithEq_of_rows h (row := 1451) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_8
  simp only [← c88989, ← c88990, ← c88991] at e_1451_8
  have e_1451_9 := arithEq_of_rows h (row := 1451) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_9
  simp only [← c88992, ← c88993, k2, ← c88994] at e_1451_9
  have hr := e_1451_9
  simp only [e_1451_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14107 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1451 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13953)) := by
  have c88995 := (publicBatchWrapper86_copies2781 a h).2.2.2.1
  have c88996 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.1
  have c88997 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.1
  have c88998 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.1
  have c88999 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.1
  have c89000 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1451_10 := arithEq_of_rows h (row := 1451) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_10
  simp only [← c88995, ← c88996, ← c88997] at e_1451_10
  have e_1451_11 := arithEq_of_rows h (row := 1451) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_11
  simp only [← c88998, ← c88999, k2, ← c89000] at e_1451_11
  have hr := e_1451_11
  simp only [e_1451_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14108 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1451 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13954)) := by
  have c89001 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.1
  have c89002 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89003 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89004 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89005 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89006 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1451_12 := arithEq_of_rows h (row := 1451) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_12
  simp only [← c89001, ← c89002, ← c89003] at e_1451_12
  have e_1451_13 := arithEq_of_rows h (row := 1451) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_13
  simp only [← c89004, ← c89005, k2, ← c89006] at e_1451_13
  have hr := e_1451_13
  simp only [e_1451_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14109 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1451 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13955)) := by
  have c89007 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89008 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89009 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89010 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89011 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89012 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1451_14 := arithEq_of_rows h (row := 1451) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_14
  simp only [← c89007, ← c89008, ← c89009] at e_1451_14
  have e_1451_15 := arithEq_of_rows h (row := 1451) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_15
  simp only [← c89010, ← c89011, k2, ← c89012] at e_1451_15
  have hr := e_1451_15
  simp only [e_1451_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14110 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1451 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13956)) := by
  have c89013 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89014 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89015 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89016 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89017 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89018 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1451_16 := arithEq_of_rows h (row := 1451) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_16
  simp only [← c89013, ← c89014, ← c89015] at e_1451_16
  have e_1451_17 := arithEq_of_rows h (row := 1451) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_17
  simp only [← c89016, ← c89017, k2, ← c89018] at e_1451_17
  have hr := e_1451_17
  simp only [e_1451_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14111 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1451 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13957)) := by
  have c89019 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89020 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89021 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89022 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89023 := (publicBatchWrapper86_copies2781 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89024 := (publicBatchWrapper86_copies2782 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1451_18 := arithEq_of_rows h (row := 1451) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_18
  simp only [← c89019, ← c89020, ← c89021] at e_1451_18
  have e_1451_19 := arithEq_of_rows h (row := 1451) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1451_19
  simp only [← c89022, ← c89023, k2, ← c89024] at e_1451_19
  have hr := e_1451_19
  simp only [e_1451_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14112 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1452 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13958)) := by
  have c89025 := (publicBatchWrapper86_copies2782 a h).2.1
  have c89026 := (publicBatchWrapper86_copies2782 a h).2.2.1
  have c89027 := (publicBatchWrapper86_copies2782 a h).2.2.2.1
  have c89028 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.1
  have c89029 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.1
  have c89030 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1452_0 := arithEq_of_rows h (row := 1452) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_0
  simp only [← c89025, ← c89026, ← c89027] at e_1452_0
  have e_1452_1 := arithEq_of_rows h (row := 1452) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_1
  simp only [← c89028, ← c89029, k2, ← c89030] at e_1452_1
  have hr := e_1452_1
  simp only [e_1452_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14113 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1452 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13959)) := by
  have c89031 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.1
  have c89032 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.1
  have c89033 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.1
  have c89034 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89035 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89036 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1452_2 := arithEq_of_rows h (row := 1452) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_2
  simp only [← c89031, ← c89032, ← c89033] at e_1452_2
  have e_1452_3 := arithEq_of_rows h (row := 1452) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_3
  simp only [← c89034, ← c89035, k2, ← c89036] at e_1452_3
  have hr := e_1452_3
  simp only [e_1452_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14114 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1452 23) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13960)) := by
  have c89037 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89038 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89039 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89040 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89041 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89042 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1452_4 := arithEq_of_rows h (row := 1452) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_4
  simp only [← c89037, ← c89038, ← c89039] at e_1452_4
  have e_1452_5 := arithEq_of_rows h (row := 1452) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_5
  simp only [← c89040, ← c89041, k2, ← c89042] at e_1452_5
  have hr := e_1452_5
  simp only [e_1452_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14115 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1452 31) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13961)) := by
  have c89043 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89044 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89045 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89046 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89047 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89048 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1452_6 := arithEq_of_rows h (row := 1452) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_6
  simp only [← c89043, ← c89044, ← c89045] at e_1452_6
  have e_1452_7 := arithEq_of_rows h (row := 1452) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_7
  simp only [← c89046, ← c89047, k2, ← c89048] at e_1452_7
  have hr := e_1452_7
  simp only [e_1452_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14116 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1452 39) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13962)) := by
  have c89049 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89050 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89051 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89052 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89053 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89054 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1452_8 := arithEq_of_rows h (row := 1452) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_8
  simp only [← c89049, ← c89050, ← c89051] at e_1452_8
  have e_1452_9 := arithEq_of_rows h (row := 1452) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_9
  simp only [← c89052, ← c89053, k2, ← c89054] at e_1452_9
  have hr := e_1452_9
  simp only [e_1452_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14117 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1452 47) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13963)) := by
  have c89055 := (publicBatchWrapper86_copies2782 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89056 := (publicBatchWrapper86_copies2783 a h).1
  have c89057 := (publicBatchWrapper86_copies2783 a h).2.1
  have c89058 := (publicBatchWrapper86_copies2783 a h).2.2.1
  have c89059 := (publicBatchWrapper86_copies2783 a h).2.2.2.1
  have c89060 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1452_10 := arithEq_of_rows h (row := 1452) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_10
  simp only [← c89055, ← c89056, ← c89057] at e_1452_10
  have e_1452_11 := arithEq_of_rows h (row := 1452) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_11
  simp only [← c89058, ← c89059, k2, ← c89060] at e_1452_11
  have hr := e_1452_11
  simp only [e_1452_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14118 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1452 55) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13964)) := by
  have c89061 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.1
  have c89062 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.1
  have c89063 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.1
  have c89064 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.1
  have c89065 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.1
  have c89066 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1452_12 := arithEq_of_rows h (row := 1452) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_12
  simp only [← c89061, ← c89062, ← c89063] at e_1452_12
  have e_1452_13 := arithEq_of_rows h (row := 1452) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_13
  simp only [← c89064, ← c89065, k2, ← c89066] at e_1452_13
  have hr := e_1452_13
  simp only [e_1452_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14119 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1452 63) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13965)) := by
  have c89067 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89068 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89069 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89070 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89071 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89072 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1452_14 := arithEq_of_rows h (row := 1452) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_14
  simp only [← c89067, ← c89068, ← c89069] at e_1452_14
  have e_1452_15 := arithEq_of_rows h (row := 1452) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_15
  simp only [← c89070, ← c89071, k2, ← c89072] at e_1452_15
  have hr := e_1452_15
  simp only [e_1452_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14120 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1452 71) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13966)) := by
  have c89073 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89074 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89075 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89076 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89077 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89078 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1452_16 := arithEq_of_rows h (row := 1452) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_16
  simp only [← c89073, ← c89074, ← c89075] at e_1452_16
  have e_1452_17 := arithEq_of_rows h (row := 1452) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_17
  simp only [← c89076, ← c89077, k2, ← c89078] at e_1452_17
  have hr := e_1452_17
  simp only [e_1452_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14121 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1452 79) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13967)) := by
  have c89079 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89080 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89081 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89082 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89083 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89084 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1452_18 := arithEq_of_rows h (row := 1452) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_18
  simp only [← c89079, ← c89080, ← c89081] at e_1452_18
  have e_1452_19 := arithEq_of_rows h (row := 1452) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1452_19
  simp only [← c89082, ← c89083, k2, ← c89084] at e_1452_19
  have hr := e_1452_19
  simp only [e_1452_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14122 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1453 7) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13968)) := by
  have c89085 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89086 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89087 := (publicBatchWrapper86_copies2783 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89088 := (publicBatchWrapper86_copies2784 a h).1
  have c89089 := (publicBatchWrapper86_copies2784 a h).2.1
  have c89090 := (publicBatchWrapper86_copies2784 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1453_0 := arithEq_of_rows h (row := 1453) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_0
  simp only [← c89085, ← c89086, ← c89087] at e_1453_0
  have e_1453_1 := arithEq_of_rows h (row := 1453) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_1
  simp only [← c89088, ← c89089, k2, ← c89090] at e_1453_1
  have hr := e_1453_1
  simp only [e_1453_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14123 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1453 15) = bselect (a (.wire 68 23)) (a (.virt 200)) (a (.virt 13969)) := by
  have c89091 := (publicBatchWrapper86_copies2784 a h).2.2.2.1
  have c89092 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.1
  have c89093 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.1
  have c89094 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.1
  have c89095 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.1
  have c89096 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1453_2 := arithEq_of_rows h (row := 1453) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_2
  simp only [← c89091, ← c89092, ← c89093] at e_1453_2
  have e_1453_3 := arithEq_of_rows h (row := 1453) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_3
  simp only [← c89094, ← c89095, k2, ← c89096] at e_1453_3
  have hr := e_1453_3
  simp only [e_1453_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14124 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1453 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14010)) := by
  have c89097 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.1
  have c89098 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89099 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89100 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89101 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89102 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1453_4 := arithEq_of_rows h (row := 1453) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_4
  simp only [← c89097, ← c89098, ← c89099] at e_1453_4
  have e_1453_5 := arithEq_of_rows h (row := 1453) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_5
  simp only [← c89100, ← c89101, k2, ← c89102] at e_1453_5
  have hr := e_1453_5
  simp only [e_1453_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14125 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1453 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14011)) := by
  have c89103 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89104 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89105 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89106 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89107 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89108 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1453_6 := arithEq_of_rows h (row := 1453) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_6
  simp only [← c89103, ← c89104, ← c89105] at e_1453_6
  have e_1453_7 := arithEq_of_rows h (row := 1453) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_7
  simp only [← c89106, ← c89107, k2, ← c89108] at e_1453_7
  have hr := e_1453_7
  simp only [e_1453_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14126 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1453 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14012)) := by
  have c89109 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89110 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89111 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89112 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89113 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89114 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1453_8 := arithEq_of_rows h (row := 1453) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_8
  simp only [← c89109, ← c89110, ← c89111] at e_1453_8
  have e_1453_9 := arithEq_of_rows h (row := 1453) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_9
  simp only [← c89112, ← c89113, k2, ← c89114] at e_1453_9
  have hr := e_1453_9
  simp only [e_1453_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14127 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1453 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14013)) := by
  have c89115 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89116 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89117 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89118 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89119 := (publicBatchWrapper86_copies2784 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89120 := (publicBatchWrapper86_copies2785 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1453_10 := arithEq_of_rows h (row := 1453) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_10
  simp only [← c89115, ← c89116, ← c89117] at e_1453_10
  have e_1453_11 := arithEq_of_rows h (row := 1453) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_11
  simp only [← c89118, ← c89119, k2, ← c89120] at e_1453_11
  have hr := e_1453_11
  simp only [e_1453_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14128 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1453 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14014)) := by
  have c89121 := (publicBatchWrapper86_copies2785 a h).2.1
  have c89122 := (publicBatchWrapper86_copies2785 a h).2.2.1
  have c89123 := (publicBatchWrapper86_copies2785 a h).2.2.2.1
  have c89124 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.1
  have c89125 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.1
  have c89126 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1453_12 := arithEq_of_rows h (row := 1453) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_12
  simp only [← c89121, ← c89122, ← c89123] at e_1453_12
  have e_1453_13 := arithEq_of_rows h (row := 1453) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_13
  simp only [← c89124, ← c89125, k2, ← c89126] at e_1453_13
  have hr := e_1453_13
  simp only [e_1453_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14129 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1453 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14015)) := by
  have c89127 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.1
  have c89128 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.1
  have c89129 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.1
  have c89130 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89131 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89132 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1453_14 := arithEq_of_rows h (row := 1453) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_14
  simp only [← c89127, ← c89128, ← c89129] at e_1453_14
  have e_1453_15 := arithEq_of_rows h (row := 1453) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_15
  simp only [← c89130, ← c89131, k2, ← c89132] at e_1453_15
  have hr := e_1453_15
  simp only [e_1453_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14130 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1453 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14016)) := by
  have c89133 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89134 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89135 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89136 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89137 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89138 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1453_16 := arithEq_of_rows h (row := 1453) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_16
  simp only [← c89133, ← c89134, ← c89135] at e_1453_16
  have e_1453_17 := arithEq_of_rows h (row := 1453) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_17
  simp only [← c89136, ← c89137, k2, ← c89138] at e_1453_17
  have hr := e_1453_17
  simp only [e_1453_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14131 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1453 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14017)) := by
  have c89139 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89140 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89141 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89142 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89143 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89144 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1453_18 := arithEq_of_rows h (row := 1453) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_18
  simp only [← c89139, ← c89140, ← c89141] at e_1453_18
  have e_1453_19 := arithEq_of_rows h (row := 1453) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1453_19
  simp only [← c89142, ← c89143, k2, ← c89144] at e_1453_19
  have hr := e_1453_19
  simp only [e_1453_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14132 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1454 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14018)) := by
  have c89145 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89146 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89147 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89148 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89149 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89150 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1454_0 := arithEq_of_rows h (row := 1454) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_0
  simp only [← c89145, ← c89146, ← c89147] at e_1454_0
  have e_1454_1 := arithEq_of_rows h (row := 1454) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_1
  simp only [← c89148, ← c89149, k2, ← c89150] at e_1454_1
  have hr := e_1454_1
  simp only [e_1454_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14133 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1454 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14019)) := by
  have c89151 := (publicBatchWrapper86_copies2785 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89152 := (publicBatchWrapper86_copies2786 a h).1
  have c89153 := (publicBatchWrapper86_copies2786 a h).2.1
  have c89154 := (publicBatchWrapper86_copies2786 a h).2.2.1
  have c89155 := (publicBatchWrapper86_copies2786 a h).2.2.2.1
  have c89156 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1454_2 := arithEq_of_rows h (row := 1454) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_2
  simp only [← c89151, ← c89152, ← c89153] at e_1454_2
  have e_1454_3 := arithEq_of_rows h (row := 1454) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_3
  simp only [← c89154, ← c89155, k2, ← c89156] at e_1454_3
  have hr := e_1454_3
  simp only [e_1454_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14134 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1454 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14020)) := by
  have c89157 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.1
  have c89158 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.1
  have c89159 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.1
  have c89160 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.1
  have c89161 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.1
  have c89162 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1454_4 := arithEq_of_rows h (row := 1454) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_4
  simp only [← c89157, ← c89158, ← c89159] at e_1454_4
  have e_1454_5 := arithEq_of_rows h (row := 1454) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_5
  simp only [← c89160, ← c89161, k2, ← c89162] at e_1454_5
  have hr := e_1454_5
  simp only [e_1454_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14135 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1454 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14021)) := by
  have c89163 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89164 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89165 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89166 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89167 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89168 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1454_6 := arithEq_of_rows h (row := 1454) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_6
  simp only [← c89163, ← c89164, ← c89165] at e_1454_6
  have e_1454_7 := arithEq_of_rows h (row := 1454) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_7
  simp only [← c89166, ← c89167, k2, ← c89168] at e_1454_7
  have hr := e_1454_7
  simp only [e_1454_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14136 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1454 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14022)) := by
  have c89169 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89170 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89171 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89172 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89173 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89174 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1454_8 := arithEq_of_rows h (row := 1454) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_8
  simp only [← c89169, ← c89170, ← c89171] at e_1454_8
  have e_1454_9 := arithEq_of_rows h (row := 1454) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_9
  simp only [← c89172, ← c89173, k2, ← c89174] at e_1454_9
  have hr := e_1454_9
  simp only [e_1454_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14137 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1454 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14023)) := by
  have c89175 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89176 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89177 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89178 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89179 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89180 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1454_10 := arithEq_of_rows h (row := 1454) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_10
  simp only [← c89175, ← c89176, ← c89177] at e_1454_10
  have e_1454_11 := arithEq_of_rows h (row := 1454) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_11
  simp only [← c89178, ← c89179, k2, ← c89180] at e_1454_11
  have hr := e_1454_11
  simp only [e_1454_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14138 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1454 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14024)) := by
  have c89181 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89182 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89183 := (publicBatchWrapper86_copies2786 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89184 := (publicBatchWrapper86_copies2787 a h).1
  have c89185 := (publicBatchWrapper86_copies2787 a h).2.1
  have c89186 := (publicBatchWrapper86_copies2787 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1454_12 := arithEq_of_rows h (row := 1454) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_12
  simp only [← c89181, ← c89182, ← c89183] at e_1454_12
  have e_1454_13 := arithEq_of_rows h (row := 1454) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_13
  simp only [← c89184, ← c89185, k2, ← c89186] at e_1454_13
  have hr := e_1454_13
  simp only [e_1454_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14139 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1454 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14025)) := by
  have c89187 := (publicBatchWrapper86_copies2787 a h).2.2.2.1
  have c89188 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.1
  have c89189 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.1
  have c89190 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.1
  have c89191 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.1
  have c89192 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1454_14 := arithEq_of_rows h (row := 1454) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_14
  simp only [← c89187, ← c89188, ← c89189] at e_1454_14
  have e_1454_15 := arithEq_of_rows h (row := 1454) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_15
  simp only [← c89190, ← c89191, k2, ← c89192] at e_1454_15
  have hr := e_1454_15
  simp only [e_1454_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14140 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1454 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14026)) := by
  have c89193 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.1
  have c89194 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89195 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89196 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89197 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89198 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1454_16 := arithEq_of_rows h (row := 1454) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_16
  simp only [← c89193, ← c89194, ← c89195] at e_1454_16
  have e_1454_17 := arithEq_of_rows h (row := 1454) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_17
  simp only [← c89196, ← c89197, k2, ← c89198] at e_1454_17
  have hr := e_1454_17
  simp only [e_1454_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14141 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1454 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14027)) := by
  have c89199 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89200 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89201 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89202 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89203 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89204 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1454_18 := arithEq_of_rows h (row := 1454) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_18
  simp only [← c89199, ← c89200, ← c89201] at e_1454_18
  have e_1454_19 := arithEq_of_rows h (row := 1454) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1454_19
  simp only [← c89202, ← c89203, k2, ← c89204] at e_1454_19
  have hr := e_1454_19
  simp only [e_1454_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14142 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1455 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14028)) := by
  have c89205 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89206 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89207 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89208 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89209 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89210 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1455_0 := arithEq_of_rows h (row := 1455) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_0
  simp only [← c89205, ← c89206, ← c89207] at e_1455_0
  have e_1455_1 := arithEq_of_rows h (row := 1455) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_1
  simp only [← c89208, ← c89209, k2, ← c89210] at e_1455_1
  have hr := e_1455_1
  simp only [e_1455_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14143 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1455 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14029)) := by
  have c89211 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89212 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89213 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89214 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89215 := (publicBatchWrapper86_copies2787 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89216 := (publicBatchWrapper86_copies2788 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1455_2 := arithEq_of_rows h (row := 1455) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_2
  simp only [← c89211, ← c89212, ← c89213] at e_1455_2
  have e_1455_3 := arithEq_of_rows h (row := 1455) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_3
  simp only [← c89214, ← c89215, k2, ← c89216] at e_1455_3
  have hr := e_1455_3
  simp only [e_1455_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14144 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1455 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14030)) := by
  have c89217 := (publicBatchWrapper86_copies2788 a h).2.1
  have c89218 := (publicBatchWrapper86_copies2788 a h).2.2.1
  have c89219 := (publicBatchWrapper86_copies2788 a h).2.2.2.1
  have c89220 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.1
  have c89221 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.1
  have c89222 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1455_4 := arithEq_of_rows h (row := 1455) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_4
  simp only [← c89217, ← c89218, ← c89219] at e_1455_4
  have e_1455_5 := arithEq_of_rows h (row := 1455) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_5
  simp only [← c89220, ← c89221, k2, ← c89222] at e_1455_5
  have hr := e_1455_5
  simp only [e_1455_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14145 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1455 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14031)) := by
  have c89223 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.1
  have c89224 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.1
  have c89225 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.1
  have c89226 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89227 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89228 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1455_6 := arithEq_of_rows h (row := 1455) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_6
  simp only [← c89223, ← c89224, ← c89225] at e_1455_6
  have e_1455_7 := arithEq_of_rows h (row := 1455) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_7
  simp only [← c89226, ← c89227, k2, ← c89228] at e_1455_7
  have hr := e_1455_7
  simp only [e_1455_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14146 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1455 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14032)) := by
  have c89229 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89230 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89231 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89232 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89233 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89234 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1455_8 := arithEq_of_rows h (row := 1455) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_8
  simp only [← c89229, ← c89230, ← c89231] at e_1455_8
  have e_1455_9 := arithEq_of_rows h (row := 1455) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_9
  simp only [← c89232, ← c89233, k2, ← c89234] at e_1455_9
  have hr := e_1455_9
  simp only [e_1455_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14147 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1455 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14033)) := by
  have c89235 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89236 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89237 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89238 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89239 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89240 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1455_10 := arithEq_of_rows h (row := 1455) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_10
  simp only [← c89235, ← c89236, ← c89237] at e_1455_10
  have e_1455_11 := arithEq_of_rows h (row := 1455) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_11
  simp only [← c89238, ← c89239, k2, ← c89240] at e_1455_11
  have hr := e_1455_11
  simp only [e_1455_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14148 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1455 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14034)) := by
  have c89241 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89242 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89243 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89244 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89245 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89246 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1455_12 := arithEq_of_rows h (row := 1455) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_12
  simp only [← c89241, ← c89242, ← c89243] at e_1455_12
  have e_1455_13 := arithEq_of_rows h (row := 1455) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_13
  simp only [← c89244, ← c89245, k2, ← c89246] at e_1455_13
  have hr := e_1455_13
  simp only [e_1455_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14149 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1455 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14035)) := by
  have c89247 := (publicBatchWrapper86_copies2788 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89248 := (publicBatchWrapper86_copies2789 a h).1
  have c89249 := (publicBatchWrapper86_copies2789 a h).2.1
  have c89250 := (publicBatchWrapper86_copies2789 a h).2.2.1
  have c89251 := (publicBatchWrapper86_copies2789 a h).2.2.2.1
  have c89252 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1455_14 := arithEq_of_rows h (row := 1455) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_14
  simp only [← c89247, ← c89248, ← c89249] at e_1455_14
  have e_1455_15 := arithEq_of_rows h (row := 1455) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_15
  simp only [← c89250, ← c89251, k2, ← c89252] at e_1455_15
  have hr := e_1455_15
  simp only [e_1455_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14150 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1455 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14036)) := by
  have c89253 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.1
  have c89254 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.1
  have c89255 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.1
  have c89256 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.1
  have c89257 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.1
  have c89258 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1455_16 := arithEq_of_rows h (row := 1455) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_16
  simp only [← c89253, ← c89254, ← c89255] at e_1455_16
  have e_1455_17 := arithEq_of_rows h (row := 1455) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_17
  simp only [← c89256, ← c89257, k2, ← c89258] at e_1455_17
  have hr := e_1455_17
  simp only [e_1455_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14151 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1455 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14037)) := by
  have c89259 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89260 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89261 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89262 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89263 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89264 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1455_18 := arithEq_of_rows h (row := 1455) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_18
  simp only [← c89259, ← c89260, ← c89261] at e_1455_18
  have e_1455_19 := arithEq_of_rows h (row := 1455) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1455_19
  simp only [← c89262, ← c89263, k2, ← c89264] at e_1455_19
  have hr := e_1455_19
  simp only [e_1455_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14152 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1456 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14038)) := by
  have c89265 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89266 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89267 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89268 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89269 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89270 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1456_0 := arithEq_of_rows h (row := 1456) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_0
  simp only [← c89265, ← c89266, ← c89267] at e_1456_0
  have e_1456_1 := arithEq_of_rows h (row := 1456) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_1
  simp only [← c89268, ← c89269, k2, ← c89270] at e_1456_1
  have hr := e_1456_1
  simp only [e_1456_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14153 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1456 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14039)) := by
  have c89271 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89272 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89273 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89274 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89275 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89276 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1456_2 := arithEq_of_rows h (row := 1456) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_2
  simp only [← c89271, ← c89272, ← c89273] at e_1456_2
  have e_1456_3 := arithEq_of_rows h (row := 1456) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_3
  simp only [← c89274, ← c89275, k2, ← c89276] at e_1456_3
  have hr := e_1456_3
  simp only [e_1456_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14154 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1456 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14040)) := by
  have c89277 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89278 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89279 := (publicBatchWrapper86_copies2789 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89280 := (publicBatchWrapper86_copies2790 a h).1
  have c89281 := (publicBatchWrapper86_copies2790 a h).2.1
  have c89282 := (publicBatchWrapper86_copies2790 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1456_4 := arithEq_of_rows h (row := 1456) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_4
  simp only [← c89277, ← c89278, ← c89279] at e_1456_4
  have e_1456_5 := arithEq_of_rows h (row := 1456) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_5
  simp only [← c89280, ← c89281, k2, ← c89282] at e_1456_5
  have hr := e_1456_5
  simp only [e_1456_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14155 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1456 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14041)) := by
  have c89283 := (publicBatchWrapper86_copies2790 a h).2.2.2.1
  have c89284 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.1
  have c89285 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.1
  have c89286 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.1
  have c89287 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.1
  have c89288 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1456_6 := arithEq_of_rows h (row := 1456) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_6
  simp only [← c89283, ← c89284, ← c89285] at e_1456_6
  have e_1456_7 := arithEq_of_rows h (row := 1456) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_7
  simp only [← c89286, ← c89287, k2, ← c89288] at e_1456_7
  have hr := e_1456_7
  simp only [e_1456_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14156 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1456 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14042)) := by
  have c89289 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.1
  have c89290 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89291 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89292 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89293 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89294 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1456_8 := arithEq_of_rows h (row := 1456) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_8
  simp only [← c89289, ← c89290, ← c89291] at e_1456_8
  have e_1456_9 := arithEq_of_rows h (row := 1456) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_9
  simp only [← c89292, ← c89293, k2, ← c89294] at e_1456_9
  have hr := e_1456_9
  simp only [e_1456_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14157 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1456 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14043)) := by
  have c89295 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89296 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89297 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89298 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89299 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89300 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1456_10 := arithEq_of_rows h (row := 1456) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_10
  simp only [← c89295, ← c89296, ← c89297] at e_1456_10
  have e_1456_11 := arithEq_of_rows h (row := 1456) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_11
  simp only [← c89298, ← c89299, k2, ← c89300] at e_1456_11
  have hr := e_1456_11
  simp only [e_1456_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14158 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1456 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14044)) := by
  have c89301 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89302 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89303 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89304 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89305 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89306 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1456_12 := arithEq_of_rows h (row := 1456) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_12
  simp only [← c89301, ← c89302, ← c89303] at e_1456_12
  have e_1456_13 := arithEq_of_rows h (row := 1456) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_13
  simp only [← c89304, ← c89305, k2, ← c89306] at e_1456_13
  have hr := e_1456_13
  simp only [e_1456_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14159 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1456 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14045)) := by
  have c89307 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89308 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89309 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89310 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89311 := (publicBatchWrapper86_copies2790 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89312 := (publicBatchWrapper86_copies2791 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1456_14 := arithEq_of_rows h (row := 1456) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_14
  simp only [← c89307, ← c89308, ← c89309] at e_1456_14
  have e_1456_15 := arithEq_of_rows h (row := 1456) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_15
  simp only [← c89310, ← c89311, k2, ← c89312] at e_1456_15
  have hr := e_1456_15
  simp only [e_1456_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14160 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1456 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14046)) := by
  have c89313 := (publicBatchWrapper86_copies2791 a h).2.1
  have c89314 := (publicBatchWrapper86_copies2791 a h).2.2.1
  have c89315 := (publicBatchWrapper86_copies2791 a h).2.2.2.1
  have c89316 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.1
  have c89317 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.1
  have c89318 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1456_16 := arithEq_of_rows h (row := 1456) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_16
  simp only [← c89313, ← c89314, ← c89315] at e_1456_16
  have e_1456_17 := arithEq_of_rows h (row := 1456) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_17
  simp only [← c89316, ← c89317, k2, ← c89318] at e_1456_17
  have hr := e_1456_17
  simp only [e_1456_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14161 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1456 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14047)) := by
  have c89319 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.1
  have c89320 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.1
  have c89321 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.1
  have c89322 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89323 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89324 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1456_18 := arithEq_of_rows h (row := 1456) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_18
  simp only [← c89319, ← c89320, ← c89321] at e_1456_18
  have e_1456_19 := arithEq_of_rows h (row := 1456) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1456_19
  simp only [← c89322, ← c89323, k2, ← c89324] at e_1456_19
  have hr := e_1456_19
  simp only [e_1456_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14162 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1457 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14048)) := by
  have c89325 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89326 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89327 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89328 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89329 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89330 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1457_0 := arithEq_of_rows h (row := 1457) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_0
  simp only [← c89325, ← c89326, ← c89327] at e_1457_0
  have e_1457_1 := arithEq_of_rows h (row := 1457) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_1
  simp only [← c89328, ← c89329, k2, ← c89330] at e_1457_1
  have hr := e_1457_1
  simp only [e_1457_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14163 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1457 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14049)) := by
  have c89331 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89332 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89333 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89334 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89335 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89336 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1457_2 := arithEq_of_rows h (row := 1457) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_2
  simp only [← c89331, ← c89332, ← c89333] at e_1457_2
  have e_1457_3 := arithEq_of_rows h (row := 1457) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_3
  simp only [← c89334, ← c89335, k2, ← c89336] at e_1457_3
  have hr := e_1457_3
  simp only [e_1457_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14164 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1457 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14050)) := by
  have c89337 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89338 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89339 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89340 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89341 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89342 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1457_4 := arithEq_of_rows h (row := 1457) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_4
  simp only [← c89337, ← c89338, ← c89339] at e_1457_4
  have e_1457_5 := arithEq_of_rows h (row := 1457) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_5
  simp only [← c89340, ← c89341, k2, ← c89342] at e_1457_5
  have hr := e_1457_5
  simp only [e_1457_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14165 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1457 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14051)) := by
  have c89343 := (publicBatchWrapper86_copies2791 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89344 := (publicBatchWrapper86_copies2792 a h).1
  have c89345 := (publicBatchWrapper86_copies2792 a h).2.1
  have c89346 := (publicBatchWrapper86_copies2792 a h).2.2.1
  have c89347 := (publicBatchWrapper86_copies2792 a h).2.2.2.1
  have c89348 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1457_6 := arithEq_of_rows h (row := 1457) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_6
  simp only [← c89343, ← c89344, ← c89345] at e_1457_6
  have e_1457_7 := arithEq_of_rows h (row := 1457) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_7
  simp only [← c89346, ← c89347, k2, ← c89348] at e_1457_7
  have hr := e_1457_7
  simp only [e_1457_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14166 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1457 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14052)) := by
  have c89349 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.1
  have c89350 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.1
  have c89351 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.1
  have c89352 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.1
  have c89353 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.1
  have c89354 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1457_8 := arithEq_of_rows h (row := 1457) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_8
  simp only [← c89349, ← c89350, ← c89351] at e_1457_8
  have e_1457_9 := arithEq_of_rows h (row := 1457) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_9
  simp only [← c89352, ← c89353, k2, ← c89354] at e_1457_9
  have hr := e_1457_9
  simp only [e_1457_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14167 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1457 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14053)) := by
  have c89355 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89356 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89357 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89358 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89359 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89360 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1457_10 := arithEq_of_rows h (row := 1457) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_10
  simp only [← c89355, ← c89356, ← c89357] at e_1457_10
  have e_1457_11 := arithEq_of_rows h (row := 1457) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_11
  simp only [← c89358, ← c89359, k2, ← c89360] at e_1457_11
  have hr := e_1457_11
  simp only [e_1457_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14168 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1457 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14054)) := by
  have c89361 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89362 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89363 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89364 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89365 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89366 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1457_12 := arithEq_of_rows h (row := 1457) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_12
  simp only [← c89361, ← c89362, ← c89363] at e_1457_12
  have e_1457_13 := arithEq_of_rows h (row := 1457) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_13
  simp only [← c89364, ← c89365, k2, ← c89366] at e_1457_13
  have hr := e_1457_13
  simp only [e_1457_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14169 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1457 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14055)) := by
  have c89367 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89368 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89369 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89370 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89371 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89372 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1457_14 := arithEq_of_rows h (row := 1457) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_14
  simp only [← c89367, ← c89368, ← c89369] at e_1457_14
  have e_1457_15 := arithEq_of_rows h (row := 1457) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_15
  simp only [← c89370, ← c89371, k2, ← c89372] at e_1457_15
  have hr := e_1457_15
  simp only [e_1457_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14170 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1457 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14056)) := by
  have c89373 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89374 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89375 := (publicBatchWrapper86_copies2792 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89376 := (publicBatchWrapper86_copies2793 a h).1
  have c89377 := (publicBatchWrapper86_copies2793 a h).2.1
  have c89378 := (publicBatchWrapper86_copies2793 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1457_16 := arithEq_of_rows h (row := 1457) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_16
  simp only [← c89373, ← c89374, ← c89375] at e_1457_16
  have e_1457_17 := arithEq_of_rows h (row := 1457) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_17
  simp only [← c89376, ← c89377, k2, ← c89378] at e_1457_17
  have hr := e_1457_17
  simp only [e_1457_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14171 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1457 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14057)) := by
  have c89379 := (publicBatchWrapper86_copies2793 a h).2.2.2.1
  have c89380 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.1
  have c89381 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.1
  have c89382 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.1
  have c89383 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.1
  have c89384 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1457_18 := arithEq_of_rows h (row := 1457) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_18
  simp only [← c89379, ← c89380, ← c89381] at e_1457_18
  have e_1457_19 := arithEq_of_rows h (row := 1457) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1457_19
  simp only [← c89382, ← c89383, k2, ← c89384] at e_1457_19
  have hr := e_1457_19
  simp only [e_1457_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14172 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1458 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14058)) := by
  have c89385 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.1
  have c89386 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89387 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89388 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89389 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89390 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1458_0 := arithEq_of_rows h (row := 1458) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_0
  simp only [← c89385, ← c89386, ← c89387] at e_1458_0
  have e_1458_1 := arithEq_of_rows h (row := 1458) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_1
  simp only [← c89388, ← c89389, k2, ← c89390] at e_1458_1
  have hr := e_1458_1
  simp only [e_1458_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14173 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1458 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14059)) := by
  have c89391 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89392 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89393 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89394 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89395 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89396 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1458_2 := arithEq_of_rows h (row := 1458) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_2
  simp only [← c89391, ← c89392, ← c89393] at e_1458_2
  have e_1458_3 := arithEq_of_rows h (row := 1458) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_3
  simp only [← c89394, ← c89395, k2, ← c89396] at e_1458_3
  have hr := e_1458_3
  simp only [e_1458_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14174 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1458 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14060)) := by
  have c89397 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89398 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89399 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89400 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89401 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89402 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1458_4 := arithEq_of_rows h (row := 1458) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_4
  simp only [← c89397, ← c89398, ← c89399] at e_1458_4
  have e_1458_5 := arithEq_of_rows h (row := 1458) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_5
  simp only [← c89400, ← c89401, k2, ← c89402] at e_1458_5
  have hr := e_1458_5
  simp only [e_1458_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14175 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1458 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14061)) := by
  have c89403 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89404 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89405 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89406 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89407 := (publicBatchWrapper86_copies2793 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89408 := (publicBatchWrapper86_copies2794 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1458_6 := arithEq_of_rows h (row := 1458) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_6
  simp only [← c89403, ← c89404, ← c89405] at e_1458_6
  have e_1458_7 := arithEq_of_rows h (row := 1458) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_7
  simp only [← c89406, ← c89407, k2, ← c89408] at e_1458_7
  have hr := e_1458_7
  simp only [e_1458_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14176 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1458 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14062)) := by
  have c89409 := (publicBatchWrapper86_copies2794 a h).2.1
  have c89410 := (publicBatchWrapper86_copies2794 a h).2.2.1
  have c89411 := (publicBatchWrapper86_copies2794 a h).2.2.2.1
  have c89412 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.1
  have c89413 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.1
  have c89414 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1458_8 := arithEq_of_rows h (row := 1458) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_8
  simp only [← c89409, ← c89410, ← c89411] at e_1458_8
  have e_1458_9 := arithEq_of_rows h (row := 1458) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_9
  simp only [← c89412, ← c89413, k2, ← c89414] at e_1458_9
  have hr := e_1458_9
  simp only [e_1458_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14177 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1458 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14063)) := by
  have c89415 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.1
  have c89416 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.1
  have c89417 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.1
  have c89418 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89419 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89420 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1458_10 := arithEq_of_rows h (row := 1458) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_10
  simp only [← c89415, ← c89416, ← c89417] at e_1458_10
  have e_1458_11 := arithEq_of_rows h (row := 1458) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_11
  simp only [← c89418, ← c89419, k2, ← c89420] at e_1458_11
  have hr := e_1458_11
  simp only [e_1458_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14178 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1458 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14064)) := by
  have c89421 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89422 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89423 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89424 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89425 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89426 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1458_12 := arithEq_of_rows h (row := 1458) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_12
  simp only [← c89421, ← c89422, ← c89423] at e_1458_12
  have e_1458_13 := arithEq_of_rows h (row := 1458) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_13
  simp only [← c89424, ← c89425, k2, ← c89426] at e_1458_13
  have hr := e_1458_13
  simp only [e_1458_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14179 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1458 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14065)) := by
  have c89427 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89428 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89429 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89430 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89431 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89432 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1458_14 := arithEq_of_rows h (row := 1458) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_14
  simp only [← c89427, ← c89428, ← c89429] at e_1458_14
  have e_1458_15 := arithEq_of_rows h (row := 1458) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_15
  simp only [← c89430, ← c89431, k2, ← c89432] at e_1458_15
  have hr := e_1458_15
  simp only [e_1458_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14180 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1458 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14066)) := by
  have c89433 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89434 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89435 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89436 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89437 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89438 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1458_16 := arithEq_of_rows h (row := 1458) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_16
  simp only [← c89433, ← c89434, ← c89435] at e_1458_16
  have e_1458_17 := arithEq_of_rows h (row := 1458) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_17
  simp only [← c89436, ← c89437, k2, ← c89438] at e_1458_17
  have hr := e_1458_17
  simp only [e_1458_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14181 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1458 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14067)) := by
  have c89439 := (publicBatchWrapper86_copies2794 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89440 := (publicBatchWrapper86_copies2795 a h).1
  have c89441 := (publicBatchWrapper86_copies2795 a h).2.1
  have c89442 := (publicBatchWrapper86_copies2795 a h).2.2.1
  have c89443 := (publicBatchWrapper86_copies2795 a h).2.2.2.1
  have c89444 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1458_18 := arithEq_of_rows h (row := 1458) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_18
  simp only [← c89439, ← c89440, ← c89441] at e_1458_18
  have e_1458_19 := arithEq_of_rows h (row := 1458) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1458_19
  simp only [← c89442, ← c89443, k2, ← c89444] at e_1458_19
  have hr := e_1458_19
  simp only [e_1458_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14182 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1459 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14068)) := by
  have c89445 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.1
  have c89446 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.1
  have c89447 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.1
  have c89448 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.1
  have c89449 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.1
  have c89450 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1459_0 := arithEq_of_rows h (row := 1459) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_0
  simp only [← c89445, ← c89446, ← c89447] at e_1459_0
  have e_1459_1 := arithEq_of_rows h (row := 1459) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_1
  simp only [← c89448, ← c89449, k2, ← c89450] at e_1459_1
  have hr := e_1459_1
  simp only [e_1459_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14183 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1459 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14069)) := by
  have c89451 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89452 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89453 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89454 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89455 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89456 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1459_2 := arithEq_of_rows h (row := 1459) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_2
  simp only [← c89451, ← c89452, ← c89453] at e_1459_2
  have e_1459_3 := arithEq_of_rows h (row := 1459) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_3
  simp only [← c89454, ← c89455, k2, ← c89456] at e_1459_3
  have hr := e_1459_3
  simp only [e_1459_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14184 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1459 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14070)) := by
  have c89457 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89458 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89459 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89460 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89461 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89462 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1459_4 := arithEq_of_rows h (row := 1459) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_4
  simp only [← c89457, ← c89458, ← c89459] at e_1459_4
  have e_1459_5 := arithEq_of_rows h (row := 1459) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_5
  simp only [← c89460, ← c89461, k2, ← c89462] at e_1459_5
  have hr := e_1459_5
  simp only [e_1459_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14185 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1459 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14071)) := by
  have c89463 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89464 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89465 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89466 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89467 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89468 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1459_6 := arithEq_of_rows h (row := 1459) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_6
  simp only [← c89463, ← c89464, ← c89465] at e_1459_6
  have e_1459_7 := arithEq_of_rows h (row := 1459) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_7
  simp only [← c89466, ← c89467, k2, ← c89468] at e_1459_7
  have hr := e_1459_7
  simp only [e_1459_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14186 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1459 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14072)) := by
  have c89469 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89470 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89471 := (publicBatchWrapper86_copies2795 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89472 := (publicBatchWrapper86_copies2796 a h).1
  have c89473 := (publicBatchWrapper86_copies2796 a h).2.1
  have c89474 := (publicBatchWrapper86_copies2796 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1459_8 := arithEq_of_rows h (row := 1459) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_8
  simp only [← c89469, ← c89470, ← c89471] at e_1459_8
  have e_1459_9 := arithEq_of_rows h (row := 1459) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_9
  simp only [← c89472, ← c89473, k2, ← c89474] at e_1459_9
  have hr := e_1459_9
  simp only [e_1459_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14187 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1459 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14073)) := by
  have c89475 := (publicBatchWrapper86_copies2796 a h).2.2.2.1
  have c89476 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.1
  have c89477 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.1
  have c89478 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.1
  have c89479 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.1
  have c89480 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1459_10 := arithEq_of_rows h (row := 1459) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_10
  simp only [← c89475, ← c89476, ← c89477] at e_1459_10
  have e_1459_11 := arithEq_of_rows h (row := 1459) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_11
  simp only [← c89478, ← c89479, k2, ← c89480] at e_1459_11
  have hr := e_1459_11
  simp only [e_1459_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14188 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1459 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14074)) := by
  have c89481 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.1
  have c89482 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89483 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89484 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89485 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89486 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1459_12 := arithEq_of_rows h (row := 1459) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_12
  simp only [← c89481, ← c89482, ← c89483] at e_1459_12
  have e_1459_13 := arithEq_of_rows h (row := 1459) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_13
  simp only [← c89484, ← c89485, k2, ← c89486] at e_1459_13
  have hr := e_1459_13
  simp only [e_1459_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14189 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1459 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14075)) := by
  have c89487 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89488 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89489 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89490 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89491 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89492 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1459_14 := arithEq_of_rows h (row := 1459) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_14
  simp only [← c89487, ← c89488, ← c89489] at e_1459_14
  have e_1459_15 := arithEq_of_rows h (row := 1459) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_15
  simp only [← c89490, ← c89491, k2, ← c89492] at e_1459_15
  have hr := e_1459_15
  simp only [e_1459_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14190 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1459 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14076)) := by
  have c89493 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89494 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89495 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89496 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89497 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89498 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1459_16 := arithEq_of_rows h (row := 1459) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_16
  simp only [← c89493, ← c89494, ← c89495] at e_1459_16
  have e_1459_17 := arithEq_of_rows h (row := 1459) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_17
  simp only [← c89496, ← c89497, k2, ← c89498] at e_1459_17
  have hr := e_1459_17
  simp only [e_1459_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14191 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1459 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14077)) := by
  have c89499 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89500 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89501 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89502 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89503 := (publicBatchWrapper86_copies2796 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89504 := (publicBatchWrapper86_copies2797 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1459_18 := arithEq_of_rows h (row := 1459) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_18
  simp only [← c89499, ← c89500, ← c89501] at e_1459_18
  have e_1459_19 := arithEq_of_rows h (row := 1459) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1459_19
  simp only [← c89502, ← c89503, k2, ← c89504] at e_1459_19
  have hr := e_1459_19
  simp only [e_1459_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14192 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1460 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14078)) := by
  have c89505 := (publicBatchWrapper86_copies2797 a h).2.1
  have c89506 := (publicBatchWrapper86_copies2797 a h).2.2.1
  have c89507 := (publicBatchWrapper86_copies2797 a h).2.2.2.1
  have c89508 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.1
  have c89509 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.1
  have c89510 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1460_0 := arithEq_of_rows h (row := 1460) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_0
  simp only [← c89505, ← c89506, ← c89507] at e_1460_0
  have e_1460_1 := arithEq_of_rows h (row := 1460) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_1
  simp only [← c89508, ← c89509, k2, ← c89510] at e_1460_1
  have hr := e_1460_1
  simp only [e_1460_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14193 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1460 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14079)) := by
  have c89511 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.1
  have c89512 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.1
  have c89513 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.1
  have c89514 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89515 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89516 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1460_2 := arithEq_of_rows h (row := 1460) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_2
  simp only [← c89511, ← c89512, ← c89513] at e_1460_2
  have e_1460_3 := arithEq_of_rows h (row := 1460) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_3
  simp only [← c89514, ← c89515, k2, ← c89516] at e_1460_3
  have hr := e_1460_3
  simp only [e_1460_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14194 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1460 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14080)) := by
  have c89517 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89518 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89519 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89520 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89521 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89522 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1460_4 := arithEq_of_rows h (row := 1460) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_4
  simp only [← c89517, ← c89518, ← c89519] at e_1460_4
  have e_1460_5 := arithEq_of_rows h (row := 1460) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_5
  simp only [← c89520, ← c89521, k2, ← c89522] at e_1460_5
  have hr := e_1460_5
  simp only [e_1460_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14195 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1460 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14081)) := by
  have c89523 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89524 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89525 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89526 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89527 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89528 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1460_6 := arithEq_of_rows h (row := 1460) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_6
  simp only [← c89523, ← c89524, ← c89525] at e_1460_6
  have e_1460_7 := arithEq_of_rows h (row := 1460) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_7
  simp only [← c89526, ← c89527, k2, ← c89528] at e_1460_7
  have hr := e_1460_7
  simp only [e_1460_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14196 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1460 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14082)) := by
  have c89529 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89530 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89531 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89532 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89533 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89534 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1460_8 := arithEq_of_rows h (row := 1460) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_8
  simp only [← c89529, ← c89530, ← c89531] at e_1460_8
  have e_1460_9 := arithEq_of_rows h (row := 1460) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_9
  simp only [← c89532, ← c89533, k2, ← c89534] at e_1460_9
  have hr := e_1460_9
  simp only [e_1460_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14197 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1460 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14083)) := by
  have c89535 := (publicBatchWrapper86_copies2797 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89536 := (publicBatchWrapper86_copies2798 a h).1
  have c89537 := (publicBatchWrapper86_copies2798 a h).2.1
  have c89538 := (publicBatchWrapper86_copies2798 a h).2.2.1
  have c89539 := (publicBatchWrapper86_copies2798 a h).2.2.2.1
  have c89540 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1460_10 := arithEq_of_rows h (row := 1460) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_10
  simp only [← c89535, ← c89536, ← c89537] at e_1460_10
  have e_1460_11 := arithEq_of_rows h (row := 1460) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_11
  simp only [← c89538, ← c89539, k2, ← c89540] at e_1460_11
  have hr := e_1460_11
  simp only [e_1460_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14198 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1460 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14084)) := by
  have c89541 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.1
  have c89542 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.1
  have c89543 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.1
  have c89544 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.1
  have c89545 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.1
  have c89546 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1460_12 := arithEq_of_rows h (row := 1460) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_12
  simp only [← c89541, ← c89542, ← c89543] at e_1460_12
  have e_1460_13 := arithEq_of_rows h (row := 1460) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_13
  simp only [← c89544, ← c89545, k2, ← c89546] at e_1460_13
  have hr := e_1460_13
  simp only [e_1460_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14199 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1460 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14085)) := by
  have c89547 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89548 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89549 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89550 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89551 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89552 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1460_14 := arithEq_of_rows h (row := 1460) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_14
  simp only [← c89547, ← c89548, ← c89549] at e_1460_14
  have e_1460_15 := arithEq_of_rows h (row := 1460) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_15
  simp only [← c89550, ← c89551, k2, ← c89552] at e_1460_15
  have hr := e_1460_15
  simp only [e_1460_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14200 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1460 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14086)) := by
  have c89553 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89554 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89555 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89556 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89557 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89558 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1460_16 := arithEq_of_rows h (row := 1460) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_16
  simp only [← c89553, ← c89554, ← c89555] at e_1460_16
  have e_1460_17 := arithEq_of_rows h (row := 1460) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_17
  simp only [← c89556, ← c89557, k2, ← c89558] at e_1460_17
  have hr := e_1460_17
  simp only [e_1460_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14201 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1460 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14087)) := by
  have c89559 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89560 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89561 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89562 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89563 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89564 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1460_18 := arithEq_of_rows h (row := 1460) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_18
  simp only [← c89559, ← c89560, ← c89561] at e_1460_18
  have e_1460_19 := arithEq_of_rows h (row := 1460) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1460_19
  simp only [← c89562, ← c89563, k2, ← c89564] at e_1460_19
  have hr := e_1460_19
  simp only [e_1460_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14202 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1461 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14088)) := by
  have c89565 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89566 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89567 := (publicBatchWrapper86_copies2798 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89568 := (publicBatchWrapper86_copies2799 a h).1
  have c89569 := (publicBatchWrapper86_copies2799 a h).2.1
  have c89570 := (publicBatchWrapper86_copies2799 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1461_0 := arithEq_of_rows h (row := 1461) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_0
  simp only [← c89565, ← c89566, ← c89567] at e_1461_0
  have e_1461_1 := arithEq_of_rows h (row := 1461) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_1
  simp only [← c89568, ← c89569, k2, ← c89570] at e_1461_1
  have hr := e_1461_1
  simp only [e_1461_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14203 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1461 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14089)) := by
  have c89571 := (publicBatchWrapper86_copies2799 a h).2.2.2.1
  have c89572 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.1
  have c89573 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.1
  have c89574 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.1
  have c89575 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.1
  have c89576 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1461_2 := arithEq_of_rows h (row := 1461) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_2
  simp only [← c89571, ← c89572, ← c89573] at e_1461_2
  have e_1461_3 := arithEq_of_rows h (row := 1461) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_3
  simp only [← c89574, ← c89575, k2, ← c89576] at e_1461_3
  have hr := e_1461_3
  simp only [e_1461_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14204 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1461 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14090)) := by
  have c89577 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.1
  have c89578 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89579 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89580 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89581 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89582 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1461_4 := arithEq_of_rows h (row := 1461) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_4
  simp only [← c89577, ← c89578, ← c89579] at e_1461_4
  have e_1461_5 := arithEq_of_rows h (row := 1461) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_5
  simp only [← c89580, ← c89581, k2, ← c89582] at e_1461_5
  have hr := e_1461_5
  simp only [e_1461_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14205 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1461 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14091)) := by
  have c89583 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89584 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89585 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89586 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89587 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89588 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1461_6 := arithEq_of_rows h (row := 1461) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_6
  simp only [← c89583, ← c89584, ← c89585] at e_1461_6
  have e_1461_7 := arithEq_of_rows h (row := 1461) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_7
  simp only [← c89586, ← c89587, k2, ← c89588] at e_1461_7
  have hr := e_1461_7
  simp only [e_1461_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14206 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1461 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14092)) := by
  have c89589 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89590 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89591 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89592 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89593 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89594 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1461_8 := arithEq_of_rows h (row := 1461) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_8
  simp only [← c89589, ← c89590, ← c89591] at e_1461_8
  have e_1461_9 := arithEq_of_rows h (row := 1461) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_9
  simp only [← c89592, ← c89593, k2, ← c89594] at e_1461_9
  have hr := e_1461_9
  simp only [e_1461_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14207 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1461 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14093)) := by
  have c89595 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89596 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89597 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89598 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89599 := (publicBatchWrapper86_copies2799 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89600 := (publicBatchWrapper86_copies2800 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1461_10 := arithEq_of_rows h (row := 1461) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_10
  simp only [← c89595, ← c89596, ← c89597] at e_1461_10
  have e_1461_11 := arithEq_of_rows h (row := 1461) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_11
  simp only [← c89598, ← c89599, k2, ← c89600] at e_1461_11
  have hr := e_1461_11
  simp only [e_1461_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14208 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1461 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14094)) := by
  have c89601 := (publicBatchWrapper86_copies2800 a h).2.1
  have c89602 := (publicBatchWrapper86_copies2800 a h).2.2.1
  have c89603 := (publicBatchWrapper86_copies2800 a h).2.2.2.1
  have c89604 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.1
  have c89605 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.1
  have c89606 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1461_12 := arithEq_of_rows h (row := 1461) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_12
  simp only [← c89601, ← c89602, ← c89603] at e_1461_12
  have e_1461_13 := arithEq_of_rows h (row := 1461) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_13
  simp only [← c89604, ← c89605, k2, ← c89606] at e_1461_13
  have hr := e_1461_13
  simp only [e_1461_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14209 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1461 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14095)) := by
  have c89607 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.1
  have c89608 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.1
  have c89609 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.1
  have c89610 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89611 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89612 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1461_14 := arithEq_of_rows h (row := 1461) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_14
  simp only [← c89607, ← c89608, ← c89609] at e_1461_14
  have e_1461_15 := arithEq_of_rows h (row := 1461) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_15
  simp only [← c89610, ← c89611, k2, ← c89612] at e_1461_15
  have hr := e_1461_15
  simp only [e_1461_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14210 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1461 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14096)) := by
  have c89613 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89614 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89615 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89616 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89617 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89618 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1461_16 := arithEq_of_rows h (row := 1461) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_16
  simp only [← c89613, ← c89614, ← c89615] at e_1461_16
  have e_1461_17 := arithEq_of_rows h (row := 1461) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_17
  simp only [← c89616, ← c89617, k2, ← c89618] at e_1461_17
  have hr := e_1461_17
  simp only [e_1461_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14211 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1461 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14097)) := by
  have c89619 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89620 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89621 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89622 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89623 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89624 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1461_18 := arithEq_of_rows h (row := 1461) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_18
  simp only [← c89619, ← c89620, ← c89621] at e_1461_18
  have e_1461_19 := arithEq_of_rows h (row := 1461) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1461_19
  simp only [← c89622, ← c89623, k2, ← c89624] at e_1461_19
  have hr := e_1461_19
  simp only [e_1461_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14212 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1462 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14098)) := by
  have c89625 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89626 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89627 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89628 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89629 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89630 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1462_0 := arithEq_of_rows h (row := 1462) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_0
  simp only [← c89625, ← c89626, ← c89627] at e_1462_0
  have e_1462_1 := arithEq_of_rows h (row := 1462) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_1
  simp only [← c89628, ← c89629, k2, ← c89630] at e_1462_1
  have hr := e_1462_1
  simp only [e_1462_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14213 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1462 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14099)) := by
  have c89631 := (publicBatchWrapper86_copies2800 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89632 := (publicBatchWrapper86_copies2801 a h).1
  have c89633 := (publicBatchWrapper86_copies2801 a h).2.1
  have c89634 := (publicBatchWrapper86_copies2801 a h).2.2.1
  have c89635 := (publicBatchWrapper86_copies2801 a h).2.2.2.1
  have c89636 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1462_2 := arithEq_of_rows h (row := 1462) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_2
  simp only [← c89631, ← c89632, ← c89633] at e_1462_2
  have e_1462_3 := arithEq_of_rows h (row := 1462) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_3
  simp only [← c89634, ← c89635, k2, ← c89636] at e_1462_3
  have hr := e_1462_3
  simp only [e_1462_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14214 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1462 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14100)) := by
  have c89637 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.1
  have c89638 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.1
  have c89639 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.1
  have c89640 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.1
  have c89641 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.1
  have c89642 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1462_4 := arithEq_of_rows h (row := 1462) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_4
  simp only [← c89637, ← c89638, ← c89639] at e_1462_4
  have e_1462_5 := arithEq_of_rows h (row := 1462) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_5
  simp only [← c89640, ← c89641, k2, ← c89642] at e_1462_5
  have hr := e_1462_5
  simp only [e_1462_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14215 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1462 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14101)) := by
  have c89643 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89644 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89645 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89646 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89647 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89648 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1462_6 := arithEq_of_rows h (row := 1462) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_6
  simp only [← c89643, ← c89644, ← c89645] at e_1462_6
  have e_1462_7 := arithEq_of_rows h (row := 1462) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_7
  simp only [← c89646, ← c89647, k2, ← c89648] at e_1462_7
  have hr := e_1462_7
  simp only [e_1462_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14216 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1462 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14102)) := by
  have c89649 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89650 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89651 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89652 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89653 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89654 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1462_8 := arithEq_of_rows h (row := 1462) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_8
  simp only [← c89649, ← c89650, ← c89651] at e_1462_8
  have e_1462_9 := arithEq_of_rows h (row := 1462) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_9
  simp only [← c89652, ← c89653, k2, ← c89654] at e_1462_9
  have hr := e_1462_9
  simp only [e_1462_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14217 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1462 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14103)) := by
  have c89655 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89656 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89657 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89658 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89659 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89660 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1462_10 := arithEq_of_rows h (row := 1462) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_10
  simp only [← c89655, ← c89656, ← c89657] at e_1462_10
  have e_1462_11 := arithEq_of_rows h (row := 1462) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_11
  simp only [← c89658, ← c89659, k2, ← c89660] at e_1462_11
  have hr := e_1462_11
  simp only [e_1462_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14218 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1462 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14104)) := by
  have c89661 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89662 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89663 := (publicBatchWrapper86_copies2801 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89664 := (publicBatchWrapper86_copies2802 a h).1
  have c89665 := (publicBatchWrapper86_copies2802 a h).2.1
  have c89666 := (publicBatchWrapper86_copies2802 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1462_12 := arithEq_of_rows h (row := 1462) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_12
  simp only [← c89661, ← c89662, ← c89663] at e_1462_12
  have e_1462_13 := arithEq_of_rows h (row := 1462) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_13
  simp only [← c89664, ← c89665, k2, ← c89666] at e_1462_13
  have hr := e_1462_13
  simp only [e_1462_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14219 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1462 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14105)) := by
  have c89667 := (publicBatchWrapper86_copies2802 a h).2.2.2.1
  have c89668 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.1
  have c89669 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.1
  have c89670 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.1
  have c89671 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.1
  have c89672 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1462_14 := arithEq_of_rows h (row := 1462) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_14
  simp only [← c89667, ← c89668, ← c89669] at e_1462_14
  have e_1462_15 := arithEq_of_rows h (row := 1462) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_15
  simp only [← c89670, ← c89671, k2, ← c89672] at e_1462_15
  have hr := e_1462_15
  simp only [e_1462_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14220 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1462 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14106)) := by
  have c89673 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.1
  have c89674 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89675 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89676 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89677 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89678 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1462_16 := arithEq_of_rows h (row := 1462) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_16
  simp only [← c89673, ← c89674, ← c89675] at e_1462_16
  have e_1462_17 := arithEq_of_rows h (row := 1462) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_17
  simp only [← c89676, ← c89677, k2, ← c89678] at e_1462_17
  have hr := e_1462_17
  simp only [e_1462_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14221 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1462 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14107)) := by
  have c89679 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89680 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89681 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89682 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89683 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89684 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1462_18 := arithEq_of_rows h (row := 1462) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_18
  simp only [← c89679, ← c89680, ← c89681] at e_1462_18
  have e_1462_19 := arithEq_of_rows h (row := 1462) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1462_19
  simp only [← c89682, ← c89683, k2, ← c89684] at e_1462_19
  have hr := e_1462_19
  simp only [e_1462_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14222 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1463 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14108)) := by
  have c89685 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89686 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89687 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89688 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89689 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89690 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1463_0 := arithEq_of_rows h (row := 1463) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_0
  simp only [← c89685, ← c89686, ← c89687] at e_1463_0
  have e_1463_1 := arithEq_of_rows h (row := 1463) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_1
  simp only [← c89688, ← c89689, k2, ← c89690] at e_1463_1
  have hr := e_1463_1
  simp only [e_1463_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14223 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1463 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14109)) := by
  have c89691 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89692 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89693 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89694 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89695 := (publicBatchWrapper86_copies2802 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89696 := (publicBatchWrapper86_copies2803 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1463_2 := arithEq_of_rows h (row := 1463) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_2
  simp only [← c89691, ← c89692, ← c89693] at e_1463_2
  have e_1463_3 := arithEq_of_rows h (row := 1463) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_3
  simp only [← c89694, ← c89695, k2, ← c89696] at e_1463_3
  have hr := e_1463_3
  simp only [e_1463_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14224 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1463 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14110)) := by
  have c89697 := (publicBatchWrapper86_copies2803 a h).2.1
  have c89698 := (publicBatchWrapper86_copies2803 a h).2.2.1
  have c89699 := (publicBatchWrapper86_copies2803 a h).2.2.2.1
  have c89700 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.1
  have c89701 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.1
  have c89702 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1463_4 := arithEq_of_rows h (row := 1463) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_4
  simp only [← c89697, ← c89698, ← c89699] at e_1463_4
  have e_1463_5 := arithEq_of_rows h (row := 1463) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_5
  simp only [← c89700, ← c89701, k2, ← c89702] at e_1463_5
  have hr := e_1463_5
  simp only [e_1463_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14225 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1463 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14111)) := by
  have c89703 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.1
  have c89704 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.1
  have c89705 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.1
  have c89706 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89707 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89708 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1463_6 := arithEq_of_rows h (row := 1463) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_6
  simp only [← c89703, ← c89704, ← c89705] at e_1463_6
  have e_1463_7 := arithEq_of_rows h (row := 1463) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_7
  simp only [← c89706, ← c89707, k2, ← c89708] at e_1463_7
  have hr := e_1463_7
  simp only [e_1463_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14226 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1463 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14112)) := by
  have c89709 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89710 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89711 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89712 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89713 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89714 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1463_8 := arithEq_of_rows h (row := 1463) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_8
  simp only [← c89709, ← c89710, ← c89711] at e_1463_8
  have e_1463_9 := arithEq_of_rows h (row := 1463) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_9
  simp only [← c89712, ← c89713, k2, ← c89714] at e_1463_9
  have hr := e_1463_9
  simp only [e_1463_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14227 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1463 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14113)) := by
  have c89715 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89716 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89717 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89718 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89719 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89720 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1463_10 := arithEq_of_rows h (row := 1463) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_10
  simp only [← c89715, ← c89716, ← c89717] at e_1463_10
  have e_1463_11 := arithEq_of_rows h (row := 1463) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_11
  simp only [← c89718, ← c89719, k2, ← c89720] at e_1463_11
  have hr := e_1463_11
  simp only [e_1463_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14228 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1463 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14114)) := by
  have c89721 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89722 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89723 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89724 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89725 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89726 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1463_12 := arithEq_of_rows h (row := 1463) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_12
  simp only [← c89721, ← c89722, ← c89723] at e_1463_12
  have e_1463_13 := arithEq_of_rows h (row := 1463) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_13
  simp only [← c89724, ← c89725, k2, ← c89726] at e_1463_13
  have hr := e_1463_13
  simp only [e_1463_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14229 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1463 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14115)) := by
  have c89727 := (publicBatchWrapper86_copies2803 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89728 := (publicBatchWrapper86_copies2804 a h).1
  have c89729 := (publicBatchWrapper86_copies2804 a h).2.1
  have c89730 := (publicBatchWrapper86_copies2804 a h).2.2.1
  have c89731 := (publicBatchWrapper86_copies2804 a h).2.2.2.1
  have c89732 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1463_14 := arithEq_of_rows h (row := 1463) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_14
  simp only [← c89727, ← c89728, ← c89729] at e_1463_14
  have e_1463_15 := arithEq_of_rows h (row := 1463) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_15
  simp only [← c89730, ← c89731, k2, ← c89732] at e_1463_15
  have hr := e_1463_15
  simp only [e_1463_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14230 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1463 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14116)) := by
  have c89733 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.1
  have c89734 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.1
  have c89735 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.1
  have c89736 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.1
  have c89737 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.1
  have c89738 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1463_16 := arithEq_of_rows h (row := 1463) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_16
  simp only [← c89733, ← c89734, ← c89735] at e_1463_16
  have e_1463_17 := arithEq_of_rows h (row := 1463) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_17
  simp only [← c89736, ← c89737, k2, ← c89738] at e_1463_17
  have hr := e_1463_17
  simp only [e_1463_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14231 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1463 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14117)) := by
  have c89739 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89740 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89741 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89742 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89743 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89744 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1463_18 := arithEq_of_rows h (row := 1463) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_18
  simp only [← c89739, ← c89740, ← c89741] at e_1463_18
  have e_1463_19 := arithEq_of_rows h (row := 1463) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1463_19
  simp only [← c89742, ← c89743, k2, ← c89744] at e_1463_19
  have hr := e_1463_19
  simp only [e_1463_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14232 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1464 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14118)) := by
  have c89745 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89746 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89747 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89748 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89749 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89750 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1464_0 := arithEq_of_rows h (row := 1464) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_0
  simp only [← c89745, ← c89746, ← c89747] at e_1464_0
  have e_1464_1 := arithEq_of_rows h (row := 1464) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_1
  simp only [← c89748, ← c89749, k2, ← c89750] at e_1464_1
  have hr := e_1464_1
  simp only [e_1464_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14233 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1464 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14119)) := by
  have c89751 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89752 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89753 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89754 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89755 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89756 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1464_2 := arithEq_of_rows h (row := 1464) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_2
  simp only [← c89751, ← c89752, ← c89753] at e_1464_2
  have e_1464_3 := arithEq_of_rows h (row := 1464) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_3
  simp only [← c89754, ← c89755, k2, ← c89756] at e_1464_3
  have hr := e_1464_3
  simp only [e_1464_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14234 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1464 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14120)) := by
  have c89757 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89758 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89759 := (publicBatchWrapper86_copies2804 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89760 := (publicBatchWrapper86_copies2805 a h).1
  have c89761 := (publicBatchWrapper86_copies2805 a h).2.1
  have c89762 := (publicBatchWrapper86_copies2805 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1464_4 := arithEq_of_rows h (row := 1464) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_4
  simp only [← c89757, ← c89758, ← c89759] at e_1464_4
  have e_1464_5 := arithEq_of_rows h (row := 1464) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_5
  simp only [← c89760, ← c89761, k2, ← c89762] at e_1464_5
  have hr := e_1464_5
  simp only [e_1464_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14235 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1464 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14121)) := by
  have c89763 := (publicBatchWrapper86_copies2805 a h).2.2.2.1
  have c89764 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.1
  have c89765 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.1
  have c89766 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.1
  have c89767 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.1
  have c89768 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1464_6 := arithEq_of_rows h (row := 1464) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_6
  simp only [← c89763, ← c89764, ← c89765] at e_1464_6
  have e_1464_7 := arithEq_of_rows h (row := 1464) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_7
  simp only [← c89766, ← c89767, k2, ← c89768] at e_1464_7
  have hr := e_1464_7
  simp only [e_1464_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14236 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1464 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14122)) := by
  have c89769 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.1
  have c89770 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89771 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89772 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89773 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89774 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1464_8 := arithEq_of_rows h (row := 1464) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_8
  simp only [← c89769, ← c89770, ← c89771] at e_1464_8
  have e_1464_9 := arithEq_of_rows h (row := 1464) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_9
  simp only [← c89772, ← c89773, k2, ← c89774] at e_1464_9
  have hr := e_1464_9
  simp only [e_1464_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14237 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1464 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14123)) := by
  have c89775 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89776 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89777 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89778 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89779 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89780 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1464_10 := arithEq_of_rows h (row := 1464) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_10
  simp only [← c89775, ← c89776, ← c89777] at e_1464_10
  have e_1464_11 := arithEq_of_rows h (row := 1464) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_11
  simp only [← c89778, ← c89779, k2, ← c89780] at e_1464_11
  have hr := e_1464_11
  simp only [e_1464_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14238 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1464 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14124)) := by
  have c89781 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89782 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89783 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89784 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89785 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89786 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1464_12 := arithEq_of_rows h (row := 1464) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_12
  simp only [← c89781, ← c89782, ← c89783] at e_1464_12
  have e_1464_13 := arithEq_of_rows h (row := 1464) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_13
  simp only [← c89784, ← c89785, k2, ← c89786] at e_1464_13
  have hr := e_1464_13
  simp only [e_1464_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14239 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1464 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14125)) := by
  have c89787 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89788 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89789 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89790 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89791 := (publicBatchWrapper86_copies2805 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89792 := (publicBatchWrapper86_copies2806 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1464_14 := arithEq_of_rows h (row := 1464) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_14
  simp only [← c89787, ← c89788, ← c89789] at e_1464_14
  have e_1464_15 := arithEq_of_rows h (row := 1464) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_15
  simp only [← c89790, ← c89791, k2, ← c89792] at e_1464_15
  have hr := e_1464_15
  simp only [e_1464_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14240 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1464 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14126)) := by
  have c89793 := (publicBatchWrapper86_copies2806 a h).2.1
  have c89794 := (publicBatchWrapper86_copies2806 a h).2.2.1
  have c89795 := (publicBatchWrapper86_copies2806 a h).2.2.2.1
  have c89796 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.1
  have c89797 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.1
  have c89798 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1464_16 := arithEq_of_rows h (row := 1464) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_16
  simp only [← c89793, ← c89794, ← c89795] at e_1464_16
  have e_1464_17 := arithEq_of_rows h (row := 1464) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_17
  simp only [← c89796, ← c89797, k2, ← c89798] at e_1464_17
  have hr := e_1464_17
  simp only [e_1464_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14241 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1464 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14127)) := by
  have c89799 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.1
  have c89800 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.1
  have c89801 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.1
  have c89802 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89803 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89804 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1464_18 := arithEq_of_rows h (row := 1464) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_18
  simp only [← c89799, ← c89800, ← c89801] at e_1464_18
  have e_1464_19 := arithEq_of_rows h (row := 1464) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1464_19
  simp only [← c89802, ← c89803, k2, ← c89804] at e_1464_19
  have hr := e_1464_19
  simp only [e_1464_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14242 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1465 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14128)) := by
  have c89805 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89806 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89807 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89808 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89809 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89810 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1465_0 := arithEq_of_rows h (row := 1465) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_0
  simp only [← c89805, ← c89806, ← c89807] at e_1465_0
  have e_1465_1 := arithEq_of_rows h (row := 1465) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_1
  simp only [← c89808, ← c89809, k2, ← c89810] at e_1465_1
  have hr := e_1465_1
  simp only [e_1465_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14243 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1465 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14129)) := by
  have c89811 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89812 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89813 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89814 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89815 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89816 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1465_2 := arithEq_of_rows h (row := 1465) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_2
  simp only [← c89811, ← c89812, ← c89813] at e_1465_2
  have e_1465_3 := arithEq_of_rows h (row := 1465) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_3
  simp only [← c89814, ← c89815, k2, ← c89816] at e_1465_3
  have hr := e_1465_3
  simp only [e_1465_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14244 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1465 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14130)) := by
  have c89817 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89818 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89819 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89820 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89821 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89822 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1465_4 := arithEq_of_rows h (row := 1465) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_4
  simp only [← c89817, ← c89818, ← c89819] at e_1465_4
  have e_1465_5 := arithEq_of_rows h (row := 1465) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_5
  simp only [← c89820, ← c89821, k2, ← c89822] at e_1465_5
  have hr := e_1465_5
  simp only [e_1465_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14245 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1465 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14131)) := by
  have c89823 := (publicBatchWrapper86_copies2806 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89824 := (publicBatchWrapper86_copies2807 a h).1
  have c89825 := (publicBatchWrapper86_copies2807 a h).2.1
  have c89826 := (publicBatchWrapper86_copies2807 a h).2.2.1
  have c89827 := (publicBatchWrapper86_copies2807 a h).2.2.2.1
  have c89828 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1465_6 := arithEq_of_rows h (row := 1465) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_6
  simp only [← c89823, ← c89824, ← c89825] at e_1465_6
  have e_1465_7 := arithEq_of_rows h (row := 1465) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_7
  simp only [← c89826, ← c89827, k2, ← c89828] at e_1465_7
  have hr := e_1465_7
  simp only [e_1465_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14246 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1465 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14132)) := by
  have c89829 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.1
  have c89830 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.1
  have c89831 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.1
  have c89832 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.1
  have c89833 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.1
  have c89834 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1465_8 := arithEq_of_rows h (row := 1465) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_8
  simp only [← c89829, ← c89830, ← c89831] at e_1465_8
  have e_1465_9 := arithEq_of_rows h (row := 1465) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_9
  simp only [← c89832, ← c89833, k2, ← c89834] at e_1465_9
  have hr := e_1465_9
  simp only [e_1465_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14247 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1465 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14133)) := by
  have c89835 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89836 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89837 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89838 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89839 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89840 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1465_10 := arithEq_of_rows h (row := 1465) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_10
  simp only [← c89835, ← c89836, ← c89837] at e_1465_10
  have e_1465_11 := arithEq_of_rows h (row := 1465) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_11
  simp only [← c89838, ← c89839, k2, ← c89840] at e_1465_11
  have hr := e_1465_11
  simp only [e_1465_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14248 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1465 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14134)) := by
  have c89841 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89842 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89843 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89844 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89845 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89846 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1465_12 := arithEq_of_rows h (row := 1465) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_12
  simp only [← c89841, ← c89842, ← c89843] at e_1465_12
  have e_1465_13 := arithEq_of_rows h (row := 1465) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_13
  simp only [← c89844, ← c89845, k2, ← c89846] at e_1465_13
  have hr := e_1465_13
  simp only [e_1465_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14249 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1465 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14135)) := by
  have c89847 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89848 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89849 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89850 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89851 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89852 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1465_14 := arithEq_of_rows h (row := 1465) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_14
  simp only [← c89847, ← c89848, ← c89849] at e_1465_14
  have e_1465_15 := arithEq_of_rows h (row := 1465) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_15
  simp only [← c89850, ← c89851, k2, ← c89852] at e_1465_15
  have hr := e_1465_15
  simp only [e_1465_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14250 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1465 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14136)) := by
  have c89853 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89854 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89855 := (publicBatchWrapper86_copies2807 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89856 := (publicBatchWrapper86_copies2808 a h).1
  have c89857 := (publicBatchWrapper86_copies2808 a h).2.1
  have c89858 := (publicBatchWrapper86_copies2808 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1465_16 := arithEq_of_rows h (row := 1465) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_16
  simp only [← c89853, ← c89854, ← c89855] at e_1465_16
  have e_1465_17 := arithEq_of_rows h (row := 1465) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_17
  simp only [← c89856, ← c89857, k2, ← c89858] at e_1465_17
  have hr := e_1465_17
  simp only [e_1465_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14251 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1465 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14137)) := by
  have c89859 := (publicBatchWrapper86_copies2808 a h).2.2.2.1
  have c89860 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.1
  have c89861 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.1
  have c89862 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.1
  have c89863 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.1
  have c89864 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1465_18 := arithEq_of_rows h (row := 1465) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_18
  simp only [← c89859, ← c89860, ← c89861] at e_1465_18
  have e_1465_19 := arithEq_of_rows h (row := 1465) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1465_19
  simp only [← c89862, ← c89863, k2, ← c89864] at e_1465_19
  have hr := e_1465_19
  simp only [e_1465_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14252 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1466 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14138)) := by
  have c89865 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.1
  have c89866 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89867 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89868 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89869 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89870 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1466_0 := arithEq_of_rows h (row := 1466) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_0
  simp only [← c89865, ← c89866, ← c89867] at e_1466_0
  have e_1466_1 := arithEq_of_rows h (row := 1466) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_1
  simp only [← c89868, ← c89869, k2, ← c89870] at e_1466_1
  have hr := e_1466_1
  simp only [e_1466_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14253 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1466 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14139)) := by
  have c89871 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89872 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89873 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89874 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89875 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89876 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1466_2 := arithEq_of_rows h (row := 1466) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_2
  simp only [← c89871, ← c89872, ← c89873] at e_1466_2
  have e_1466_3 := arithEq_of_rows h (row := 1466) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_3
  simp only [← c89874, ← c89875, k2, ← c89876] at e_1466_3
  have hr := e_1466_3
  simp only [e_1466_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14254 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1466 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14140)) := by
  have c89877 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89878 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89879 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89880 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89881 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89882 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1466_4 := arithEq_of_rows h (row := 1466) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_4
  simp only [← c89877, ← c89878, ← c89879] at e_1466_4
  have e_1466_5 := arithEq_of_rows h (row := 1466) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_5
  simp only [← c89880, ← c89881, k2, ← c89882] at e_1466_5
  have hr := e_1466_5
  simp only [e_1466_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14255 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1466 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14141)) := by
  have c89883 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89884 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89885 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89886 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89887 := (publicBatchWrapper86_copies2808 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89888 := (publicBatchWrapper86_copies2809 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1466_6 := arithEq_of_rows h (row := 1466) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_6
  simp only [← c89883, ← c89884, ← c89885] at e_1466_6
  have e_1466_7 := arithEq_of_rows h (row := 1466) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_7
  simp only [← c89886, ← c89887, k2, ← c89888] at e_1466_7
  have hr := e_1466_7
  simp only [e_1466_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14256 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1466 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14142)) := by
  have c89889 := (publicBatchWrapper86_copies2809 a h).2.1
  have c89890 := (publicBatchWrapper86_copies2809 a h).2.2.1
  have c89891 := (publicBatchWrapper86_copies2809 a h).2.2.2.1
  have c89892 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.1
  have c89893 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.1
  have c89894 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1466_8 := arithEq_of_rows h (row := 1466) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_8
  simp only [← c89889, ← c89890, ← c89891] at e_1466_8
  have e_1466_9 := arithEq_of_rows h (row := 1466) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_9
  simp only [← c89892, ← c89893, k2, ← c89894] at e_1466_9
  have hr := e_1466_9
  simp only [e_1466_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14257 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1466 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14143)) := by
  have c89895 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.1
  have c89896 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.1
  have c89897 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.1
  have c89898 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89899 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89900 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1466_10 := arithEq_of_rows h (row := 1466) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_10
  simp only [← c89895, ← c89896, ← c89897] at e_1466_10
  have e_1466_11 := arithEq_of_rows h (row := 1466) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_11
  simp only [← c89898, ← c89899, k2, ← c89900] at e_1466_11
  have hr := e_1466_11
  simp only [e_1466_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14258 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1466 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14144)) := by
  have c89901 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89902 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89903 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89904 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89905 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89906 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1466_12 := arithEq_of_rows h (row := 1466) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_12
  simp only [← c89901, ← c89902, ← c89903] at e_1466_12
  have e_1466_13 := arithEq_of_rows h (row := 1466) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_13
  simp only [← c89904, ← c89905, k2, ← c89906] at e_1466_13
  have hr := e_1466_13
  simp only [e_1466_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14259 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1466 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14145)) := by
  have c89907 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89908 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89909 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89910 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89911 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89912 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1466_14 := arithEq_of_rows h (row := 1466) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_14
  simp only [← c89907, ← c89908, ← c89909] at e_1466_14
  have e_1466_15 := arithEq_of_rows h (row := 1466) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_15
  simp only [← c89910, ← c89911, k2, ← c89912] at e_1466_15
  have hr := e_1466_15
  simp only [e_1466_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14260 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1466 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14146)) := by
  have c89913 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89914 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89915 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89916 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89917 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89918 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1466_16 := arithEq_of_rows h (row := 1466) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_16
  simp only [← c89913, ← c89914, ← c89915] at e_1466_16
  have e_1466_17 := arithEq_of_rows h (row := 1466) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_17
  simp only [← c89916, ← c89917, k2, ← c89918] at e_1466_17
  have hr := e_1466_17
  simp only [e_1466_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14261 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1466 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14147)) := by
  have c89919 := (publicBatchWrapper86_copies2809 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89920 := (publicBatchWrapper86_copies2810 a h).1
  have c89921 := (publicBatchWrapper86_copies2810 a h).2.1
  have c89922 := (publicBatchWrapper86_copies2810 a h).2.2.1
  have c89923 := (publicBatchWrapper86_copies2810 a h).2.2.2.1
  have c89924 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1466_18 := arithEq_of_rows h (row := 1466) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_18
  simp only [← c89919, ← c89920, ← c89921] at e_1466_18
  have e_1466_19 := arithEq_of_rows h (row := 1466) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1466_19
  simp only [← c89922, ← c89923, k2, ← c89924] at e_1466_19
  have hr := e_1466_19
  simp only [e_1466_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14262 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1467 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14148)) := by
  have c89925 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.1
  have c89926 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.1
  have c89927 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.1
  have c89928 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.1
  have c89929 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.1
  have c89930 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1467_0 := arithEq_of_rows h (row := 1467) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_0
  simp only [← c89925, ← c89926, ← c89927] at e_1467_0
  have e_1467_1 := arithEq_of_rows h (row := 1467) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_1
  simp only [← c89928, ← c89929, k2, ← c89930] at e_1467_1
  have hr := e_1467_1
  simp only [e_1467_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14263 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1467 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14149)) := by
  have c89931 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89932 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89933 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89934 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89935 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89936 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1467_2 := arithEq_of_rows h (row := 1467) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_2
  simp only [← c89931, ← c89932, ← c89933] at e_1467_2
  have e_1467_3 := arithEq_of_rows h (row := 1467) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_3
  simp only [← c89934, ← c89935, k2, ← c89936] at e_1467_3
  have hr := e_1467_3
  simp only [e_1467_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14264 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1467 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14150)) := by
  have c89937 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89938 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89939 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89940 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89941 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89942 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1467_4 := arithEq_of_rows h (row := 1467) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_4
  simp only [← c89937, ← c89938, ← c89939] at e_1467_4
  have e_1467_5 := arithEq_of_rows h (row := 1467) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_5
  simp only [← c89940, ← c89941, k2, ← c89942] at e_1467_5
  have hr := e_1467_5
  simp only [e_1467_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14265 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1467 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14151)) := by
  have c89943 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89944 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89945 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89946 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89947 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89948 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1467_6 := arithEq_of_rows h (row := 1467) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_6
  simp only [← c89943, ← c89944, ← c89945] at e_1467_6
  have e_1467_7 := arithEq_of_rows h (row := 1467) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_7
  simp only [← c89946, ← c89947, k2, ← c89948] at e_1467_7
  have hr := e_1467_7
  simp only [e_1467_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14266 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1467 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14152)) := by
  have c89949 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89950 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89951 := (publicBatchWrapper86_copies2810 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89952 := (publicBatchWrapper86_copies2811 a h).1
  have c89953 := (publicBatchWrapper86_copies2811 a h).2.1
  have c89954 := (publicBatchWrapper86_copies2811 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1467_8 := arithEq_of_rows h (row := 1467) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_8
  simp only [← c89949, ← c89950, ← c89951] at e_1467_8
  have e_1467_9 := arithEq_of_rows h (row := 1467) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_9
  simp only [← c89952, ← c89953, k2, ← c89954] at e_1467_9
  have hr := e_1467_9
  simp only [e_1467_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14267 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1467 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14153)) := by
  have c89955 := (publicBatchWrapper86_copies2811 a h).2.2.2.1
  have c89956 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.1
  have c89957 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.1
  have c89958 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.1
  have c89959 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.1
  have c89960 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1467_10 := arithEq_of_rows h (row := 1467) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_10
  simp only [← c89955, ← c89956, ← c89957] at e_1467_10
  have e_1467_11 := arithEq_of_rows h (row := 1467) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_11
  simp only [← c89958, ← c89959, k2, ← c89960] at e_1467_11
  have hr := e_1467_11
  simp only [e_1467_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14268 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1467 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14154)) := by
  have c89961 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.1
  have c89962 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89963 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89964 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89965 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89966 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1467_12 := arithEq_of_rows h (row := 1467) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_12
  simp only [← c89961, ← c89962, ← c89963] at e_1467_12
  have e_1467_13 := arithEq_of_rows h (row := 1467) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_13
  simp only [← c89964, ← c89965, k2, ← c89966] at e_1467_13
  have hr := e_1467_13
  simp only [e_1467_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14269 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1467 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14155)) := by
  have c89967 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89968 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89969 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89970 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89971 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89972 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1467_14 := arithEq_of_rows h (row := 1467) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_14
  simp only [← c89967, ← c89968, ← c89969] at e_1467_14
  have e_1467_15 := arithEq_of_rows h (row := 1467) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_15
  simp only [← c89970, ← c89971, k2, ← c89972] at e_1467_15
  have hr := e_1467_15
  simp only [e_1467_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14270 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1467 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14156)) := by
  have c89973 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89974 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89975 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89976 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89977 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89978 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1467_16 := arithEq_of_rows h (row := 1467) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_16
  simp only [← c89973, ← c89974, ← c89975] at e_1467_16
  have e_1467_17 := arithEq_of_rows h (row := 1467) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_17
  simp only [← c89976, ← c89977, k2, ← c89978] at e_1467_17
  have hr := e_1467_17
  simp only [e_1467_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14271 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1467 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14157)) := by
  have c89979 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89980 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89981 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89982 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89983 := (publicBatchWrapper86_copies2811 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c89984 := (publicBatchWrapper86_copies2812 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1467_18 := arithEq_of_rows h (row := 1467) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_18
  simp only [← c89979, ← c89980, ← c89981] at e_1467_18
  have e_1467_19 := arithEq_of_rows h (row := 1467) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1467_19
  simp only [← c89982, ← c89983, k2, ← c89984] at e_1467_19
  have hr := e_1467_19
  simp only [e_1467_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14272 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1468 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14158)) := by
  have c89985 := (publicBatchWrapper86_copies2812 a h).2.1
  have c89986 := (publicBatchWrapper86_copies2812 a h).2.2.1
  have c89987 := (publicBatchWrapper86_copies2812 a h).2.2.2.1
  have c89988 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.1
  have c89989 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.1
  have c89990 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1468_0 := arithEq_of_rows h (row := 1468) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_0
  simp only [← c89985, ← c89986, ← c89987] at e_1468_0
  have e_1468_1 := arithEq_of_rows h (row := 1468) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_1
  simp only [← c89988, ← c89989, k2, ← c89990] at e_1468_1
  have hr := e_1468_1
  simp only [e_1468_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14273 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1468 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14159)) := by
  have c89991 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.1
  have c89992 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.1
  have c89993 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.1
  have c89994 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.1
  have c89995 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c89996 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1468_2 := arithEq_of_rows h (row := 1468) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_2
  simp only [← c89991, ← c89992, ← c89993] at e_1468_2
  have e_1468_3 := arithEq_of_rows h (row := 1468) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_3
  simp only [← c89994, ← c89995, k2, ← c89996] at e_1468_3
  have hr := e_1468_3
  simp only [e_1468_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14274 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1468 23) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14160)) := by
  have c89997 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89998 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c89999 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90000 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90001 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90002 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1468_4 := arithEq_of_rows h (row := 1468) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_4
  simp only [← c89997, ← c89998, ← c89999] at e_1468_4
  have e_1468_5 := arithEq_of_rows h (row := 1468) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_5
  simp only [← c90000, ← c90001, k2, ← c90002] at e_1468_5
  have hr := e_1468_5
  simp only [e_1468_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14275 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1468 31) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14161)) := by
  have c90003 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90004 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90005 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90006 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90007 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90008 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1468_6 := arithEq_of_rows h (row := 1468) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_6
  simp only [← c90003, ← c90004, ← c90005] at e_1468_6
  have e_1468_7 := arithEq_of_rows h (row := 1468) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_7
  simp only [← c90006, ← c90007, k2, ← c90008] at e_1468_7
  have hr := e_1468_7
  simp only [e_1468_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14276 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1468 39) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14162)) := by
  have c90009 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90010 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90011 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90012 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90013 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90014 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1468_8 := arithEq_of_rows h (row := 1468) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_8
  simp only [← c90009, ← c90010, ← c90011] at e_1468_8
  have e_1468_9 := arithEq_of_rows h (row := 1468) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_9
  simp only [← c90012, ← c90013, k2, ← c90014] at e_1468_9
  have hr := e_1468_9
  simp only [e_1468_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14277 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1468 47) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14163)) := by
  have c90015 := (publicBatchWrapper86_copies2812 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90016 := (publicBatchWrapper86_copies2813 a h).1
  have c90017 := (publicBatchWrapper86_copies2813 a h).2.1
  have c90018 := (publicBatchWrapper86_copies2813 a h).2.2.1
  have c90019 := (publicBatchWrapper86_copies2813 a h).2.2.2.1
  have c90020 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1468_10 := arithEq_of_rows h (row := 1468) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_10
  simp only [← c90015, ← c90016, ← c90017] at e_1468_10
  have e_1468_11 := arithEq_of_rows h (row := 1468) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_11
  simp only [← c90018, ← c90019, k2, ← c90020] at e_1468_11
  have hr := e_1468_11
  simp only [e_1468_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14278 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1468 55) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14164)) := by
  have c90021 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.1
  have c90022 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.1
  have c90023 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.1
  have c90024 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.1
  have c90025 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.1
  have c90026 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1468_12 := arithEq_of_rows h (row := 1468) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_12
  simp only [← c90021, ← c90022, ← c90023] at e_1468_12
  have e_1468_13 := arithEq_of_rows h (row := 1468) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_13
  simp only [← c90024, ← c90025, k2, ← c90026] at e_1468_13
  have hr := e_1468_13
  simp only [e_1468_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14279 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1468 63) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14165)) := by
  have c90027 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90028 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90029 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90030 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90031 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90032 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1468_14 := arithEq_of_rows h (row := 1468) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_14
  simp only [← c90027, ← c90028, ← c90029] at e_1468_14
  have e_1468_15 := arithEq_of_rows h (row := 1468) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_15
  simp only [← c90030, ← c90031, k2, ← c90032] at e_1468_15
  have hr := e_1468_15
  simp only [e_1468_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14280 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1468 71) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14166)) := by
  have c90033 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90034 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90035 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90036 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90037 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90038 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1468_16 := arithEq_of_rows h (row := 1468) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_16
  simp only [← c90033, ← c90034, ← c90035] at e_1468_16
  have e_1468_17 := arithEq_of_rows h (row := 1468) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_17
  simp only [← c90036, ← c90037, k2, ← c90038] at e_1468_17
  have hr := e_1468_17
  simp only [e_1468_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14281 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1468 79) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14167)) := by
  have c90039 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90040 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90041 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90042 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90043 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90044 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1468_18 := arithEq_of_rows h (row := 1468) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_18
  simp only [← c90039, ← c90040, ← c90041] at e_1468_18
  have e_1468_19 := arithEq_of_rows h (row := 1468) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1468_19
  simp only [← c90042, ← c90043, k2, ← c90044] at e_1468_19
  have hr := e_1468_19
  simp only [e_1468_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14282 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1469 7) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14168)) := by
  have c90045 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90046 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90047 := (publicBatchWrapper86_copies2813 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90048 := (publicBatchWrapper86_copies2814 a h).1
  have c90049 := (publicBatchWrapper86_copies2814 a h).2.1
  have c90050 := (publicBatchWrapper86_copies2814 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1469_0 := arithEq_of_rows h (row := 1469) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_0
  simp only [← c90045, ← c90046, ← c90047] at e_1469_0
  have e_1469_1 := arithEq_of_rows h (row := 1469) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_1
  simp only [← c90048, ← c90049, k2, ← c90050] at e_1469_1
  have hr := e_1469_1
  simp only [e_1469_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14283 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1469 15) = bselect (a (.wire 68 67)) (a (.virt 200)) (a (.virt 14169)) := by
  have c90051 := (publicBatchWrapper86_copies2814 a h).2.2.2.1
  have c90052 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.1
  have c90053 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.1
  have c90054 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.1
  have c90055 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.1
  have c90056 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1469_2 := arithEq_of_rows h (row := 1469) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_2
  simp only [← c90051, ← c90052, ← c90053] at e_1469_2
  have e_1469_3 := arithEq_of_rows h (row := 1469) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_3
  simp only [← c90054, ← c90055, k2, ← c90056] at e_1469_3
  have hr := e_1469_3
  simp only [e_1469_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14284 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1469 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14210)) := by
  have c90057 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.1
  have c90058 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90059 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90060 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90061 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90062 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1469_4 := arithEq_of_rows h (row := 1469) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_4
  simp only [← c90057, ← c90058, ← c90059] at e_1469_4
  have e_1469_5 := arithEq_of_rows h (row := 1469) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_5
  simp only [← c90060, ← c90061, k2, ← c90062] at e_1469_5
  have hr := e_1469_5
  simp only [e_1469_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14285 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1469 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14211)) := by
  have c90063 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90064 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90065 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90066 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90067 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90068 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1469_6 := arithEq_of_rows h (row := 1469) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_6
  simp only [← c90063, ← c90064, ← c90065] at e_1469_6
  have e_1469_7 := arithEq_of_rows h (row := 1469) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_7
  simp only [← c90066, ← c90067, k2, ← c90068] at e_1469_7
  have hr := e_1469_7
  simp only [e_1469_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14286 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1469 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14212)) := by
  have c90069 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90070 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90071 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90072 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90073 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90074 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1469_8 := arithEq_of_rows h (row := 1469) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_8
  simp only [← c90069, ← c90070, ← c90071] at e_1469_8
  have e_1469_9 := arithEq_of_rows h (row := 1469) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_9
  simp only [← c90072, ← c90073, k2, ← c90074] at e_1469_9
  have hr := e_1469_9
  simp only [e_1469_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14287 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1469 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14213)) := by
  have c90075 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90076 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90077 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90078 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90079 := (publicBatchWrapper86_copies2814 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90080 := (publicBatchWrapper86_copies2815 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1469_10 := arithEq_of_rows h (row := 1469) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_10
  simp only [← c90075, ← c90076, ← c90077] at e_1469_10
  have e_1469_11 := arithEq_of_rows h (row := 1469) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_11
  simp only [← c90078, ← c90079, k2, ← c90080] at e_1469_11
  have hr := e_1469_11
  simp only [e_1469_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14288 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1469 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14214)) := by
  have c90081 := (publicBatchWrapper86_copies2815 a h).2.1
  have c90082 := (publicBatchWrapper86_copies2815 a h).2.2.1
  have c90083 := (publicBatchWrapper86_copies2815 a h).2.2.2.1
  have c90084 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.1
  have c90085 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.1
  have c90086 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1469_12 := arithEq_of_rows h (row := 1469) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_12
  simp only [← c90081, ← c90082, ← c90083] at e_1469_12
  have e_1469_13 := arithEq_of_rows h (row := 1469) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_13
  simp only [← c90084, ← c90085, k2, ← c90086] at e_1469_13
  have hr := e_1469_13
  simp only [e_1469_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14289 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1469 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14215)) := by
  have c90087 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.1
  have c90088 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.1
  have c90089 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.1
  have c90090 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90091 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90092 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1469_14 := arithEq_of_rows h (row := 1469) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_14
  simp only [← c90087, ← c90088, ← c90089] at e_1469_14
  have e_1469_15 := arithEq_of_rows h (row := 1469) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_15
  simp only [← c90090, ← c90091, k2, ← c90092] at e_1469_15
  have hr := e_1469_15
  simp only [e_1469_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14290 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1469 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14216)) := by
  have c90093 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90094 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90095 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90096 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90097 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90098 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1469_16 := arithEq_of_rows h (row := 1469) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_16
  simp only [← c90093, ← c90094, ← c90095] at e_1469_16
  have e_1469_17 := arithEq_of_rows h (row := 1469) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_17
  simp only [← c90096, ← c90097, k2, ← c90098] at e_1469_17
  have hr := e_1469_17
  simp only [e_1469_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14291 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1469 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14217)) := by
  have c90099 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90100 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90101 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90102 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90103 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90104 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1469_18 := arithEq_of_rows h (row := 1469) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_18
  simp only [← c90099, ← c90100, ← c90101] at e_1469_18
  have e_1469_19 := arithEq_of_rows h (row := 1469) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1469_19
  simp only [← c90102, ← c90103, k2, ← c90104] at e_1469_19
  have hr := e_1469_19
  simp only [e_1469_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14292 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1470 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14218)) := by
  have c90105 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90106 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90107 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90108 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90109 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90110 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1470_0 := arithEq_of_rows h (row := 1470) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_0
  simp only [← c90105, ← c90106, ← c90107] at e_1470_0
  have e_1470_1 := arithEq_of_rows h (row := 1470) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_1
  simp only [← c90108, ← c90109, k2, ← c90110] at e_1470_1
  have hr := e_1470_1
  simp only [e_1470_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14293 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1470 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14219)) := by
  have c90111 := (publicBatchWrapper86_copies2815 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90112 := (publicBatchWrapper86_copies2816 a h).1
  have c90113 := (publicBatchWrapper86_copies2816 a h).2.1
  have c90114 := (publicBatchWrapper86_copies2816 a h).2.2.1
  have c90115 := (publicBatchWrapper86_copies2816 a h).2.2.2.1
  have c90116 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1470_2 := arithEq_of_rows h (row := 1470) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_2
  simp only [← c90111, ← c90112, ← c90113] at e_1470_2
  have e_1470_3 := arithEq_of_rows h (row := 1470) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_3
  simp only [← c90114, ← c90115, k2, ← c90116] at e_1470_3
  have hr := e_1470_3
  simp only [e_1470_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14294 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1470 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14220)) := by
  have c90117 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.1
  have c90118 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.1
  have c90119 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.1
  have c90120 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.1
  have c90121 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.1
  have c90122 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1470_4 := arithEq_of_rows h (row := 1470) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_4
  simp only [← c90117, ← c90118, ← c90119] at e_1470_4
  have e_1470_5 := arithEq_of_rows h (row := 1470) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_5
  simp only [← c90120, ← c90121, k2, ← c90122] at e_1470_5
  have hr := e_1470_5
  simp only [e_1470_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14295 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1470 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14221)) := by
  have c90123 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90124 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90125 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90126 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90127 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90128 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1470_6 := arithEq_of_rows h (row := 1470) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_6
  simp only [← c90123, ← c90124, ← c90125] at e_1470_6
  have e_1470_7 := arithEq_of_rows h (row := 1470) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_7
  simp only [← c90126, ← c90127, k2, ← c90128] at e_1470_7
  have hr := e_1470_7
  simp only [e_1470_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14296 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1470 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14222)) := by
  have c90129 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90130 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90131 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90132 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90133 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90134 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1470_8 := arithEq_of_rows h (row := 1470) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_8
  simp only [← c90129, ← c90130, ← c90131] at e_1470_8
  have e_1470_9 := arithEq_of_rows h (row := 1470) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_9
  simp only [← c90132, ← c90133, k2, ← c90134] at e_1470_9
  have hr := e_1470_9
  simp only [e_1470_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14297 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1470 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14223)) := by
  have c90135 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90136 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90137 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90138 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90139 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90140 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1470_10 := arithEq_of_rows h (row := 1470) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_10
  simp only [← c90135, ← c90136, ← c90137] at e_1470_10
  have e_1470_11 := arithEq_of_rows h (row := 1470) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_11
  simp only [← c90138, ← c90139, k2, ← c90140] at e_1470_11
  have hr := e_1470_11
  simp only [e_1470_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14298 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1470 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14224)) := by
  have c90141 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90142 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90143 := (publicBatchWrapper86_copies2816 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90144 := (publicBatchWrapper86_copies2817 a h).1
  have c90145 := (publicBatchWrapper86_copies2817 a h).2.1
  have c90146 := (publicBatchWrapper86_copies2817 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1470_12 := arithEq_of_rows h (row := 1470) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_12
  simp only [← c90141, ← c90142, ← c90143] at e_1470_12
  have e_1470_13 := arithEq_of_rows h (row := 1470) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_13
  simp only [← c90144, ← c90145, k2, ← c90146] at e_1470_13
  have hr := e_1470_13
  simp only [e_1470_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14299 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1470 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14225)) := by
  have c90147 := (publicBatchWrapper86_copies2817 a h).2.2.2.1
  have c90148 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.1
  have c90149 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.1
  have c90150 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.1
  have c90151 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.1
  have c90152 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1470_14 := arithEq_of_rows h (row := 1470) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_14
  simp only [← c90147, ← c90148, ← c90149] at e_1470_14
  have e_1470_15 := arithEq_of_rows h (row := 1470) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_15
  simp only [← c90150, ← c90151, k2, ← c90152] at e_1470_15
  have hr := e_1470_15
  simp only [e_1470_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14300 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1470 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14226)) := by
  have c90153 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.1
  have c90154 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90155 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90156 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90157 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90158 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1470_16 := arithEq_of_rows h (row := 1470) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_16
  simp only [← c90153, ← c90154, ← c90155] at e_1470_16
  have e_1470_17 := arithEq_of_rows h (row := 1470) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_17
  simp only [← c90156, ← c90157, k2, ← c90158] at e_1470_17
  have hr := e_1470_17
  simp only [e_1470_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14301 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1470 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14227)) := by
  have c90159 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90160 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90161 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90162 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90163 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90164 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1470_18 := arithEq_of_rows h (row := 1470) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_18
  simp only [← c90159, ← c90160, ← c90161] at e_1470_18
  have e_1470_19 := arithEq_of_rows h (row := 1470) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1470_19
  simp only [← c90162, ← c90163, k2, ← c90164] at e_1470_19
  have hr := e_1470_19
  simp only [e_1470_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14302 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1471 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14228)) := by
  have c90165 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90166 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90167 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90168 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90169 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90170 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1471_0 := arithEq_of_rows h (row := 1471) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_0
  simp only [← c90165, ← c90166, ← c90167] at e_1471_0
  have e_1471_1 := arithEq_of_rows h (row := 1471) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_1
  simp only [← c90168, ← c90169, k2, ← c90170] at e_1471_1
  have hr := e_1471_1
  simp only [e_1471_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14303 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1471 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14229)) := by
  have c90171 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90172 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90173 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90174 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90175 := (publicBatchWrapper86_copies2817 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90176 := (publicBatchWrapper86_copies2818 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1471_2 := arithEq_of_rows h (row := 1471) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_2
  simp only [← c90171, ← c90172, ← c90173] at e_1471_2
  have e_1471_3 := arithEq_of_rows h (row := 1471) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_3
  simp only [← c90174, ← c90175, k2, ← c90176] at e_1471_3
  have hr := e_1471_3
  simp only [e_1471_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14304 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1471 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14230)) := by
  have c90177 := (publicBatchWrapper86_copies2818 a h).2.1
  have c90178 := (publicBatchWrapper86_copies2818 a h).2.2.1
  have c90179 := (publicBatchWrapper86_copies2818 a h).2.2.2.1
  have c90180 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.1
  have c90181 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.1
  have c90182 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1471_4 := arithEq_of_rows h (row := 1471) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_4
  simp only [← c90177, ← c90178, ← c90179] at e_1471_4
  have e_1471_5 := arithEq_of_rows h (row := 1471) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_5
  simp only [← c90180, ← c90181, k2, ← c90182] at e_1471_5
  have hr := e_1471_5
  simp only [e_1471_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14305 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1471 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14231)) := by
  have c90183 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.1
  have c90184 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.1
  have c90185 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.1
  have c90186 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90187 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90188 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1471_6 := arithEq_of_rows h (row := 1471) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_6
  simp only [← c90183, ← c90184, ← c90185] at e_1471_6
  have e_1471_7 := arithEq_of_rows h (row := 1471) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_7
  simp only [← c90186, ← c90187, k2, ← c90188] at e_1471_7
  have hr := e_1471_7
  simp only [e_1471_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14306 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1471 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14232)) := by
  have c90189 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90190 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90191 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90192 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90193 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90194 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1471_8 := arithEq_of_rows h (row := 1471) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_8
  simp only [← c90189, ← c90190, ← c90191] at e_1471_8
  have e_1471_9 := arithEq_of_rows h (row := 1471) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_9
  simp only [← c90192, ← c90193, k2, ← c90194] at e_1471_9
  have hr := e_1471_9
  simp only [e_1471_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14307 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1471 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14233)) := by
  have c90195 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90196 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90197 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90198 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90199 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90200 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1471_10 := arithEq_of_rows h (row := 1471) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_10
  simp only [← c90195, ← c90196, ← c90197] at e_1471_10
  have e_1471_11 := arithEq_of_rows h (row := 1471) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_11
  simp only [← c90198, ← c90199, k2, ← c90200] at e_1471_11
  have hr := e_1471_11
  simp only [e_1471_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14308 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1471 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14234)) := by
  have c90201 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90202 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90203 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90204 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90205 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90206 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1471_12 := arithEq_of_rows h (row := 1471) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_12
  simp only [← c90201, ← c90202, ← c90203] at e_1471_12
  have e_1471_13 := arithEq_of_rows h (row := 1471) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_13
  simp only [← c90204, ← c90205, k2, ← c90206] at e_1471_13
  have hr := e_1471_13
  simp only [e_1471_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14309 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1471 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14235)) := by
  have c90207 := (publicBatchWrapper86_copies2818 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90208 := (publicBatchWrapper86_copies2819 a h).1
  have c90209 := (publicBatchWrapper86_copies2819 a h).2.1
  have c90210 := (publicBatchWrapper86_copies2819 a h).2.2.1
  have c90211 := (publicBatchWrapper86_copies2819 a h).2.2.2.1
  have c90212 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1471_14 := arithEq_of_rows h (row := 1471) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_14
  simp only [← c90207, ← c90208, ← c90209] at e_1471_14
  have e_1471_15 := arithEq_of_rows h (row := 1471) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_15
  simp only [← c90210, ← c90211, k2, ← c90212] at e_1471_15
  have hr := e_1471_15
  simp only [e_1471_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14310 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1471 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14236)) := by
  have c90213 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.1
  have c90214 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.1
  have c90215 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.1
  have c90216 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.1
  have c90217 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.1
  have c90218 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1471_16 := arithEq_of_rows h (row := 1471) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_16
  simp only [← c90213, ← c90214, ← c90215] at e_1471_16
  have e_1471_17 := arithEq_of_rows h (row := 1471) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_17
  simp only [← c90216, ← c90217, k2, ← c90218] at e_1471_17
  have hr := e_1471_17
  simp only [e_1471_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14311 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1471 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14237)) := by
  have c90219 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90220 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90221 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90222 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90223 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90224 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1471_18 := arithEq_of_rows h (row := 1471) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_18
  simp only [← c90219, ← c90220, ← c90221] at e_1471_18
  have e_1471_19 := arithEq_of_rows h (row := 1471) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1471_19
  simp only [← c90222, ← c90223, k2, ← c90224] at e_1471_19
  have hr := e_1471_19
  simp only [e_1471_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14312 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1472 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14238)) := by
  have c90225 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90226 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90227 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90228 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90229 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90230 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1472_0 := arithEq_of_rows h (row := 1472) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_0
  simp only [← c90225, ← c90226, ← c90227] at e_1472_0
  have e_1472_1 := arithEq_of_rows h (row := 1472) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_1
  simp only [← c90228, ← c90229, k2, ← c90230] at e_1472_1
  have hr := e_1472_1
  simp only [e_1472_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14313 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1472 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14239)) := by
  have c90231 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90232 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90233 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90234 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90235 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90236 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1472_2 := arithEq_of_rows h (row := 1472) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_2
  simp only [← c90231, ← c90232, ← c90233] at e_1472_2
  have e_1472_3 := arithEq_of_rows h (row := 1472) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_3
  simp only [← c90234, ← c90235, k2, ← c90236] at e_1472_3
  have hr := e_1472_3
  simp only [e_1472_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14314 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1472 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14240)) := by
  have c90237 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90238 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90239 := (publicBatchWrapper86_copies2819 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90240 := (publicBatchWrapper86_copies2820 a h).1
  have c90241 := (publicBatchWrapper86_copies2820 a h).2.1
  have c90242 := (publicBatchWrapper86_copies2820 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1472_4 := arithEq_of_rows h (row := 1472) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_4
  simp only [← c90237, ← c90238, ← c90239] at e_1472_4
  have e_1472_5 := arithEq_of_rows h (row := 1472) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_5
  simp only [← c90240, ← c90241, k2, ← c90242] at e_1472_5
  have hr := e_1472_5
  simp only [e_1472_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14315 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1472 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14241)) := by
  have c90243 := (publicBatchWrapper86_copies2820 a h).2.2.2.1
  have c90244 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.1
  have c90245 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.1
  have c90246 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.1
  have c90247 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.1
  have c90248 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1472_6 := arithEq_of_rows h (row := 1472) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_6
  simp only [← c90243, ← c90244, ← c90245] at e_1472_6
  have e_1472_7 := arithEq_of_rows h (row := 1472) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_7
  simp only [← c90246, ← c90247, k2, ← c90248] at e_1472_7
  have hr := e_1472_7
  simp only [e_1472_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14316 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1472 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14242)) := by
  have c90249 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.1
  have c90250 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90251 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90252 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90253 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90254 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1472_8 := arithEq_of_rows h (row := 1472) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_8
  simp only [← c90249, ← c90250, ← c90251] at e_1472_8
  have e_1472_9 := arithEq_of_rows h (row := 1472) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_9
  simp only [← c90252, ← c90253, k2, ← c90254] at e_1472_9
  have hr := e_1472_9
  simp only [e_1472_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14317 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1472 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14243)) := by
  have c90255 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90256 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90257 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90258 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90259 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90260 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1472_10 := arithEq_of_rows h (row := 1472) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_10
  simp only [← c90255, ← c90256, ← c90257] at e_1472_10
  have e_1472_11 := arithEq_of_rows h (row := 1472) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_11
  simp only [← c90258, ← c90259, k2, ← c90260] at e_1472_11
  have hr := e_1472_11
  simp only [e_1472_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14318 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1472 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14244)) := by
  have c90261 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90262 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90263 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90264 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90265 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90266 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1472_12 := arithEq_of_rows h (row := 1472) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_12
  simp only [← c90261, ← c90262, ← c90263] at e_1472_12
  have e_1472_13 := arithEq_of_rows h (row := 1472) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_13
  simp only [← c90264, ← c90265, k2, ← c90266] at e_1472_13
  have hr := e_1472_13
  simp only [e_1472_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14319 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1472 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14245)) := by
  have c90267 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90268 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90269 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90270 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90271 := (publicBatchWrapper86_copies2820 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90272 := (publicBatchWrapper86_copies2821 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1472_14 := arithEq_of_rows h (row := 1472) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_14
  simp only [← c90267, ← c90268, ← c90269] at e_1472_14
  have e_1472_15 := arithEq_of_rows h (row := 1472) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_15
  simp only [← c90270, ← c90271, k2, ← c90272] at e_1472_15
  have hr := e_1472_15
  simp only [e_1472_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14320 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1472 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14246)) := by
  have c90273 := (publicBatchWrapper86_copies2821 a h).2.1
  have c90274 := (publicBatchWrapper86_copies2821 a h).2.2.1
  have c90275 := (publicBatchWrapper86_copies2821 a h).2.2.2.1
  have c90276 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.1
  have c90277 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.1
  have c90278 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1472_16 := arithEq_of_rows h (row := 1472) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_16
  simp only [← c90273, ← c90274, ← c90275] at e_1472_16
  have e_1472_17 := arithEq_of_rows h (row := 1472) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_17
  simp only [← c90276, ← c90277, k2, ← c90278] at e_1472_17
  have hr := e_1472_17
  simp only [e_1472_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14321 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1472 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14247)) := by
  have c90279 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.1
  have c90280 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.1
  have c90281 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.1
  have c90282 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90283 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90284 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1472_18 := arithEq_of_rows h (row := 1472) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_18
  simp only [← c90279, ← c90280, ← c90281] at e_1472_18
  have e_1472_19 := arithEq_of_rows h (row := 1472) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1472_19
  simp only [← c90282, ← c90283, k2, ← c90284] at e_1472_19
  have hr := e_1472_19
  simp only [e_1472_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14322 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1473 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14248)) := by
  have c90285 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90286 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90287 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90288 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90289 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90290 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1473_0 := arithEq_of_rows h (row := 1473) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_0
  simp only [← c90285, ← c90286, ← c90287] at e_1473_0
  have e_1473_1 := arithEq_of_rows h (row := 1473) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_1
  simp only [← c90288, ← c90289, k2, ← c90290] at e_1473_1
  have hr := e_1473_1
  simp only [e_1473_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14323 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1473 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14249)) := by
  have c90291 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90292 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90293 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90294 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90295 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90296 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1473_2 := arithEq_of_rows h (row := 1473) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_2
  simp only [← c90291, ← c90292, ← c90293] at e_1473_2
  have e_1473_3 := arithEq_of_rows h (row := 1473) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_3
  simp only [← c90294, ← c90295, k2, ← c90296] at e_1473_3
  have hr := e_1473_3
  simp only [e_1473_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14324 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1473 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14250)) := by
  have c90297 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90298 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90299 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90300 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90301 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90302 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1473_4 := arithEq_of_rows h (row := 1473) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_4
  simp only [← c90297, ← c90298, ← c90299] at e_1473_4
  have e_1473_5 := arithEq_of_rows h (row := 1473) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_5
  simp only [← c90300, ← c90301, k2, ← c90302] at e_1473_5
  have hr := e_1473_5
  simp only [e_1473_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14325 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1473 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14251)) := by
  have c90303 := (publicBatchWrapper86_copies2821 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90304 := (publicBatchWrapper86_copies2822 a h).1
  have c90305 := (publicBatchWrapper86_copies2822 a h).2.1
  have c90306 := (publicBatchWrapper86_copies2822 a h).2.2.1
  have c90307 := (publicBatchWrapper86_copies2822 a h).2.2.2.1
  have c90308 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1473_6 := arithEq_of_rows h (row := 1473) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_6
  simp only [← c90303, ← c90304, ← c90305] at e_1473_6
  have e_1473_7 := arithEq_of_rows h (row := 1473) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_7
  simp only [← c90306, ← c90307, k2, ← c90308] at e_1473_7
  have hr := e_1473_7
  simp only [e_1473_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14326 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1473 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14252)) := by
  have c90309 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.1
  have c90310 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.1
  have c90311 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.1
  have c90312 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.1
  have c90313 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.1
  have c90314 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1473_8 := arithEq_of_rows h (row := 1473) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_8
  simp only [← c90309, ← c90310, ← c90311] at e_1473_8
  have e_1473_9 := arithEq_of_rows h (row := 1473) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_9
  simp only [← c90312, ← c90313, k2, ← c90314] at e_1473_9
  have hr := e_1473_9
  simp only [e_1473_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14327 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1473 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14253)) := by
  have c90315 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90316 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90317 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90318 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90319 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90320 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1473_10 := arithEq_of_rows h (row := 1473) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_10
  simp only [← c90315, ← c90316, ← c90317] at e_1473_10
  have e_1473_11 := arithEq_of_rows h (row := 1473) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_11
  simp only [← c90318, ← c90319, k2, ← c90320] at e_1473_11
  have hr := e_1473_11
  simp only [e_1473_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14328 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1473 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14254)) := by
  have c90321 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90322 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90323 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90324 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90325 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90326 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1473_12 := arithEq_of_rows h (row := 1473) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_12
  simp only [← c90321, ← c90322, ← c90323] at e_1473_12
  have e_1473_13 := arithEq_of_rows h (row := 1473) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_13
  simp only [← c90324, ← c90325, k2, ← c90326] at e_1473_13
  have hr := e_1473_13
  simp only [e_1473_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14329 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1473 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14255)) := by
  have c90327 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90328 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90329 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90330 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90331 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90332 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1473_14 := arithEq_of_rows h (row := 1473) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_14
  simp only [← c90327, ← c90328, ← c90329] at e_1473_14
  have e_1473_15 := arithEq_of_rows h (row := 1473) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_15
  simp only [← c90330, ← c90331, k2, ← c90332] at e_1473_15
  have hr := e_1473_15
  simp only [e_1473_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14330 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1473 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14256)) := by
  have c90333 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90334 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90335 := (publicBatchWrapper86_copies2822 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90336 := (publicBatchWrapper86_copies2823 a h).1
  have c90337 := (publicBatchWrapper86_copies2823 a h).2.1
  have c90338 := (publicBatchWrapper86_copies2823 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1473_16 := arithEq_of_rows h (row := 1473) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_16
  simp only [← c90333, ← c90334, ← c90335] at e_1473_16
  have e_1473_17 := arithEq_of_rows h (row := 1473) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_17
  simp only [← c90336, ← c90337, k2, ← c90338] at e_1473_17
  have hr := e_1473_17
  simp only [e_1473_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14331 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1473 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14257)) := by
  have c90339 := (publicBatchWrapper86_copies2823 a h).2.2.2.1
  have c90340 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.1
  have c90341 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.1
  have c90342 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.1
  have c90343 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.1
  have c90344 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1473_18 := arithEq_of_rows h (row := 1473) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_18
  simp only [← c90339, ← c90340, ← c90341] at e_1473_18
  have e_1473_19 := arithEq_of_rows h (row := 1473) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1473_19
  simp only [← c90342, ← c90343, k2, ← c90344] at e_1473_19
  have hr := e_1473_19
  simp only [e_1473_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14332 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1474 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14258)) := by
  have c90345 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.1
  have c90346 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90347 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90348 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90349 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90350 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1474_0 := arithEq_of_rows h (row := 1474) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_0
  simp only [← c90345, ← c90346, ← c90347] at e_1474_0
  have e_1474_1 := arithEq_of_rows h (row := 1474) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_1
  simp only [← c90348, ← c90349, k2, ← c90350] at e_1474_1
  have hr := e_1474_1
  simp only [e_1474_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14333 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1474 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14259)) := by
  have c90351 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90352 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90353 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90354 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90355 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90356 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1474_2 := arithEq_of_rows h (row := 1474) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_2
  simp only [← c90351, ← c90352, ← c90353] at e_1474_2
  have e_1474_3 := arithEq_of_rows h (row := 1474) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_3
  simp only [← c90354, ← c90355, k2, ← c90356] at e_1474_3
  have hr := e_1474_3
  simp only [e_1474_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14334 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1474 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14260)) := by
  have c90357 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90358 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90359 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90360 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90361 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90362 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1474_4 := arithEq_of_rows h (row := 1474) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_4
  simp only [← c90357, ← c90358, ← c90359] at e_1474_4
  have e_1474_5 := arithEq_of_rows h (row := 1474) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_5
  simp only [← c90360, ← c90361, k2, ← c90362] at e_1474_5
  have hr := e_1474_5
  simp only [e_1474_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14335 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1474 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14261)) := by
  have c90363 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90364 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90365 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90366 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90367 := (publicBatchWrapper86_copies2823 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90368 := (publicBatchWrapper86_copies2824 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1474_6 := arithEq_of_rows h (row := 1474) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_6
  simp only [← c90363, ← c90364, ← c90365] at e_1474_6
  have e_1474_7 := arithEq_of_rows h (row := 1474) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_7
  simp only [← c90366, ← c90367, k2, ← c90368] at e_1474_7
  have hr := e_1474_7
  simp only [e_1474_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14336 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1474 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14262)) := by
  have c90369 := (publicBatchWrapper86_copies2824 a h).2.1
  have c90370 := (publicBatchWrapper86_copies2824 a h).2.2.1
  have c90371 := (publicBatchWrapper86_copies2824 a h).2.2.2.1
  have c90372 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.1
  have c90373 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.1
  have c90374 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1474_8 := arithEq_of_rows h (row := 1474) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_8
  simp only [← c90369, ← c90370, ← c90371] at e_1474_8
  have e_1474_9 := arithEq_of_rows h (row := 1474) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_9
  simp only [← c90372, ← c90373, k2, ← c90374] at e_1474_9
  have hr := e_1474_9
  simp only [e_1474_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14337 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1474 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14263)) := by
  have c90375 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.1
  have c90376 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.1
  have c90377 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.1
  have c90378 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90379 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90380 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1474_10 := arithEq_of_rows h (row := 1474) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_10
  simp only [← c90375, ← c90376, ← c90377] at e_1474_10
  have e_1474_11 := arithEq_of_rows h (row := 1474) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_11
  simp only [← c90378, ← c90379, k2, ← c90380] at e_1474_11
  have hr := e_1474_11
  simp only [e_1474_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14338 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1474 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14264)) := by
  have c90381 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90382 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90383 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90384 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90385 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90386 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1474_12 := arithEq_of_rows h (row := 1474) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_12
  simp only [← c90381, ← c90382, ← c90383] at e_1474_12
  have e_1474_13 := arithEq_of_rows h (row := 1474) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_13
  simp only [← c90384, ← c90385, k2, ← c90386] at e_1474_13
  have hr := e_1474_13
  simp only [e_1474_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14339 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1474 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14265)) := by
  have c90387 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90388 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90389 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90390 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90391 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90392 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1474_14 := arithEq_of_rows h (row := 1474) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_14
  simp only [← c90387, ← c90388, ← c90389] at e_1474_14
  have e_1474_15 := arithEq_of_rows h (row := 1474) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_15
  simp only [← c90390, ← c90391, k2, ← c90392] at e_1474_15
  have hr := e_1474_15
  simp only [e_1474_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14340 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1474 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14266)) := by
  have c90393 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90394 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90395 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90396 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90397 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90398 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1474_16 := arithEq_of_rows h (row := 1474) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_16
  simp only [← c90393, ← c90394, ← c90395] at e_1474_16
  have e_1474_17 := arithEq_of_rows h (row := 1474) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_17
  simp only [← c90396, ← c90397, k2, ← c90398] at e_1474_17
  have hr := e_1474_17
  simp only [e_1474_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14341 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1474 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14267)) := by
  have c90399 := (publicBatchWrapper86_copies2824 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90400 := (publicBatchWrapper86_copies2825 a h).1
  have c90401 := (publicBatchWrapper86_copies2825 a h).2.1
  have c90402 := (publicBatchWrapper86_copies2825 a h).2.2.1
  have c90403 := (publicBatchWrapper86_copies2825 a h).2.2.2.1
  have c90404 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1474_18 := arithEq_of_rows h (row := 1474) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_18
  simp only [← c90399, ← c90400, ← c90401] at e_1474_18
  have e_1474_19 := arithEq_of_rows h (row := 1474) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1474_19
  simp only [← c90402, ← c90403, k2, ← c90404] at e_1474_19
  have hr := e_1474_19
  simp only [e_1474_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14342 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1475 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14268)) := by
  have c90405 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.1
  have c90406 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.1
  have c90407 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.1
  have c90408 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.1
  have c90409 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.1
  have c90410 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1475_0 := arithEq_of_rows h (row := 1475) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_0
  simp only [← c90405, ← c90406, ← c90407] at e_1475_0
  have e_1475_1 := arithEq_of_rows h (row := 1475) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_1
  simp only [← c90408, ← c90409, k2, ← c90410] at e_1475_1
  have hr := e_1475_1
  simp only [e_1475_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14343 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1475 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14269)) := by
  have c90411 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90412 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90413 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90414 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90415 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90416 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1475_2 := arithEq_of_rows h (row := 1475) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_2
  simp only [← c90411, ← c90412, ← c90413] at e_1475_2
  have e_1475_3 := arithEq_of_rows h (row := 1475) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_3
  simp only [← c90414, ← c90415, k2, ← c90416] at e_1475_3
  have hr := e_1475_3
  simp only [e_1475_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14344 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1475 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14270)) := by
  have c90417 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90418 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90419 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90420 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90421 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90422 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1475_4 := arithEq_of_rows h (row := 1475) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_4
  simp only [← c90417, ← c90418, ← c90419] at e_1475_4
  have e_1475_5 := arithEq_of_rows h (row := 1475) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_5
  simp only [← c90420, ← c90421, k2, ← c90422] at e_1475_5
  have hr := e_1475_5
  simp only [e_1475_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14345 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1475 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14271)) := by
  have c90423 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90424 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90425 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90426 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90427 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90428 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1475_6 := arithEq_of_rows h (row := 1475) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_6
  simp only [← c90423, ← c90424, ← c90425] at e_1475_6
  have e_1475_7 := arithEq_of_rows h (row := 1475) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_7
  simp only [← c90426, ← c90427, k2, ← c90428] at e_1475_7
  have hr := e_1475_7
  simp only [e_1475_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14346 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1475 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14272)) := by
  have c90429 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90430 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90431 := (publicBatchWrapper86_copies2825 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90432 := (publicBatchWrapper86_copies2826 a h).1
  have c90433 := (publicBatchWrapper86_copies2826 a h).2.1
  have c90434 := (publicBatchWrapper86_copies2826 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1475_8 := arithEq_of_rows h (row := 1475) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_8
  simp only [← c90429, ← c90430, ← c90431] at e_1475_8
  have e_1475_9 := arithEq_of_rows h (row := 1475) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_9
  simp only [← c90432, ← c90433, k2, ← c90434] at e_1475_9
  have hr := e_1475_9
  simp only [e_1475_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14347 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1475 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14273)) := by
  have c90435 := (publicBatchWrapper86_copies2826 a h).2.2.2.1
  have c90436 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.1
  have c90437 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.1
  have c90438 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.1
  have c90439 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.1
  have c90440 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1475_10 := arithEq_of_rows h (row := 1475) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_10
  simp only [← c90435, ← c90436, ← c90437] at e_1475_10
  have e_1475_11 := arithEq_of_rows h (row := 1475) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_11
  simp only [← c90438, ← c90439, k2, ← c90440] at e_1475_11
  have hr := e_1475_11
  simp only [e_1475_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14348 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1475 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14274)) := by
  have c90441 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.1
  have c90442 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90443 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90444 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90445 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90446 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1475_12 := arithEq_of_rows h (row := 1475) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_12
  simp only [← c90441, ← c90442, ← c90443] at e_1475_12
  have e_1475_13 := arithEq_of_rows h (row := 1475) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_13
  simp only [← c90444, ← c90445, k2, ← c90446] at e_1475_13
  have hr := e_1475_13
  simp only [e_1475_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14349 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1475 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14275)) := by
  have c90447 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90448 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90449 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90450 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90451 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90452 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1475_14 := arithEq_of_rows h (row := 1475) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_14
  simp only [← c90447, ← c90448, ← c90449] at e_1475_14
  have e_1475_15 := arithEq_of_rows h (row := 1475) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_15
  simp only [← c90450, ← c90451, k2, ← c90452] at e_1475_15
  have hr := e_1475_15
  simp only [e_1475_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14350 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1475 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14276)) := by
  have c90453 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90454 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90455 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90456 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90457 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90458 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1475_16 := arithEq_of_rows h (row := 1475) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_16
  simp only [← c90453, ← c90454, ← c90455] at e_1475_16
  have e_1475_17 := arithEq_of_rows h (row := 1475) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_17
  simp only [← c90456, ← c90457, k2, ← c90458] at e_1475_17
  have hr := e_1475_17
  simp only [e_1475_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14351 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1475 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14277)) := by
  have c90459 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90460 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90461 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90462 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90463 := (publicBatchWrapper86_copies2826 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90464 := (publicBatchWrapper86_copies2827 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1475_18 := arithEq_of_rows h (row := 1475) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_18
  simp only [← c90459, ← c90460, ← c90461] at e_1475_18
  have e_1475_19 := arithEq_of_rows h (row := 1475) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1475_19
  simp only [← c90462, ← c90463, k2, ← c90464] at e_1475_19
  have hr := e_1475_19
  simp only [e_1475_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14352 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1476 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14278)) := by
  have c90465 := (publicBatchWrapper86_copies2827 a h).2.1
  have c90466 := (publicBatchWrapper86_copies2827 a h).2.2.1
  have c90467 := (publicBatchWrapper86_copies2827 a h).2.2.2.1
  have c90468 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.1
  have c90469 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.1
  have c90470 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1476_0 := arithEq_of_rows h (row := 1476) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_0
  simp only [← c90465, ← c90466, ← c90467] at e_1476_0
  have e_1476_1 := arithEq_of_rows h (row := 1476) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_1
  simp only [← c90468, ← c90469, k2, ← c90470] at e_1476_1
  have hr := e_1476_1
  simp only [e_1476_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14353 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1476 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14279)) := by
  have c90471 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.1
  have c90472 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.1
  have c90473 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.1
  have c90474 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90475 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90476 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1476_2 := arithEq_of_rows h (row := 1476) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_2
  simp only [← c90471, ← c90472, ← c90473] at e_1476_2
  have e_1476_3 := arithEq_of_rows h (row := 1476) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_3
  simp only [← c90474, ← c90475, k2, ← c90476] at e_1476_3
  have hr := e_1476_3
  simp only [e_1476_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14354 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1476 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14280)) := by
  have c90477 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90478 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90479 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90480 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90481 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90482 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1476_4 := arithEq_of_rows h (row := 1476) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_4
  simp only [← c90477, ← c90478, ← c90479] at e_1476_4
  have e_1476_5 := arithEq_of_rows h (row := 1476) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_5
  simp only [← c90480, ← c90481, k2, ← c90482] at e_1476_5
  have hr := e_1476_5
  simp only [e_1476_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14355 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1476 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14281)) := by
  have c90483 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90484 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90485 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90486 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90487 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90488 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1476_6 := arithEq_of_rows h (row := 1476) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_6
  simp only [← c90483, ← c90484, ← c90485] at e_1476_6
  have e_1476_7 := arithEq_of_rows h (row := 1476) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_7
  simp only [← c90486, ← c90487, k2, ← c90488] at e_1476_7
  have hr := e_1476_7
  simp only [e_1476_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14356 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1476 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14282)) := by
  have c90489 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90490 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90491 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90492 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90493 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90494 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1476_8 := arithEq_of_rows h (row := 1476) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_8
  simp only [← c90489, ← c90490, ← c90491] at e_1476_8
  have e_1476_9 := arithEq_of_rows h (row := 1476) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_9
  simp only [← c90492, ← c90493, k2, ← c90494] at e_1476_9
  have hr := e_1476_9
  simp only [e_1476_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14357 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1476 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14283)) := by
  have c90495 := (publicBatchWrapper86_copies2827 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90496 := (publicBatchWrapper86_copies2828 a h).1
  have c90497 := (publicBatchWrapper86_copies2828 a h).2.1
  have c90498 := (publicBatchWrapper86_copies2828 a h).2.2.1
  have c90499 := (publicBatchWrapper86_copies2828 a h).2.2.2.1
  have c90500 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1476_10 := arithEq_of_rows h (row := 1476) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_10
  simp only [← c90495, ← c90496, ← c90497] at e_1476_10
  have e_1476_11 := arithEq_of_rows h (row := 1476) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_11
  simp only [← c90498, ← c90499, k2, ← c90500] at e_1476_11
  have hr := e_1476_11
  simp only [e_1476_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14358 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1476 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14284)) := by
  have c90501 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.1
  have c90502 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.1
  have c90503 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.1
  have c90504 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.1
  have c90505 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.1
  have c90506 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1476_12 := arithEq_of_rows h (row := 1476) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_12
  simp only [← c90501, ← c90502, ← c90503] at e_1476_12
  have e_1476_13 := arithEq_of_rows h (row := 1476) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_13
  simp only [← c90504, ← c90505, k2, ← c90506] at e_1476_13
  have hr := e_1476_13
  simp only [e_1476_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14359 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1476 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14285)) := by
  have c90507 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90508 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90509 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90510 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90511 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90512 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1476_14 := arithEq_of_rows h (row := 1476) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_14
  simp only [← c90507, ← c90508, ← c90509] at e_1476_14
  have e_1476_15 := arithEq_of_rows h (row := 1476) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_15
  simp only [← c90510, ← c90511, k2, ← c90512] at e_1476_15
  have hr := e_1476_15
  simp only [e_1476_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14360 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1476 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14286)) := by
  have c90513 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90514 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90515 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90516 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90517 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90518 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1476_16 := arithEq_of_rows h (row := 1476) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_16
  simp only [← c90513, ← c90514, ← c90515] at e_1476_16
  have e_1476_17 := arithEq_of_rows h (row := 1476) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_17
  simp only [← c90516, ← c90517, k2, ← c90518] at e_1476_17
  have hr := e_1476_17
  simp only [e_1476_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14361 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1476 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14287)) := by
  have c90519 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90520 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90521 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90522 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90523 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90524 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1476_18 := arithEq_of_rows h (row := 1476) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_18
  simp only [← c90519, ← c90520, ← c90521] at e_1476_18
  have e_1476_19 := arithEq_of_rows h (row := 1476) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1476_19
  simp only [← c90522, ← c90523, k2, ← c90524] at e_1476_19
  have hr := e_1476_19
  simp only [e_1476_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14362 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1477 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14288)) := by
  have c90525 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90526 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90527 := (publicBatchWrapper86_copies2828 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90528 := (publicBatchWrapper86_copies2829 a h).1
  have c90529 := (publicBatchWrapper86_copies2829 a h).2.1
  have c90530 := (publicBatchWrapper86_copies2829 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1477_0 := arithEq_of_rows h (row := 1477) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_0
  simp only [← c90525, ← c90526, ← c90527] at e_1477_0
  have e_1477_1 := arithEq_of_rows h (row := 1477) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_1
  simp only [← c90528, ← c90529, k2, ← c90530] at e_1477_1
  have hr := e_1477_1
  simp only [e_1477_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14363 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1477 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14289)) := by
  have c90531 := (publicBatchWrapper86_copies2829 a h).2.2.2.1
  have c90532 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.1
  have c90533 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.1
  have c90534 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.1
  have c90535 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.1
  have c90536 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1477_2 := arithEq_of_rows h (row := 1477) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_2
  simp only [← c90531, ← c90532, ← c90533] at e_1477_2
  have e_1477_3 := arithEq_of_rows h (row := 1477) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_3
  simp only [← c90534, ← c90535, k2, ← c90536] at e_1477_3
  have hr := e_1477_3
  simp only [e_1477_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14364 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1477 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14290)) := by
  have c90537 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.1
  have c90538 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90539 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90540 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90541 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90542 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1477_4 := arithEq_of_rows h (row := 1477) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_4
  simp only [← c90537, ← c90538, ← c90539] at e_1477_4
  have e_1477_5 := arithEq_of_rows h (row := 1477) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_5
  simp only [← c90540, ← c90541, k2, ← c90542] at e_1477_5
  have hr := e_1477_5
  simp only [e_1477_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14365 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1477 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14291)) := by
  have c90543 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90544 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90545 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90546 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90547 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90548 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1477_6 := arithEq_of_rows h (row := 1477) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_6
  simp only [← c90543, ← c90544, ← c90545] at e_1477_6
  have e_1477_7 := arithEq_of_rows h (row := 1477) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_7
  simp only [← c90546, ← c90547, k2, ← c90548] at e_1477_7
  have hr := e_1477_7
  simp only [e_1477_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14366 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1477 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14292)) := by
  have c90549 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90550 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90551 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90552 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90553 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90554 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1477_8 := arithEq_of_rows h (row := 1477) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_8
  simp only [← c90549, ← c90550, ← c90551] at e_1477_8
  have e_1477_9 := arithEq_of_rows h (row := 1477) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_9
  simp only [← c90552, ← c90553, k2, ← c90554] at e_1477_9
  have hr := e_1477_9
  simp only [e_1477_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14367 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1477 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14293)) := by
  have c90555 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90556 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90557 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90558 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90559 := (publicBatchWrapper86_copies2829 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90560 := (publicBatchWrapper86_copies2830 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1477_10 := arithEq_of_rows h (row := 1477) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_10
  simp only [← c90555, ← c90556, ← c90557] at e_1477_10
  have e_1477_11 := arithEq_of_rows h (row := 1477) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_11
  simp only [← c90558, ← c90559, k2, ← c90560] at e_1477_11
  have hr := e_1477_11
  simp only [e_1477_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14368 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1477 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14294)) := by
  have c90561 := (publicBatchWrapper86_copies2830 a h).2.1
  have c90562 := (publicBatchWrapper86_copies2830 a h).2.2.1
  have c90563 := (publicBatchWrapper86_copies2830 a h).2.2.2.1
  have c90564 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.1
  have c90565 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.1
  have c90566 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1477_12 := arithEq_of_rows h (row := 1477) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_12
  simp only [← c90561, ← c90562, ← c90563] at e_1477_12
  have e_1477_13 := arithEq_of_rows h (row := 1477) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_13
  simp only [← c90564, ← c90565, k2, ← c90566] at e_1477_13
  have hr := e_1477_13
  simp only [e_1477_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14369 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1477 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14295)) := by
  have c90567 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.1
  have c90568 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.1
  have c90569 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.1
  have c90570 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90571 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90572 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1477_14 := arithEq_of_rows h (row := 1477) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_14
  simp only [← c90567, ← c90568, ← c90569] at e_1477_14
  have e_1477_15 := arithEq_of_rows h (row := 1477) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_15
  simp only [← c90570, ← c90571, k2, ← c90572] at e_1477_15
  have hr := e_1477_15
  simp only [e_1477_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14370 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1477 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14296)) := by
  have c90573 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90574 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90575 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90576 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90577 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90578 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1477_16 := arithEq_of_rows h (row := 1477) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_16
  simp only [← c90573, ← c90574, ← c90575] at e_1477_16
  have e_1477_17 := arithEq_of_rows h (row := 1477) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_17
  simp only [← c90576, ← c90577, k2, ← c90578] at e_1477_17
  have hr := e_1477_17
  simp only [e_1477_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14371 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1477 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14297)) := by
  have c90579 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90580 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90581 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90582 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90583 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90584 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1477_18 := arithEq_of_rows h (row := 1477) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_18
  simp only [← c90579, ← c90580, ← c90581] at e_1477_18
  have e_1477_19 := arithEq_of_rows h (row := 1477) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1477_19
  simp only [← c90582, ← c90583, k2, ← c90584] at e_1477_19
  have hr := e_1477_19
  simp only [e_1477_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14372 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1478 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14298)) := by
  have c90585 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90586 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90587 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90588 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90589 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90590 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1478_0 := arithEq_of_rows h (row := 1478) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_0
  simp only [← c90585, ← c90586, ← c90587] at e_1478_0
  have e_1478_1 := arithEq_of_rows h (row := 1478) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_1
  simp only [← c90588, ← c90589, k2, ← c90590] at e_1478_1
  have hr := e_1478_1
  simp only [e_1478_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14373 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1478 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14299)) := by
  have c90591 := (publicBatchWrapper86_copies2830 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90592 := (publicBatchWrapper86_copies2831 a h).1
  have c90593 := (publicBatchWrapper86_copies2831 a h).2.1
  have c90594 := (publicBatchWrapper86_copies2831 a h).2.2.1
  have c90595 := (publicBatchWrapper86_copies2831 a h).2.2.2.1
  have c90596 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1478_2 := arithEq_of_rows h (row := 1478) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_2
  simp only [← c90591, ← c90592, ← c90593] at e_1478_2
  have e_1478_3 := arithEq_of_rows h (row := 1478) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_3
  simp only [← c90594, ← c90595, k2, ← c90596] at e_1478_3
  have hr := e_1478_3
  simp only [e_1478_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14374 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1478 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14300)) := by
  have c90597 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.1
  have c90598 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.1
  have c90599 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.1
  have c90600 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.1
  have c90601 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.1
  have c90602 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1478_4 := arithEq_of_rows h (row := 1478) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_4
  simp only [← c90597, ← c90598, ← c90599] at e_1478_4
  have e_1478_5 := arithEq_of_rows h (row := 1478) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_5
  simp only [← c90600, ← c90601, k2, ← c90602] at e_1478_5
  have hr := e_1478_5
  simp only [e_1478_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14375 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1478 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14301)) := by
  have c90603 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90604 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90605 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90606 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90607 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90608 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1478_6 := arithEq_of_rows h (row := 1478) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_6
  simp only [← c90603, ← c90604, ← c90605] at e_1478_6
  have e_1478_7 := arithEq_of_rows h (row := 1478) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_7
  simp only [← c90606, ← c90607, k2, ← c90608] at e_1478_7
  have hr := e_1478_7
  simp only [e_1478_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14376 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1478 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14302)) := by
  have c90609 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90610 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90611 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90612 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90613 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90614 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1478_8 := arithEq_of_rows h (row := 1478) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_8
  simp only [← c90609, ← c90610, ← c90611] at e_1478_8
  have e_1478_9 := arithEq_of_rows h (row := 1478) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_9
  simp only [← c90612, ← c90613, k2, ← c90614] at e_1478_9
  have hr := e_1478_9
  simp only [e_1478_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14377 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1478 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14303)) := by
  have c90615 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90616 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90617 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90618 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90619 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90620 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1478_10 := arithEq_of_rows h (row := 1478) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_10
  simp only [← c90615, ← c90616, ← c90617] at e_1478_10
  have e_1478_11 := arithEq_of_rows h (row := 1478) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_11
  simp only [← c90618, ← c90619, k2, ← c90620] at e_1478_11
  have hr := e_1478_11
  simp only [e_1478_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14378 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1478 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14304)) := by
  have c90621 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90622 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90623 := (publicBatchWrapper86_copies2831 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90624 := (publicBatchWrapper86_copies2832 a h).1
  have c90625 := (publicBatchWrapper86_copies2832 a h).2.1
  have c90626 := (publicBatchWrapper86_copies2832 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1478_12 := arithEq_of_rows h (row := 1478) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_12
  simp only [← c90621, ← c90622, ← c90623] at e_1478_12
  have e_1478_13 := arithEq_of_rows h (row := 1478) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_13
  simp only [← c90624, ← c90625, k2, ← c90626] at e_1478_13
  have hr := e_1478_13
  simp only [e_1478_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14379 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1478 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14305)) := by
  have c90627 := (publicBatchWrapper86_copies2832 a h).2.2.2.1
  have c90628 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.1
  have c90629 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.1
  have c90630 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.1
  have c90631 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.1
  have c90632 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1478_14 := arithEq_of_rows h (row := 1478) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_14
  simp only [← c90627, ← c90628, ← c90629] at e_1478_14
  have e_1478_15 := arithEq_of_rows h (row := 1478) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_15
  simp only [← c90630, ← c90631, k2, ← c90632] at e_1478_15
  have hr := e_1478_15
  simp only [e_1478_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14380 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1478 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14306)) := by
  have c90633 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.1
  have c90634 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90635 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90636 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90637 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90638 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1478_16 := arithEq_of_rows h (row := 1478) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_16
  simp only [← c90633, ← c90634, ← c90635] at e_1478_16
  have e_1478_17 := arithEq_of_rows h (row := 1478) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_17
  simp only [← c90636, ← c90637, k2, ← c90638] at e_1478_17
  have hr := e_1478_17
  simp only [e_1478_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14381 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1478 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14307)) := by
  have c90639 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90640 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90641 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90642 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90643 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90644 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1478_18 := arithEq_of_rows h (row := 1478) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_18
  simp only [← c90639, ← c90640, ← c90641] at e_1478_18
  have e_1478_19 := arithEq_of_rows h (row := 1478) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1478_19
  simp only [← c90642, ← c90643, k2, ← c90644] at e_1478_19
  have hr := e_1478_19
  simp only [e_1478_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14382 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1479 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14308)) := by
  have c90645 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90646 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90647 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90648 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90649 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90650 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1479_0 := arithEq_of_rows h (row := 1479) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_0
  simp only [← c90645, ← c90646, ← c90647] at e_1479_0
  have e_1479_1 := arithEq_of_rows h (row := 1479) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_1
  simp only [← c90648, ← c90649, k2, ← c90650] at e_1479_1
  have hr := e_1479_1
  simp only [e_1479_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14383 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1479 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14309)) := by
  have c90651 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90652 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90653 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90654 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90655 := (publicBatchWrapper86_copies2832 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90656 := (publicBatchWrapper86_copies2833 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1479_2 := arithEq_of_rows h (row := 1479) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_2
  simp only [← c90651, ← c90652, ← c90653] at e_1479_2
  have e_1479_3 := arithEq_of_rows h (row := 1479) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_3
  simp only [← c90654, ← c90655, k2, ← c90656] at e_1479_3
  have hr := e_1479_3
  simp only [e_1479_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14384 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1479 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14310)) := by
  have c90657 := (publicBatchWrapper86_copies2833 a h).2.1
  have c90658 := (publicBatchWrapper86_copies2833 a h).2.2.1
  have c90659 := (publicBatchWrapper86_copies2833 a h).2.2.2.1
  have c90660 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.1
  have c90661 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.1
  have c90662 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1479_4 := arithEq_of_rows h (row := 1479) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_4
  simp only [← c90657, ← c90658, ← c90659] at e_1479_4
  have e_1479_5 := arithEq_of_rows h (row := 1479) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_5
  simp only [← c90660, ← c90661, k2, ← c90662] at e_1479_5
  have hr := e_1479_5
  simp only [e_1479_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14385 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1479 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14311)) := by
  have c90663 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.1
  have c90664 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.1
  have c90665 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.1
  have c90666 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90667 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90668 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1479_6 := arithEq_of_rows h (row := 1479) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_6
  simp only [← c90663, ← c90664, ← c90665] at e_1479_6
  have e_1479_7 := arithEq_of_rows h (row := 1479) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_7
  simp only [← c90666, ← c90667, k2, ← c90668] at e_1479_7
  have hr := e_1479_7
  simp only [e_1479_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14386 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1479 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14312)) := by
  have c90669 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90670 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90671 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90672 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90673 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90674 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1479_8 := arithEq_of_rows h (row := 1479) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_8
  simp only [← c90669, ← c90670, ← c90671] at e_1479_8
  have e_1479_9 := arithEq_of_rows h (row := 1479) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_9
  simp only [← c90672, ← c90673, k2, ← c90674] at e_1479_9
  have hr := e_1479_9
  simp only [e_1479_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14387 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1479 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14313)) := by
  have c90675 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90676 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90677 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90678 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90679 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90680 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1479_10 := arithEq_of_rows h (row := 1479) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_10
  simp only [← c90675, ← c90676, ← c90677] at e_1479_10
  have e_1479_11 := arithEq_of_rows h (row := 1479) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_11
  simp only [← c90678, ← c90679, k2, ← c90680] at e_1479_11
  have hr := e_1479_11
  simp only [e_1479_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14388 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1479 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14314)) := by
  have c90681 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90682 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90683 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90684 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90685 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90686 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1479_12 := arithEq_of_rows h (row := 1479) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_12
  simp only [← c90681, ← c90682, ← c90683] at e_1479_12
  have e_1479_13 := arithEq_of_rows h (row := 1479) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_13
  simp only [← c90684, ← c90685, k2, ← c90686] at e_1479_13
  have hr := e_1479_13
  simp only [e_1479_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14389 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1479 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14315)) := by
  have c90687 := (publicBatchWrapper86_copies2833 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90688 := (publicBatchWrapper86_copies2834 a h).1
  have c90689 := (publicBatchWrapper86_copies2834 a h).2.1
  have c90690 := (publicBatchWrapper86_copies2834 a h).2.2.1
  have c90691 := (publicBatchWrapper86_copies2834 a h).2.2.2.1
  have c90692 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1479_14 := arithEq_of_rows h (row := 1479) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_14
  simp only [← c90687, ← c90688, ← c90689] at e_1479_14
  have e_1479_15 := arithEq_of_rows h (row := 1479) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_15
  simp only [← c90690, ← c90691, k2, ← c90692] at e_1479_15
  have hr := e_1479_15
  simp only [e_1479_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14390 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1479 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14316)) := by
  have c90693 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.1
  have c90694 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.1
  have c90695 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.1
  have c90696 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.1
  have c90697 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.1
  have c90698 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1479_16 := arithEq_of_rows h (row := 1479) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_16
  simp only [← c90693, ← c90694, ← c90695] at e_1479_16
  have e_1479_17 := arithEq_of_rows h (row := 1479) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_17
  simp only [← c90696, ← c90697, k2, ← c90698] at e_1479_17
  have hr := e_1479_17
  simp only [e_1479_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14391 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1479 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14317)) := by
  have c90699 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90700 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90701 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90702 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90703 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90704 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1479_18 := arithEq_of_rows h (row := 1479) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_18
  simp only [← c90699, ← c90700, ← c90701] at e_1479_18
  have e_1479_19 := arithEq_of_rows h (row := 1479) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1479_19
  simp only [← c90702, ← c90703, k2, ← c90704] at e_1479_19
  have hr := e_1479_19
  simp only [e_1479_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14392 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1480 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14318)) := by
  have c90705 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90706 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90707 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90708 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90709 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90710 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1480_0 := arithEq_of_rows h (row := 1480) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_0
  simp only [← c90705, ← c90706, ← c90707] at e_1480_0
  have e_1480_1 := arithEq_of_rows h (row := 1480) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_1
  simp only [← c90708, ← c90709, k2, ← c90710] at e_1480_1
  have hr := e_1480_1
  simp only [e_1480_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14393 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1480 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14319)) := by
  have c90711 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90712 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90713 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90714 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90715 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90716 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1480_2 := arithEq_of_rows h (row := 1480) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_2
  simp only [← c90711, ← c90712, ← c90713] at e_1480_2
  have e_1480_3 := arithEq_of_rows h (row := 1480) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_3
  simp only [← c90714, ← c90715, k2, ← c90716] at e_1480_3
  have hr := e_1480_3
  simp only [e_1480_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14394 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1480 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14320)) := by
  have c90717 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90718 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90719 := (publicBatchWrapper86_copies2834 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90720 := (publicBatchWrapper86_copies2835 a h).1
  have c90721 := (publicBatchWrapper86_copies2835 a h).2.1
  have c90722 := (publicBatchWrapper86_copies2835 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1480_4 := arithEq_of_rows h (row := 1480) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_4
  simp only [← c90717, ← c90718, ← c90719] at e_1480_4
  have e_1480_5 := arithEq_of_rows h (row := 1480) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_5
  simp only [← c90720, ← c90721, k2, ← c90722] at e_1480_5
  have hr := e_1480_5
  simp only [e_1480_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14395 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1480 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14321)) := by
  have c90723 := (publicBatchWrapper86_copies2835 a h).2.2.2.1
  have c90724 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.1
  have c90725 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.1
  have c90726 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.1
  have c90727 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.1
  have c90728 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1480_6 := arithEq_of_rows h (row := 1480) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_6
  simp only [← c90723, ← c90724, ← c90725] at e_1480_6
  have e_1480_7 := arithEq_of_rows h (row := 1480) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_7
  simp only [← c90726, ← c90727, k2, ← c90728] at e_1480_7
  have hr := e_1480_7
  simp only [e_1480_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14396 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1480 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14322)) := by
  have c90729 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.1
  have c90730 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90731 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90732 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90733 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90734 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1480_8 := arithEq_of_rows h (row := 1480) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_8
  simp only [← c90729, ← c90730, ← c90731] at e_1480_8
  have e_1480_9 := arithEq_of_rows h (row := 1480) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_9
  simp only [← c90732, ← c90733, k2, ← c90734] at e_1480_9
  have hr := e_1480_9
  simp only [e_1480_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14397 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1480 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14323)) := by
  have c90735 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90736 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90737 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90738 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90739 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90740 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1480_10 := arithEq_of_rows h (row := 1480) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_10
  simp only [← c90735, ← c90736, ← c90737] at e_1480_10
  have e_1480_11 := arithEq_of_rows h (row := 1480) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_11
  simp only [← c90738, ← c90739, k2, ← c90740] at e_1480_11
  have hr := e_1480_11
  simp only [e_1480_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14398 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1480 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14324)) := by
  have c90741 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90742 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90743 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90744 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90745 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90746 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1480_12 := arithEq_of_rows h (row := 1480) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_12
  simp only [← c90741, ← c90742, ← c90743] at e_1480_12
  have e_1480_13 := arithEq_of_rows h (row := 1480) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_13
  simp only [← c90744, ← c90745, k2, ← c90746] at e_1480_13
  have hr := e_1480_13
  simp only [e_1480_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14399 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1480 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14325)) := by
  have c90747 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90748 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90749 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90750 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90751 := (publicBatchWrapper86_copies2835 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90752 := (publicBatchWrapper86_copies2836 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1480_14 := arithEq_of_rows h (row := 1480) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_14
  simp only [← c90747, ← c90748, ← c90749] at e_1480_14
  have e_1480_15 := arithEq_of_rows h (row := 1480) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_15
  simp only [← c90750, ← c90751, k2, ← c90752] at e_1480_15
  have hr := e_1480_15
  simp only [e_1480_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14400 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1480 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14326)) := by
  have c90753 := (publicBatchWrapper86_copies2836 a h).2.1
  have c90754 := (publicBatchWrapper86_copies2836 a h).2.2.1
  have c90755 := (publicBatchWrapper86_copies2836 a h).2.2.2.1
  have c90756 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.1
  have c90757 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.1
  have c90758 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1480_16 := arithEq_of_rows h (row := 1480) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_16
  simp only [← c90753, ← c90754, ← c90755] at e_1480_16
  have e_1480_17 := arithEq_of_rows h (row := 1480) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_17
  simp only [← c90756, ← c90757, k2, ← c90758] at e_1480_17
  have hr := e_1480_17
  simp only [e_1480_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14401 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1480 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14327)) := by
  have c90759 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.1
  have c90760 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.1
  have c90761 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.1
  have c90762 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90763 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90764 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1480_18 := arithEq_of_rows h (row := 1480) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_18
  simp only [← c90759, ← c90760, ← c90761] at e_1480_18
  have e_1480_19 := arithEq_of_rows h (row := 1480) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1480_19
  simp only [← c90762, ← c90763, k2, ← c90764] at e_1480_19
  have hr := e_1480_19
  simp only [e_1480_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14402 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1481 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14328)) := by
  have c90765 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90766 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90767 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90768 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90769 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90770 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1481_0 := arithEq_of_rows h (row := 1481) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_0
  simp only [← c90765, ← c90766, ← c90767] at e_1481_0
  have e_1481_1 := arithEq_of_rows h (row := 1481) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_1
  simp only [← c90768, ← c90769, k2, ← c90770] at e_1481_1
  have hr := e_1481_1
  simp only [e_1481_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14403 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1481 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14329)) := by
  have c90771 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90772 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90773 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90774 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90775 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90776 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1481_2 := arithEq_of_rows h (row := 1481) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_2
  simp only [← c90771, ← c90772, ← c90773] at e_1481_2
  have e_1481_3 := arithEq_of_rows h (row := 1481) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_3
  simp only [← c90774, ← c90775, k2, ← c90776] at e_1481_3
  have hr := e_1481_3
  simp only [e_1481_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14404 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1481 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14330)) := by
  have c90777 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90778 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90779 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90780 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90781 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90782 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1481_4 := arithEq_of_rows h (row := 1481) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_4
  simp only [← c90777, ← c90778, ← c90779] at e_1481_4
  have e_1481_5 := arithEq_of_rows h (row := 1481) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_5
  simp only [← c90780, ← c90781, k2, ← c90782] at e_1481_5
  have hr := e_1481_5
  simp only [e_1481_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14405 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1481 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14331)) := by
  have c90783 := (publicBatchWrapper86_copies2836 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90784 := (publicBatchWrapper86_copies2837 a h).1
  have c90785 := (publicBatchWrapper86_copies2837 a h).2.1
  have c90786 := (publicBatchWrapper86_copies2837 a h).2.2.1
  have c90787 := (publicBatchWrapper86_copies2837 a h).2.2.2.1
  have c90788 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1481_6 := arithEq_of_rows h (row := 1481) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_6
  simp only [← c90783, ← c90784, ← c90785] at e_1481_6
  have e_1481_7 := arithEq_of_rows h (row := 1481) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_7
  simp only [← c90786, ← c90787, k2, ← c90788] at e_1481_7
  have hr := e_1481_7
  simp only [e_1481_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14406 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1481 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14332)) := by
  have c90789 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.1
  have c90790 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.1
  have c90791 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.1
  have c90792 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.1
  have c90793 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.1
  have c90794 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1481_8 := arithEq_of_rows h (row := 1481) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_8
  simp only [← c90789, ← c90790, ← c90791] at e_1481_8
  have e_1481_9 := arithEq_of_rows h (row := 1481) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_9
  simp only [← c90792, ← c90793, k2, ← c90794] at e_1481_9
  have hr := e_1481_9
  simp only [e_1481_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14407 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1481 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14333)) := by
  have c90795 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90796 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90797 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90798 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90799 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90800 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1481_10 := arithEq_of_rows h (row := 1481) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_10
  simp only [← c90795, ← c90796, ← c90797] at e_1481_10
  have e_1481_11 := arithEq_of_rows h (row := 1481) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_11
  simp only [← c90798, ← c90799, k2, ← c90800] at e_1481_11
  have hr := e_1481_11
  simp only [e_1481_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14408 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1481 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14334)) := by
  have c90801 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90802 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90803 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90804 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90805 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90806 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1481_12 := arithEq_of_rows h (row := 1481) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_12
  simp only [← c90801, ← c90802, ← c90803] at e_1481_12
  have e_1481_13 := arithEq_of_rows h (row := 1481) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_13
  simp only [← c90804, ← c90805, k2, ← c90806] at e_1481_13
  have hr := e_1481_13
  simp only [e_1481_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14409 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1481 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14335)) := by
  have c90807 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90808 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90809 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90810 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90811 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90812 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1481_14 := arithEq_of_rows h (row := 1481) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_14
  simp only [← c90807, ← c90808, ← c90809] at e_1481_14
  have e_1481_15 := arithEq_of_rows h (row := 1481) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_15
  simp only [← c90810, ← c90811, k2, ← c90812] at e_1481_15
  have hr := e_1481_15
  simp only [e_1481_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14410 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1481 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14336)) := by
  have c90813 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90814 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90815 := (publicBatchWrapper86_copies2837 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90816 := (publicBatchWrapper86_copies2838 a h).1
  have c90817 := (publicBatchWrapper86_copies2838 a h).2.1
  have c90818 := (publicBatchWrapper86_copies2838 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1481_16 := arithEq_of_rows h (row := 1481) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_16
  simp only [← c90813, ← c90814, ← c90815] at e_1481_16
  have e_1481_17 := arithEq_of_rows h (row := 1481) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_17
  simp only [← c90816, ← c90817, k2, ← c90818] at e_1481_17
  have hr := e_1481_17
  simp only [e_1481_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14411 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1481 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14337)) := by
  have c90819 := (publicBatchWrapper86_copies2838 a h).2.2.2.1
  have c90820 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.1
  have c90821 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.1
  have c90822 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.1
  have c90823 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.1
  have c90824 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1481_18 := arithEq_of_rows h (row := 1481) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_18
  simp only [← c90819, ← c90820, ← c90821] at e_1481_18
  have e_1481_19 := arithEq_of_rows h (row := 1481) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1481_19
  simp only [← c90822, ← c90823, k2, ← c90824] at e_1481_19
  have hr := e_1481_19
  simp only [e_1481_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14412 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1482 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14338)) := by
  have c90825 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.1
  have c90826 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90827 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90828 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90829 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90830 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1482_0 := arithEq_of_rows h (row := 1482) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_0
  simp only [← c90825, ← c90826, ← c90827] at e_1482_0
  have e_1482_1 := arithEq_of_rows h (row := 1482) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_1
  simp only [← c90828, ← c90829, k2, ← c90830] at e_1482_1
  have hr := e_1482_1
  simp only [e_1482_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14413 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1482 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14339)) := by
  have c90831 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90832 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90833 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90834 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90835 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90836 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1482_2 := arithEq_of_rows h (row := 1482) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_2
  simp only [← c90831, ← c90832, ← c90833] at e_1482_2
  have e_1482_3 := arithEq_of_rows h (row := 1482) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_3
  simp only [← c90834, ← c90835, k2, ← c90836] at e_1482_3
  have hr := e_1482_3
  simp only [e_1482_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14414 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1482 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14340)) := by
  have c90837 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90838 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90839 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90840 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90841 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90842 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1482_4 := arithEq_of_rows h (row := 1482) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_4
  simp only [← c90837, ← c90838, ← c90839] at e_1482_4
  have e_1482_5 := arithEq_of_rows h (row := 1482) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_5
  simp only [← c90840, ← c90841, k2, ← c90842] at e_1482_5
  have hr := e_1482_5
  simp only [e_1482_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14415 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1482 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14341)) := by
  have c90843 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90844 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90845 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90846 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90847 := (publicBatchWrapper86_copies2838 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90848 := (publicBatchWrapper86_copies2839 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1482_6 := arithEq_of_rows h (row := 1482) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_6
  simp only [← c90843, ← c90844, ← c90845] at e_1482_6
  have e_1482_7 := arithEq_of_rows h (row := 1482) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_7
  simp only [← c90846, ← c90847, k2, ← c90848] at e_1482_7
  have hr := e_1482_7
  simp only [e_1482_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14416 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1482 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14342)) := by
  have c90849 := (publicBatchWrapper86_copies2839 a h).2.1
  have c90850 := (publicBatchWrapper86_copies2839 a h).2.2.1
  have c90851 := (publicBatchWrapper86_copies2839 a h).2.2.2.1
  have c90852 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.1
  have c90853 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.1
  have c90854 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1482_8 := arithEq_of_rows h (row := 1482) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_8
  simp only [← c90849, ← c90850, ← c90851] at e_1482_8
  have e_1482_9 := arithEq_of_rows h (row := 1482) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_9
  simp only [← c90852, ← c90853, k2, ← c90854] at e_1482_9
  have hr := e_1482_9
  simp only [e_1482_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14417 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1482 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14343)) := by
  have c90855 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.1
  have c90856 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.1
  have c90857 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.1
  have c90858 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90859 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90860 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1482_10 := arithEq_of_rows h (row := 1482) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_10
  simp only [← c90855, ← c90856, ← c90857] at e_1482_10
  have e_1482_11 := arithEq_of_rows h (row := 1482) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_11
  simp only [← c90858, ← c90859, k2, ← c90860] at e_1482_11
  have hr := e_1482_11
  simp only [e_1482_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14418 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1482 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14344)) := by
  have c90861 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90862 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90863 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90864 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90865 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90866 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1482_12 := arithEq_of_rows h (row := 1482) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_12
  simp only [← c90861, ← c90862, ← c90863] at e_1482_12
  have e_1482_13 := arithEq_of_rows h (row := 1482) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_13
  simp only [← c90864, ← c90865, k2, ← c90866] at e_1482_13
  have hr := e_1482_13
  simp only [e_1482_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14419 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1482 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14345)) := by
  have c90867 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90868 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90869 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90870 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90871 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90872 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1482_14 := arithEq_of_rows h (row := 1482) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_14
  simp only [← c90867, ← c90868, ← c90869] at e_1482_14
  have e_1482_15 := arithEq_of_rows h (row := 1482) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_15
  simp only [← c90870, ← c90871, k2, ← c90872] at e_1482_15
  have hr := e_1482_15
  simp only [e_1482_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14420 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1482 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14346)) := by
  have c90873 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90874 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90875 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90876 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90877 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90878 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1482_16 := arithEq_of_rows h (row := 1482) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_16
  simp only [← c90873, ← c90874, ← c90875] at e_1482_16
  have e_1482_17 := arithEq_of_rows h (row := 1482) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_17
  simp only [← c90876, ← c90877, k2, ← c90878] at e_1482_17
  have hr := e_1482_17
  simp only [e_1482_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14421 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1482 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14347)) := by
  have c90879 := (publicBatchWrapper86_copies2839 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90880 := (publicBatchWrapper86_copies2840 a h).1
  have c90881 := (publicBatchWrapper86_copies2840 a h).2.1
  have c90882 := (publicBatchWrapper86_copies2840 a h).2.2.1
  have c90883 := (publicBatchWrapper86_copies2840 a h).2.2.2.1
  have c90884 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1482_18 := arithEq_of_rows h (row := 1482) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_18
  simp only [← c90879, ← c90880, ← c90881] at e_1482_18
  have e_1482_19 := arithEq_of_rows h (row := 1482) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1482_19
  simp only [← c90882, ← c90883, k2, ← c90884] at e_1482_19
  have hr := e_1482_19
  simp only [e_1482_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14422 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1483 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14348)) := by
  have c90885 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.1
  have c90886 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.1
  have c90887 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.1
  have c90888 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.1
  have c90889 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.1
  have c90890 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1483_0 := arithEq_of_rows h (row := 1483) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_0
  simp only [← c90885, ← c90886, ← c90887] at e_1483_0
  have e_1483_1 := arithEq_of_rows h (row := 1483) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_1
  simp only [← c90888, ← c90889, k2, ← c90890] at e_1483_1
  have hr := e_1483_1
  simp only [e_1483_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14423 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1483 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14349)) := by
  have c90891 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90892 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90893 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90894 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90895 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90896 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1483_2 := arithEq_of_rows h (row := 1483) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_2
  simp only [← c90891, ← c90892, ← c90893] at e_1483_2
  have e_1483_3 := arithEq_of_rows h (row := 1483) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_3
  simp only [← c90894, ← c90895, k2, ← c90896] at e_1483_3
  have hr := e_1483_3
  simp only [e_1483_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14424 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1483 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14350)) := by
  have c90897 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90898 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90899 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90900 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90901 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90902 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1483_4 := arithEq_of_rows h (row := 1483) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_4
  simp only [← c90897, ← c90898, ← c90899] at e_1483_4
  have e_1483_5 := arithEq_of_rows h (row := 1483) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_5
  simp only [← c90900, ← c90901, k2, ← c90902] at e_1483_5
  have hr := e_1483_5
  simp only [e_1483_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14425 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1483 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14351)) := by
  have c90903 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90904 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90905 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90906 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90907 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90908 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1483_6 := arithEq_of_rows h (row := 1483) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_6
  simp only [← c90903, ← c90904, ← c90905] at e_1483_6
  have e_1483_7 := arithEq_of_rows h (row := 1483) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_7
  simp only [← c90906, ← c90907, k2, ← c90908] at e_1483_7
  have hr := e_1483_7
  simp only [e_1483_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14426 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1483 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14352)) := by
  have c90909 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90910 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90911 := (publicBatchWrapper86_copies2840 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90912 := (publicBatchWrapper86_copies2841 a h).1
  have c90913 := (publicBatchWrapper86_copies2841 a h).2.1
  have c90914 := (publicBatchWrapper86_copies2841 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1483_8 := arithEq_of_rows h (row := 1483) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_8
  simp only [← c90909, ← c90910, ← c90911] at e_1483_8
  have e_1483_9 := arithEq_of_rows h (row := 1483) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_9
  simp only [← c90912, ← c90913, k2, ← c90914] at e_1483_9
  have hr := e_1483_9
  simp only [e_1483_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14427 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1483 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14353)) := by
  have c90915 := (publicBatchWrapper86_copies2841 a h).2.2.2.1
  have c90916 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.1
  have c90917 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.1
  have c90918 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.1
  have c90919 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.1
  have c90920 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1483_10 := arithEq_of_rows h (row := 1483) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_10
  simp only [← c90915, ← c90916, ← c90917] at e_1483_10
  have e_1483_11 := arithEq_of_rows h (row := 1483) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_11
  simp only [← c90918, ← c90919, k2, ← c90920] at e_1483_11
  have hr := e_1483_11
  simp only [e_1483_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14428 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1483 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14354)) := by
  have c90921 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.1
  have c90922 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90923 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90924 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90925 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90926 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1483_12 := arithEq_of_rows h (row := 1483) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_12
  simp only [← c90921, ← c90922, ← c90923] at e_1483_12
  have e_1483_13 := arithEq_of_rows h (row := 1483) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_13
  simp only [← c90924, ← c90925, k2, ← c90926] at e_1483_13
  have hr := e_1483_13
  simp only [e_1483_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14429 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1483 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14355)) := by
  have c90927 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90928 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90929 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90930 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90931 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90932 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1483_14 := arithEq_of_rows h (row := 1483) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_14
  simp only [← c90927, ← c90928, ← c90929] at e_1483_14
  have e_1483_15 := arithEq_of_rows h (row := 1483) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_15
  simp only [← c90930, ← c90931, k2, ← c90932] at e_1483_15
  have hr := e_1483_15
  simp only [e_1483_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14430 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1483 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14356)) := by
  have c90933 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90934 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90935 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90936 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90937 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90938 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1483_16 := arithEq_of_rows h (row := 1483) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_16
  simp only [← c90933, ← c90934, ← c90935] at e_1483_16
  have e_1483_17 := arithEq_of_rows h (row := 1483) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_17
  simp only [← c90936, ← c90937, k2, ← c90938] at e_1483_17
  have hr := e_1483_17
  simp only [e_1483_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14431 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1483 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14357)) := by
  have c90939 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90940 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90941 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90942 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90943 := (publicBatchWrapper86_copies2841 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90944 := (publicBatchWrapper86_copies2842 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1483_18 := arithEq_of_rows h (row := 1483) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_18
  simp only [← c90939, ← c90940, ← c90941] at e_1483_18
  have e_1483_19 := arithEq_of_rows h (row := 1483) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1483_19
  simp only [← c90942, ← c90943, k2, ← c90944] at e_1483_19
  have hr := e_1483_19
  simp only [e_1483_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14432 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1484 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14358)) := by
  have c90945 := (publicBatchWrapper86_copies2842 a h).2.1
  have c90946 := (publicBatchWrapper86_copies2842 a h).2.2.1
  have c90947 := (publicBatchWrapper86_copies2842 a h).2.2.2.1
  have c90948 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.1
  have c90949 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.1
  have c90950 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1484_0 := arithEq_of_rows h (row := 1484) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_0
  simp only [← c90945, ← c90946, ← c90947] at e_1484_0
  have e_1484_1 := arithEq_of_rows h (row := 1484) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_1
  simp only [← c90948, ← c90949, k2, ← c90950] at e_1484_1
  have hr := e_1484_1
  simp only [e_1484_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14433 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1484 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14359)) := by
  have c90951 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.1
  have c90952 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.1
  have c90953 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.1
  have c90954 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.1
  have c90955 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90956 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1484_2 := arithEq_of_rows h (row := 1484) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_2
  simp only [← c90951, ← c90952, ← c90953] at e_1484_2
  have e_1484_3 := arithEq_of_rows h (row := 1484) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_3
  simp only [← c90954, ← c90955, k2, ← c90956] at e_1484_3
  have hr := e_1484_3
  simp only [e_1484_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14434 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1484 23) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14360)) := by
  have c90957 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90958 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90959 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90960 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90961 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90962 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1484_4 := arithEq_of_rows h (row := 1484) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_4
  simp only [← c90957, ← c90958, ← c90959] at e_1484_4
  have e_1484_5 := arithEq_of_rows h (row := 1484) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_5
  simp only [← c90960, ← c90961, k2, ← c90962] at e_1484_5
  have hr := e_1484_5
  simp only [e_1484_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14435 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1484 31) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14361)) := by
  have c90963 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90964 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90965 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90966 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90967 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90968 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1484_6 := arithEq_of_rows h (row := 1484) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_6
  simp only [← c90963, ← c90964, ← c90965] at e_1484_6
  have e_1484_7 := arithEq_of_rows h (row := 1484) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_7
  simp only [← c90966, ← c90967, k2, ← c90968] at e_1484_7
  have hr := e_1484_7
  simp only [e_1484_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14436 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1484 39) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14362)) := by
  have c90969 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90970 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90971 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90972 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90973 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90974 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1484_8 := arithEq_of_rows h (row := 1484) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_8
  simp only [← c90969, ← c90970, ← c90971] at e_1484_8
  have e_1484_9 := arithEq_of_rows h (row := 1484) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_9
  simp only [← c90972, ← c90973, k2, ← c90974] at e_1484_9
  have hr := e_1484_9
  simp only [e_1484_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14437 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1484 47) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14363)) := by
  have c90975 := (publicBatchWrapper86_copies2842 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c90976 := (publicBatchWrapper86_copies2843 a h).1
  have c90977 := (publicBatchWrapper86_copies2843 a h).2.1
  have c90978 := (publicBatchWrapper86_copies2843 a h).2.2.1
  have c90979 := (publicBatchWrapper86_copies2843 a h).2.2.2.1
  have c90980 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1484_10 := arithEq_of_rows h (row := 1484) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_10
  simp only [← c90975, ← c90976, ← c90977] at e_1484_10
  have e_1484_11 := arithEq_of_rows h (row := 1484) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_11
  simp only [← c90978, ← c90979, k2, ← c90980] at e_1484_11
  have hr := e_1484_11
  simp only [e_1484_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14438 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1484 55) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14364)) := by
  have c90981 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.1
  have c90982 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.1
  have c90983 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.1
  have c90984 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.1
  have c90985 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.1
  have c90986 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1484_12 := arithEq_of_rows h (row := 1484) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_12
  simp only [← c90981, ← c90982, ← c90983] at e_1484_12
  have e_1484_13 := arithEq_of_rows h (row := 1484) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_13
  simp only [← c90984, ← c90985, k2, ← c90986] at e_1484_13
  have hr := e_1484_13
  simp only [e_1484_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14439 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1484 63) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14365)) := by
  have c90987 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c90988 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90989 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90990 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90991 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90992 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1484_14 := arithEq_of_rows h (row := 1484) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_14
  simp only [← c90987, ← c90988, ← c90989] at e_1484_14
  have e_1484_15 := arithEq_of_rows h (row := 1484) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_15
  simp only [← c90990, ← c90991, k2, ← c90992] at e_1484_15
  have hr := e_1484_15
  simp only [e_1484_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14440 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1484 71) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14366)) := by
  have c90993 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90994 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90995 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90996 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90997 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c90998 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1484_16 := arithEq_of_rows h (row := 1484) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_16
  simp only [← c90993, ← c90994, ← c90995] at e_1484_16
  have e_1484_17 := arithEq_of_rows h (row := 1484) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_17
  simp only [← c90996, ← c90997, k2, ← c90998] at e_1484_17
  have hr := e_1484_17
  simp only [e_1484_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14441 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1484 79) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14367)) := by
  have c90999 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91000 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91001 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91002 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91003 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91004 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1484_18 := arithEq_of_rows h (row := 1484) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_18
  simp only [← c90999, ← c91000, ← c91001] at e_1484_18
  have e_1484_19 := arithEq_of_rows h (row := 1484) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1484_19
  simp only [← c91002, ← c91003, k2, ← c91004] at e_1484_19
  have hr := e_1484_19
  simp only [e_1484_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14442 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1485 7) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14368)) := by
  have c91005 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91006 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91007 := (publicBatchWrapper86_copies2843 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91008 := (publicBatchWrapper86_copies2844 a h).1
  have c91009 := (publicBatchWrapper86_copies2844 a h).2.1
  have c91010 := (publicBatchWrapper86_copies2844 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1485_0 := arithEq_of_rows h (row := 1485) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_0
  simp only [← c91005, ← c91006, ← c91007] at e_1485_0
  have e_1485_1 := arithEq_of_rows h (row := 1485) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_1
  simp only [← c91008, ← c91009, k2, ← c91010] at e_1485_1
  have hr := e_1485_1
  simp only [e_1485_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14443 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1485 15) = bselect (a (.wire 70 31)) (a (.virt 200)) (a (.virt 14369)) := by
  have c91011 := (publicBatchWrapper86_copies2844 a h).2.2.2.1
  have c91012 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.1
  have c91013 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.1
  have c91014 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.1
  have c91015 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.1
  have c91016 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1485_2 := arithEq_of_rows h (row := 1485) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_2
  simp only [← c91011, ← c91012, ← c91013] at e_1485_2
  have e_1485_3 := arithEq_of_rows h (row := 1485) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_3
  simp only [← c91014, ← c91015, k2, ← c91016] at e_1485_3
  have hr := e_1485_3
  simp only [e_1485_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14444 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1485 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14410)) := by
  have c91017 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.1
  have c91018 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91019 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91020 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91021 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91022 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1485_4 := arithEq_of_rows h (row := 1485) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_4
  simp only [← c91017, ← c91018, ← c91019] at e_1485_4
  have e_1485_5 := arithEq_of_rows h (row := 1485) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_5
  simp only [← c91020, ← c91021, k2, ← c91022] at e_1485_5
  have hr := e_1485_5
  simp only [e_1485_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14445 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1485 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14411)) := by
  have c91023 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91024 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91025 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91026 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91027 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91028 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1485_6 := arithEq_of_rows h (row := 1485) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_6
  simp only [← c91023, ← c91024, ← c91025] at e_1485_6
  have e_1485_7 := arithEq_of_rows h (row := 1485) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_7
  simp only [← c91026, ← c91027, k2, ← c91028] at e_1485_7
  have hr := e_1485_7
  simp only [e_1485_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14446 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1485 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14412)) := by
  have c91029 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91030 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91031 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91032 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91033 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91034 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1485_8 := arithEq_of_rows h (row := 1485) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_8
  simp only [← c91029, ← c91030, ← c91031] at e_1485_8
  have e_1485_9 := arithEq_of_rows h (row := 1485) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_9
  simp only [← c91032, ← c91033, k2, ← c91034] at e_1485_9
  have hr := e_1485_9
  simp only [e_1485_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14447 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1485 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14413)) := by
  have c91035 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91036 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91037 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91038 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91039 := (publicBatchWrapper86_copies2844 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91040 := (publicBatchWrapper86_copies2845 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1485_10 := arithEq_of_rows h (row := 1485) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_10
  simp only [← c91035, ← c91036, ← c91037] at e_1485_10
  have e_1485_11 := arithEq_of_rows h (row := 1485) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_11
  simp only [← c91038, ← c91039, k2, ← c91040] at e_1485_11
  have hr := e_1485_11
  simp only [e_1485_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14448 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1485 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14414)) := by
  have c91041 := (publicBatchWrapper86_copies2845 a h).2.1
  have c91042 := (publicBatchWrapper86_copies2845 a h).2.2.1
  have c91043 := (publicBatchWrapper86_copies2845 a h).2.2.2.1
  have c91044 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.1
  have c91045 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.1
  have c91046 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1485_12 := arithEq_of_rows h (row := 1485) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_12
  simp only [← c91041, ← c91042, ← c91043] at e_1485_12
  have e_1485_13 := arithEq_of_rows h (row := 1485) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_13
  simp only [← c91044, ← c91045, k2, ← c91046] at e_1485_13
  have hr := e_1485_13
  simp only [e_1485_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14449 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1485 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14415)) := by
  have c91047 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.1
  have c91048 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.1
  have c91049 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.1
  have c91050 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91051 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91052 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1485_14 := arithEq_of_rows h (row := 1485) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_14
  simp only [← c91047, ← c91048, ← c91049] at e_1485_14
  have e_1485_15 := arithEq_of_rows h (row := 1485) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_15
  simp only [← c91050, ← c91051, k2, ← c91052] at e_1485_15
  have hr := e_1485_15
  simp only [e_1485_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14450 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1485 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14416)) := by
  have c91053 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91054 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91055 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91056 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91057 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91058 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1485_16 := arithEq_of_rows h (row := 1485) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_16
  simp only [← c91053, ← c91054, ← c91055] at e_1485_16
  have e_1485_17 := arithEq_of_rows h (row := 1485) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_17
  simp only [← c91056, ← c91057, k2, ← c91058] at e_1485_17
  have hr := e_1485_17
  simp only [e_1485_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14451 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1485 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14417)) := by
  have c91059 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91060 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91061 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91062 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91063 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91064 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1485_18 := arithEq_of_rows h (row := 1485) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_18
  simp only [← c91059, ← c91060, ← c91061] at e_1485_18
  have e_1485_19 := arithEq_of_rows h (row := 1485) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1485_19
  simp only [← c91062, ← c91063, k2, ← c91064] at e_1485_19
  have hr := e_1485_19
  simp only [e_1485_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14452 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1486 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14418)) := by
  have c91065 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91066 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91067 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91068 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91069 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91070 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1486_0 := arithEq_of_rows h (row := 1486) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_0
  simp only [← c91065, ← c91066, ← c91067] at e_1486_0
  have e_1486_1 := arithEq_of_rows h (row := 1486) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_1
  simp only [← c91068, ← c91069, k2, ← c91070] at e_1486_1
  have hr := e_1486_1
  simp only [e_1486_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14453 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1486 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14419)) := by
  have c91071 := (publicBatchWrapper86_copies2845 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91072 := (publicBatchWrapper86_copies2846 a h).1
  have c91073 := (publicBatchWrapper86_copies2846 a h).2.1
  have c91074 := (publicBatchWrapper86_copies2846 a h).2.2.1
  have c91075 := (publicBatchWrapper86_copies2846 a h).2.2.2.1
  have c91076 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1486_2 := arithEq_of_rows h (row := 1486) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_2
  simp only [← c91071, ← c91072, ← c91073] at e_1486_2
  have e_1486_3 := arithEq_of_rows h (row := 1486) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_3
  simp only [← c91074, ← c91075, k2, ← c91076] at e_1486_3
  have hr := e_1486_3
  simp only [e_1486_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14454 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1486 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14420)) := by
  have c91077 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.1
  have c91078 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.1
  have c91079 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.1
  have c91080 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.1
  have c91081 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.1
  have c91082 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1486_4 := arithEq_of_rows h (row := 1486) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_4
  simp only [← c91077, ← c91078, ← c91079] at e_1486_4
  have e_1486_5 := arithEq_of_rows h (row := 1486) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_5
  simp only [← c91080, ← c91081, k2, ← c91082] at e_1486_5
  have hr := e_1486_5
  simp only [e_1486_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14455 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1486 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14421)) := by
  have c91083 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91084 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91085 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91086 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91087 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91088 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1486_6 := arithEq_of_rows h (row := 1486) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_6
  simp only [← c91083, ← c91084, ← c91085] at e_1486_6
  have e_1486_7 := arithEq_of_rows h (row := 1486) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_7
  simp only [← c91086, ← c91087, k2, ← c91088] at e_1486_7
  have hr := e_1486_7
  simp only [e_1486_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14456 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1486 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14422)) := by
  have c91089 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91090 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91091 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91092 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91093 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91094 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1486_8 := arithEq_of_rows h (row := 1486) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_8
  simp only [← c91089, ← c91090, ← c91091] at e_1486_8
  have e_1486_9 := arithEq_of_rows h (row := 1486) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_9
  simp only [← c91092, ← c91093, k2, ← c91094] at e_1486_9
  have hr := e_1486_9
  simp only [e_1486_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14457 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1486 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14423)) := by
  have c91095 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91096 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91097 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91098 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91099 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91100 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1486_10 := arithEq_of_rows h (row := 1486) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_10
  simp only [← c91095, ← c91096, ← c91097] at e_1486_10
  have e_1486_11 := arithEq_of_rows h (row := 1486) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_11
  simp only [← c91098, ← c91099, k2, ← c91100] at e_1486_11
  have hr := e_1486_11
  simp only [e_1486_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14458 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1486 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14424)) := by
  have c91101 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91102 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91103 := (publicBatchWrapper86_copies2846 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91104 := (publicBatchWrapper86_copies2847 a h).1
  have c91105 := (publicBatchWrapper86_copies2847 a h).2.1
  have c91106 := (publicBatchWrapper86_copies2847 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1486_12 := arithEq_of_rows h (row := 1486) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_12
  simp only [← c91101, ← c91102, ← c91103] at e_1486_12
  have e_1486_13 := arithEq_of_rows h (row := 1486) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_13
  simp only [← c91104, ← c91105, k2, ← c91106] at e_1486_13
  have hr := e_1486_13
  simp only [e_1486_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14459 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1486 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14425)) := by
  have c91107 := (publicBatchWrapper86_copies2847 a h).2.2.2.1
  have c91108 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.1
  have c91109 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.1
  have c91110 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.1
  have c91111 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.1
  have c91112 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1486_14 := arithEq_of_rows h (row := 1486) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_14
  simp only [← c91107, ← c91108, ← c91109] at e_1486_14
  have e_1486_15 := arithEq_of_rows h (row := 1486) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_15
  simp only [← c91110, ← c91111, k2, ← c91112] at e_1486_15
  have hr := e_1486_15
  simp only [e_1486_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14460 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1486 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14426)) := by
  have c91113 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.1
  have c91114 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91115 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91116 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91117 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91118 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1486_16 := arithEq_of_rows h (row := 1486) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_16
  simp only [← c91113, ← c91114, ← c91115] at e_1486_16
  have e_1486_17 := arithEq_of_rows h (row := 1486) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_17
  simp only [← c91116, ← c91117, k2, ← c91118] at e_1486_17
  have hr := e_1486_17
  simp only [e_1486_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14461 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1486 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14427)) := by
  have c91119 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91120 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91121 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91122 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91123 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91124 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1486_18 := arithEq_of_rows h (row := 1486) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_18
  simp only [← c91119, ← c91120, ← c91121] at e_1486_18
  have e_1486_19 := arithEq_of_rows h (row := 1486) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1486_19
  simp only [← c91122, ← c91123, k2, ← c91124] at e_1486_19
  have hr := e_1486_19
  simp only [e_1486_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14462 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1487 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14428)) := by
  have c91125 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91126 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91127 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91128 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91129 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91130 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1487_0 := arithEq_of_rows h (row := 1487) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_0
  simp only [← c91125, ← c91126, ← c91127] at e_1487_0
  have e_1487_1 := arithEq_of_rows h (row := 1487) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_1
  simp only [← c91128, ← c91129, k2, ← c91130] at e_1487_1
  have hr := e_1487_1
  simp only [e_1487_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14463 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1487 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14429)) := by
  have c91131 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91132 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91133 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91134 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91135 := (publicBatchWrapper86_copies2847 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91136 := (publicBatchWrapper86_copies2848 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1487_2 := arithEq_of_rows h (row := 1487) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_2
  simp only [← c91131, ← c91132, ← c91133] at e_1487_2
  have e_1487_3 := arithEq_of_rows h (row := 1487) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_3
  simp only [← c91134, ← c91135, k2, ← c91136] at e_1487_3
  have hr := e_1487_3
  simp only [e_1487_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14464 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1487 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14430)) := by
  have c91137 := (publicBatchWrapper86_copies2848 a h).2.1
  have c91138 := (publicBatchWrapper86_copies2848 a h).2.2.1
  have c91139 := (publicBatchWrapper86_copies2848 a h).2.2.2.1
  have c91140 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.1
  have c91141 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.1
  have c91142 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1487_4 := arithEq_of_rows h (row := 1487) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_4
  simp only [← c91137, ← c91138, ← c91139] at e_1487_4
  have e_1487_5 := arithEq_of_rows h (row := 1487) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_5
  simp only [← c91140, ← c91141, k2, ← c91142] at e_1487_5
  have hr := e_1487_5
  simp only [e_1487_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14465 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1487 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14431)) := by
  have c91143 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.1
  have c91144 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.1
  have c91145 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.1
  have c91146 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91147 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91148 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1487_6 := arithEq_of_rows h (row := 1487) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_6
  simp only [← c91143, ← c91144, ← c91145] at e_1487_6
  have e_1487_7 := arithEq_of_rows h (row := 1487) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_7
  simp only [← c91146, ← c91147, k2, ← c91148] at e_1487_7
  have hr := e_1487_7
  simp only [e_1487_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14466 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1487 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14432)) := by
  have c91149 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91150 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91151 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91152 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91153 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91154 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1487_8 := arithEq_of_rows h (row := 1487) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_8
  simp only [← c91149, ← c91150, ← c91151] at e_1487_8
  have e_1487_9 := arithEq_of_rows h (row := 1487) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_9
  simp only [← c91152, ← c91153, k2, ← c91154] at e_1487_9
  have hr := e_1487_9
  simp only [e_1487_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14467 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1487 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14433)) := by
  have c91155 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91156 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91157 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91158 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91159 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91160 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1487_10 := arithEq_of_rows h (row := 1487) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_10
  simp only [← c91155, ← c91156, ← c91157] at e_1487_10
  have e_1487_11 := arithEq_of_rows h (row := 1487) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_11
  simp only [← c91158, ← c91159, k2, ← c91160] at e_1487_11
  have hr := e_1487_11
  simp only [e_1487_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14468 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1487 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14434)) := by
  have c91161 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91162 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91163 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91164 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91165 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91166 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1487_12 := arithEq_of_rows h (row := 1487) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_12
  simp only [← c91161, ← c91162, ← c91163] at e_1487_12
  have e_1487_13 := arithEq_of_rows h (row := 1487) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_13
  simp only [← c91164, ← c91165, k2, ← c91166] at e_1487_13
  have hr := e_1487_13
  simp only [e_1487_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14469 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1487 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14435)) := by
  have c91167 := (publicBatchWrapper86_copies2848 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91168 := (publicBatchWrapper86_copies2849 a h).1
  have c91169 := (publicBatchWrapper86_copies2849 a h).2.1
  have c91170 := (publicBatchWrapper86_copies2849 a h).2.2.1
  have c91171 := (publicBatchWrapper86_copies2849 a h).2.2.2.1
  have c91172 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1487_14 := arithEq_of_rows h (row := 1487) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_14
  simp only [← c91167, ← c91168, ← c91169] at e_1487_14
  have e_1487_15 := arithEq_of_rows h (row := 1487) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_15
  simp only [← c91170, ← c91171, k2, ← c91172] at e_1487_15
  have hr := e_1487_15
  simp only [e_1487_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14470 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1487 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14436)) := by
  have c91173 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.1
  have c91174 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.1
  have c91175 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.1
  have c91176 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.1
  have c91177 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.1
  have c91178 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1487_16 := arithEq_of_rows h (row := 1487) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_16
  simp only [← c91173, ← c91174, ← c91175] at e_1487_16
  have e_1487_17 := arithEq_of_rows h (row := 1487) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_17
  simp only [← c91176, ← c91177, k2, ← c91178] at e_1487_17
  have hr := e_1487_17
  simp only [e_1487_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14471 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1487 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14437)) := by
  have c91179 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91180 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91181 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91182 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91183 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91184 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1487_18 := arithEq_of_rows h (row := 1487) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_18
  simp only [← c91179, ← c91180, ← c91181] at e_1487_18
  have e_1487_19 := arithEq_of_rows h (row := 1487) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1487_19
  simp only [← c91182, ← c91183, k2, ← c91184] at e_1487_19
  have hr := e_1487_19
  simp only [e_1487_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14472 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1488 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14438)) := by
  have c91185 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91186 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91187 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91188 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91189 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91190 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1488_0 := arithEq_of_rows h (row := 1488) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_0
  simp only [← c91185, ← c91186, ← c91187] at e_1488_0
  have e_1488_1 := arithEq_of_rows h (row := 1488) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_1
  simp only [← c91188, ← c91189, k2, ← c91190] at e_1488_1
  have hr := e_1488_1
  simp only [e_1488_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14473 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1488 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14439)) := by
  have c91191 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91192 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91193 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91194 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91195 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91196 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1488_2 := arithEq_of_rows h (row := 1488) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_2
  simp only [← c91191, ← c91192, ← c91193] at e_1488_2
  have e_1488_3 := arithEq_of_rows h (row := 1488) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_3
  simp only [← c91194, ← c91195, k2, ← c91196] at e_1488_3
  have hr := e_1488_3
  simp only [e_1488_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14474 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1488 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14440)) := by
  have c91197 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91198 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91199 := (publicBatchWrapper86_copies2849 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91200 := (publicBatchWrapper86_copies2850 a h).1
  have c91201 := (publicBatchWrapper86_copies2850 a h).2.1
  have c91202 := (publicBatchWrapper86_copies2850 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1488_4 := arithEq_of_rows h (row := 1488) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_4
  simp only [← c91197, ← c91198, ← c91199] at e_1488_4
  have e_1488_5 := arithEq_of_rows h (row := 1488) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_5
  simp only [← c91200, ← c91201, k2, ← c91202] at e_1488_5
  have hr := e_1488_5
  simp only [e_1488_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14475 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1488 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14441)) := by
  have c91203 := (publicBatchWrapper86_copies2850 a h).2.2.2.1
  have c91204 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.1
  have c91205 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.1
  have c91206 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.1
  have c91207 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.1
  have c91208 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1488_6 := arithEq_of_rows h (row := 1488) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_6
  simp only [← c91203, ← c91204, ← c91205] at e_1488_6
  have e_1488_7 := arithEq_of_rows h (row := 1488) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_7
  simp only [← c91206, ← c91207, k2, ← c91208] at e_1488_7
  have hr := e_1488_7
  simp only [e_1488_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14476 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1488 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14442)) := by
  have c91209 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.1
  have c91210 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91211 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91212 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91213 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91214 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1488_8 := arithEq_of_rows h (row := 1488) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_8
  simp only [← c91209, ← c91210, ← c91211] at e_1488_8
  have e_1488_9 := arithEq_of_rows h (row := 1488) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_9
  simp only [← c91212, ← c91213, k2, ← c91214] at e_1488_9
  have hr := e_1488_9
  simp only [e_1488_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14477 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1488 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14443)) := by
  have c91215 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91216 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91217 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91218 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91219 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91220 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1488_10 := arithEq_of_rows h (row := 1488) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_10
  simp only [← c91215, ← c91216, ← c91217] at e_1488_10
  have e_1488_11 := arithEq_of_rows h (row := 1488) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_11
  simp only [← c91218, ← c91219, k2, ← c91220] at e_1488_11
  have hr := e_1488_11
  simp only [e_1488_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14478 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1488 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14444)) := by
  have c91221 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91222 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91223 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91224 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91225 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91226 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1488_12 := arithEq_of_rows h (row := 1488) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_12
  simp only [← c91221, ← c91222, ← c91223] at e_1488_12
  have e_1488_13 := arithEq_of_rows h (row := 1488) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_13
  simp only [← c91224, ← c91225, k2, ← c91226] at e_1488_13
  have hr := e_1488_13
  simp only [e_1488_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14479 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1488 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14445)) := by
  have c91227 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91228 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91229 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91230 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91231 := (publicBatchWrapper86_copies2850 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91232 := (publicBatchWrapper86_copies2851 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1488_14 := arithEq_of_rows h (row := 1488) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_14
  simp only [← c91227, ← c91228, ← c91229] at e_1488_14
  have e_1488_15 := arithEq_of_rows h (row := 1488) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_15
  simp only [← c91230, ← c91231, k2, ← c91232] at e_1488_15
  have hr := e_1488_15
  simp only [e_1488_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14480 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1488 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14446)) := by
  have c91233 := (publicBatchWrapper86_copies2851 a h).2.1
  have c91234 := (publicBatchWrapper86_copies2851 a h).2.2.1
  have c91235 := (publicBatchWrapper86_copies2851 a h).2.2.2.1
  have c91236 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.1
  have c91237 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.1
  have c91238 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1488_16 := arithEq_of_rows h (row := 1488) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_16
  simp only [← c91233, ← c91234, ← c91235] at e_1488_16
  have e_1488_17 := arithEq_of_rows h (row := 1488) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_17
  simp only [← c91236, ← c91237, k2, ← c91238] at e_1488_17
  have hr := e_1488_17
  simp only [e_1488_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14481 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1488 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14447)) := by
  have c91239 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.1
  have c91240 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.1
  have c91241 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.1
  have c91242 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91243 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91244 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1488_18 := arithEq_of_rows h (row := 1488) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_18
  simp only [← c91239, ← c91240, ← c91241] at e_1488_18
  have e_1488_19 := arithEq_of_rows h (row := 1488) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1488_19
  simp only [← c91242, ← c91243, k2, ← c91244] at e_1488_19
  have hr := e_1488_19
  simp only [e_1488_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14482 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1489 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14448)) := by
  have c91245 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91246 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91247 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91248 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91249 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91250 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1489_0 := arithEq_of_rows h (row := 1489) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_0
  simp only [← c91245, ← c91246, ← c91247] at e_1489_0
  have e_1489_1 := arithEq_of_rows h (row := 1489) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_1
  simp only [← c91248, ← c91249, k2, ← c91250] at e_1489_1
  have hr := e_1489_1
  simp only [e_1489_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14483 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1489 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14449)) := by
  have c91251 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91252 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91253 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91254 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91255 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91256 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1489_2 := arithEq_of_rows h (row := 1489) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_2
  simp only [← c91251, ← c91252, ← c91253] at e_1489_2
  have e_1489_3 := arithEq_of_rows h (row := 1489) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_3
  simp only [← c91254, ← c91255, k2, ← c91256] at e_1489_3
  have hr := e_1489_3
  simp only [e_1489_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14484 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1489 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14450)) := by
  have c91257 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91258 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91259 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91260 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91261 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91262 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1489_4 := arithEq_of_rows h (row := 1489) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_4
  simp only [← c91257, ← c91258, ← c91259] at e_1489_4
  have e_1489_5 := arithEq_of_rows h (row := 1489) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_5
  simp only [← c91260, ← c91261, k2, ← c91262] at e_1489_5
  have hr := e_1489_5
  simp only [e_1489_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14485 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1489 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14451)) := by
  have c91263 := (publicBatchWrapper86_copies2851 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91264 := (publicBatchWrapper86_copies2852 a h).1
  have c91265 := (publicBatchWrapper86_copies2852 a h).2.1
  have c91266 := (publicBatchWrapper86_copies2852 a h).2.2.1
  have c91267 := (publicBatchWrapper86_copies2852 a h).2.2.2.1
  have c91268 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1489_6 := arithEq_of_rows h (row := 1489) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_6
  simp only [← c91263, ← c91264, ← c91265] at e_1489_6
  have e_1489_7 := arithEq_of_rows h (row := 1489) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_7
  simp only [← c91266, ← c91267, k2, ← c91268] at e_1489_7
  have hr := e_1489_7
  simp only [e_1489_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14486 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1489 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14452)) := by
  have c91269 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.1
  have c91270 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.1
  have c91271 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.1
  have c91272 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.1
  have c91273 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.1
  have c91274 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1489_8 := arithEq_of_rows h (row := 1489) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_8
  simp only [← c91269, ← c91270, ← c91271] at e_1489_8
  have e_1489_9 := arithEq_of_rows h (row := 1489) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_9
  simp only [← c91272, ← c91273, k2, ← c91274] at e_1489_9
  have hr := e_1489_9
  simp only [e_1489_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14487 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1489 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14453)) := by
  have c91275 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91276 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91277 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91278 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91279 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91280 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1489_10 := arithEq_of_rows h (row := 1489) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_10
  simp only [← c91275, ← c91276, ← c91277] at e_1489_10
  have e_1489_11 := arithEq_of_rows h (row := 1489) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_11
  simp only [← c91278, ← c91279, k2, ← c91280] at e_1489_11
  have hr := e_1489_11
  simp only [e_1489_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14488 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1489 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14454)) := by
  have c91281 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91282 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91283 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91284 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91285 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91286 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1489_12 := arithEq_of_rows h (row := 1489) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_12
  simp only [← c91281, ← c91282, ← c91283] at e_1489_12
  have e_1489_13 := arithEq_of_rows h (row := 1489) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_13
  simp only [← c91284, ← c91285, k2, ← c91286] at e_1489_13
  have hr := e_1489_13
  simp only [e_1489_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14489 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1489 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14455)) := by
  have c91287 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91288 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91289 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91290 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91291 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91292 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1489_14 := arithEq_of_rows h (row := 1489) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_14
  simp only [← c91287, ← c91288, ← c91289] at e_1489_14
  have e_1489_15 := arithEq_of_rows h (row := 1489) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_15
  simp only [← c91290, ← c91291, k2, ← c91292] at e_1489_15
  have hr := e_1489_15
  simp only [e_1489_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14490 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1489 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14456)) := by
  have c91293 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91294 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91295 := (publicBatchWrapper86_copies2852 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91296 := (publicBatchWrapper86_copies2853 a h).1
  have c91297 := (publicBatchWrapper86_copies2853 a h).2.1
  have c91298 := (publicBatchWrapper86_copies2853 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1489_16 := arithEq_of_rows h (row := 1489) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_16
  simp only [← c91293, ← c91294, ← c91295] at e_1489_16
  have e_1489_17 := arithEq_of_rows h (row := 1489) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_17
  simp only [← c91296, ← c91297, k2, ← c91298] at e_1489_17
  have hr := e_1489_17
  simp only [e_1489_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14491 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1489 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14457)) := by
  have c91299 := (publicBatchWrapper86_copies2853 a h).2.2.2.1
  have c91300 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.1
  have c91301 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.1
  have c91302 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.1
  have c91303 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.1
  have c91304 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1489_18 := arithEq_of_rows h (row := 1489) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_18
  simp only [← c91299, ← c91300, ← c91301] at e_1489_18
  have e_1489_19 := arithEq_of_rows h (row := 1489) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1489_19
  simp only [← c91302, ← c91303, k2, ← c91304] at e_1489_19
  have hr := e_1489_19
  simp only [e_1489_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14492 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1490 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14458)) := by
  have c91305 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.1
  have c91306 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91307 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91308 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91309 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91310 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1490_0 := arithEq_of_rows h (row := 1490) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_0
  simp only [← c91305, ← c91306, ← c91307] at e_1490_0
  have e_1490_1 := arithEq_of_rows h (row := 1490) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_1
  simp only [← c91308, ← c91309, k2, ← c91310] at e_1490_1
  have hr := e_1490_1
  simp only [e_1490_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14493 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1490 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14459)) := by
  have c91311 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91312 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91313 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91314 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91315 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91316 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1490_2 := arithEq_of_rows h (row := 1490) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_2
  simp only [← c91311, ← c91312, ← c91313] at e_1490_2
  have e_1490_3 := arithEq_of_rows h (row := 1490) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_3
  simp only [← c91314, ← c91315, k2, ← c91316] at e_1490_3
  have hr := e_1490_3
  simp only [e_1490_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14494 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1490 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14460)) := by
  have c91317 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91318 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91319 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91320 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91321 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91322 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1490_4 := arithEq_of_rows h (row := 1490) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_4
  simp only [← c91317, ← c91318, ← c91319] at e_1490_4
  have e_1490_5 := arithEq_of_rows h (row := 1490) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_5
  simp only [← c91320, ← c91321, k2, ← c91322] at e_1490_5
  have hr := e_1490_5
  simp only [e_1490_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14495 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1490 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14461)) := by
  have c91323 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91324 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91325 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91326 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91327 := (publicBatchWrapper86_copies2853 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91328 := (publicBatchWrapper86_copies2854 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1490_6 := arithEq_of_rows h (row := 1490) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_6
  simp only [← c91323, ← c91324, ← c91325] at e_1490_6
  have e_1490_7 := arithEq_of_rows h (row := 1490) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_7
  simp only [← c91326, ← c91327, k2, ← c91328] at e_1490_7
  have hr := e_1490_7
  simp only [e_1490_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14496 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1490 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14462)) := by
  have c91329 := (publicBatchWrapper86_copies2854 a h).2.1
  have c91330 := (publicBatchWrapper86_copies2854 a h).2.2.1
  have c91331 := (publicBatchWrapper86_copies2854 a h).2.2.2.1
  have c91332 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.1
  have c91333 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.1
  have c91334 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1490_8 := arithEq_of_rows h (row := 1490) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_8
  simp only [← c91329, ← c91330, ← c91331] at e_1490_8
  have e_1490_9 := arithEq_of_rows h (row := 1490) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_9
  simp only [← c91332, ← c91333, k2, ← c91334] at e_1490_9
  have hr := e_1490_9
  simp only [e_1490_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14497 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1490 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14463)) := by
  have c91335 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.1
  have c91336 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.1
  have c91337 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.1
  have c91338 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91339 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91340 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1490_10 := arithEq_of_rows h (row := 1490) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_10
  simp only [← c91335, ← c91336, ← c91337] at e_1490_10
  have e_1490_11 := arithEq_of_rows h (row := 1490) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_11
  simp only [← c91338, ← c91339, k2, ← c91340] at e_1490_11
  have hr := e_1490_11
  simp only [e_1490_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14498 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1490 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14464)) := by
  have c91341 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91342 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91343 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91344 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91345 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91346 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1490_12 := arithEq_of_rows h (row := 1490) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_12
  simp only [← c91341, ← c91342, ← c91343] at e_1490_12
  have e_1490_13 := arithEq_of_rows h (row := 1490) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_13
  simp only [← c91344, ← c91345, k2, ← c91346] at e_1490_13
  have hr := e_1490_13
  simp only [e_1490_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14499 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1490 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14465)) := by
  have c91347 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91348 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91349 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91350 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91351 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91352 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1490_14 := arithEq_of_rows h (row := 1490) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_14
  simp only [← c91347, ← c91348, ← c91349] at e_1490_14
  have e_1490_15 := arithEq_of_rows h (row := 1490) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_15
  simp only [← c91350, ← c91351, k2, ← c91352] at e_1490_15
  have hr := e_1490_15
  simp only [e_1490_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14500 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1490 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14466)) := by
  have c91353 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91354 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91355 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91356 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91357 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91358 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1490_16 := arithEq_of_rows h (row := 1490) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_16
  simp only [← c91353, ← c91354, ← c91355] at e_1490_16
  have e_1490_17 := arithEq_of_rows h (row := 1490) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_17
  simp only [← c91356, ← c91357, k2, ← c91358] at e_1490_17
  have hr := e_1490_17
  simp only [e_1490_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14501 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1490 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14467)) := by
  have c91359 := (publicBatchWrapper86_copies2854 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91360 := (publicBatchWrapper86_copies2855 a h).1
  have c91361 := (publicBatchWrapper86_copies2855 a h).2.1
  have c91362 := (publicBatchWrapper86_copies2855 a h).2.2.1
  have c91363 := (publicBatchWrapper86_copies2855 a h).2.2.2.1
  have c91364 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1490_18 := arithEq_of_rows h (row := 1490) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_18
  simp only [← c91359, ← c91360, ← c91361] at e_1490_18
  have e_1490_19 := arithEq_of_rows h (row := 1490) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1490_19
  simp only [← c91362, ← c91363, k2, ← c91364] at e_1490_19
  have hr := e_1490_19
  simp only [e_1490_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14502 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1491 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14468)) := by
  have c91365 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.1
  have c91366 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.1
  have c91367 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.1
  have c91368 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.1
  have c91369 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.1
  have c91370 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1491_0 := arithEq_of_rows h (row := 1491) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_0
  simp only [← c91365, ← c91366, ← c91367] at e_1491_0
  have e_1491_1 := arithEq_of_rows h (row := 1491) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_1
  simp only [← c91368, ← c91369, k2, ← c91370] at e_1491_1
  have hr := e_1491_1
  simp only [e_1491_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14503 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1491 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14469)) := by
  have c91371 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91372 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91373 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91374 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91375 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91376 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1491_2 := arithEq_of_rows h (row := 1491) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_2
  simp only [← c91371, ← c91372, ← c91373] at e_1491_2
  have e_1491_3 := arithEq_of_rows h (row := 1491) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_3
  simp only [← c91374, ← c91375, k2, ← c91376] at e_1491_3
  have hr := e_1491_3
  simp only [e_1491_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14504 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1491 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14470)) := by
  have c91377 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91378 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91379 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91380 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91381 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91382 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1491_4 := arithEq_of_rows h (row := 1491) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_4
  simp only [← c91377, ← c91378, ← c91379] at e_1491_4
  have e_1491_5 := arithEq_of_rows h (row := 1491) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_5
  simp only [← c91380, ← c91381, k2, ← c91382] at e_1491_5
  have hr := e_1491_5
  simp only [e_1491_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14505 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1491 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14471)) := by
  have c91383 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91384 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91385 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91386 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91387 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91388 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1491_6 := arithEq_of_rows h (row := 1491) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_6
  simp only [← c91383, ← c91384, ← c91385] at e_1491_6
  have e_1491_7 := arithEq_of_rows h (row := 1491) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_7
  simp only [← c91386, ← c91387, k2, ← c91388] at e_1491_7
  have hr := e_1491_7
  simp only [e_1491_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14506 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1491 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14472)) := by
  have c91389 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91390 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91391 := (publicBatchWrapper86_copies2855 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91392 := (publicBatchWrapper86_copies2856 a h).1
  have c91393 := (publicBatchWrapper86_copies2856 a h).2.1
  have c91394 := (publicBatchWrapper86_copies2856 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1491_8 := arithEq_of_rows h (row := 1491) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_8
  simp only [← c91389, ← c91390, ← c91391] at e_1491_8
  have e_1491_9 := arithEq_of_rows h (row := 1491) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_9
  simp only [← c91392, ← c91393, k2, ← c91394] at e_1491_9
  have hr := e_1491_9
  simp only [e_1491_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14507 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1491 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14473)) := by
  have c91395 := (publicBatchWrapper86_copies2856 a h).2.2.2.1
  have c91396 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.1
  have c91397 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.1
  have c91398 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.1
  have c91399 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.1
  have c91400 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1491_10 := arithEq_of_rows h (row := 1491) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_10
  simp only [← c91395, ← c91396, ← c91397] at e_1491_10
  have e_1491_11 := arithEq_of_rows h (row := 1491) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_11
  simp only [← c91398, ← c91399, k2, ← c91400] at e_1491_11
  have hr := e_1491_11
  simp only [e_1491_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14508 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1491 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14474)) := by
  have c91401 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.1
  have c91402 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91403 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91404 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91405 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91406 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1491_12 := arithEq_of_rows h (row := 1491) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_12
  simp only [← c91401, ← c91402, ← c91403] at e_1491_12
  have e_1491_13 := arithEq_of_rows h (row := 1491) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_13
  simp only [← c91404, ← c91405, k2, ← c91406] at e_1491_13
  have hr := e_1491_13
  simp only [e_1491_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14509 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1491 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14475)) := by
  have c91407 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91408 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91409 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91410 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91411 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91412 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1491_14 := arithEq_of_rows h (row := 1491) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_14
  simp only [← c91407, ← c91408, ← c91409] at e_1491_14
  have e_1491_15 := arithEq_of_rows h (row := 1491) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_15
  simp only [← c91410, ← c91411, k2, ← c91412] at e_1491_15
  have hr := e_1491_15
  simp only [e_1491_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14510 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1491 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14476)) := by
  have c91413 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91414 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91415 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91416 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91417 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91418 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1491_16 := arithEq_of_rows h (row := 1491) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_16
  simp only [← c91413, ← c91414, ← c91415] at e_1491_16
  have e_1491_17 := arithEq_of_rows h (row := 1491) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_17
  simp only [← c91416, ← c91417, k2, ← c91418] at e_1491_17
  have hr := e_1491_17
  simp only [e_1491_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14511 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1491 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14477)) := by
  have c91419 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91420 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91421 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91422 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91423 := (publicBatchWrapper86_copies2856 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91424 := (publicBatchWrapper86_copies2857 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1491_18 := arithEq_of_rows h (row := 1491) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_18
  simp only [← c91419, ← c91420, ← c91421] at e_1491_18
  have e_1491_19 := arithEq_of_rows h (row := 1491) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1491_19
  simp only [← c91422, ← c91423, k2, ← c91424] at e_1491_19
  have hr := e_1491_19
  simp only [e_1491_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14512 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1492 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14478)) := by
  have c91425 := (publicBatchWrapper86_copies2857 a h).2.1
  have c91426 := (publicBatchWrapper86_copies2857 a h).2.2.1
  have c91427 := (publicBatchWrapper86_copies2857 a h).2.2.2.1
  have c91428 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.1
  have c91429 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.1
  have c91430 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1492_0 := arithEq_of_rows h (row := 1492) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_0
  simp only [← c91425, ← c91426, ← c91427] at e_1492_0
  have e_1492_1 := arithEq_of_rows h (row := 1492) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_1
  simp only [← c91428, ← c91429, k2, ← c91430] at e_1492_1
  have hr := e_1492_1
  simp only [e_1492_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14513 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1492 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14479)) := by
  have c91431 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.1
  have c91432 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.1
  have c91433 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.1
  have c91434 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91435 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91436 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1492_2 := arithEq_of_rows h (row := 1492) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_2
  simp only [← c91431, ← c91432, ← c91433] at e_1492_2
  have e_1492_3 := arithEq_of_rows h (row := 1492) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_3
  simp only [← c91434, ← c91435, k2, ← c91436] at e_1492_3
  have hr := e_1492_3
  simp only [e_1492_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14514 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1492 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14480)) := by
  have c91437 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91438 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91439 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91440 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91441 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91442 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1492_4 := arithEq_of_rows h (row := 1492) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_4
  simp only [← c91437, ← c91438, ← c91439] at e_1492_4
  have e_1492_5 := arithEq_of_rows h (row := 1492) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_5
  simp only [← c91440, ← c91441, k2, ← c91442] at e_1492_5
  have hr := e_1492_5
  simp only [e_1492_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14515 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1492 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14481)) := by
  have c91443 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91444 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91445 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91446 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91447 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91448 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1492_6 := arithEq_of_rows h (row := 1492) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_6
  simp only [← c91443, ← c91444, ← c91445] at e_1492_6
  have e_1492_7 := arithEq_of_rows h (row := 1492) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_7
  simp only [← c91446, ← c91447, k2, ← c91448] at e_1492_7
  have hr := e_1492_7
  simp only [e_1492_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14516 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1492 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14482)) := by
  have c91449 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91450 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91451 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91452 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91453 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91454 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1492_8 := arithEq_of_rows h (row := 1492) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_8
  simp only [← c91449, ← c91450, ← c91451] at e_1492_8
  have e_1492_9 := arithEq_of_rows h (row := 1492) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_9
  simp only [← c91452, ← c91453, k2, ← c91454] at e_1492_9
  have hr := e_1492_9
  simp only [e_1492_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14517 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1492 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14483)) := by
  have c91455 := (publicBatchWrapper86_copies2857 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91456 := (publicBatchWrapper86_copies2858 a h).1
  have c91457 := (publicBatchWrapper86_copies2858 a h).2.1
  have c91458 := (publicBatchWrapper86_copies2858 a h).2.2.1
  have c91459 := (publicBatchWrapper86_copies2858 a h).2.2.2.1
  have c91460 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1492_10 := arithEq_of_rows h (row := 1492) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_10
  simp only [← c91455, ← c91456, ← c91457] at e_1492_10
  have e_1492_11 := arithEq_of_rows h (row := 1492) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_11
  simp only [← c91458, ← c91459, k2, ← c91460] at e_1492_11
  have hr := e_1492_11
  simp only [e_1492_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14518 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1492 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14484)) := by
  have c91461 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.1
  have c91462 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.1
  have c91463 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.1
  have c91464 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.1
  have c91465 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.1
  have c91466 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1492_12 := arithEq_of_rows h (row := 1492) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_12
  simp only [← c91461, ← c91462, ← c91463] at e_1492_12
  have e_1492_13 := arithEq_of_rows h (row := 1492) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_13
  simp only [← c91464, ← c91465, k2, ← c91466] at e_1492_13
  have hr := e_1492_13
  simp only [e_1492_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14519 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1492 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14485)) := by
  have c91467 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91468 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91469 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91470 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91471 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91472 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1492_14 := arithEq_of_rows h (row := 1492) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_14
  simp only [← c91467, ← c91468, ← c91469] at e_1492_14
  have e_1492_15 := arithEq_of_rows h (row := 1492) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_15
  simp only [← c91470, ← c91471, k2, ← c91472] at e_1492_15
  have hr := e_1492_15
  simp only [e_1492_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14520 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1492 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14486)) := by
  have c91473 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91474 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91475 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91476 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91477 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91478 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1492_16 := arithEq_of_rows h (row := 1492) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_16
  simp only [← c91473, ← c91474, ← c91475] at e_1492_16
  have e_1492_17 := arithEq_of_rows h (row := 1492) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_17
  simp only [← c91476, ← c91477, k2, ← c91478] at e_1492_17
  have hr := e_1492_17
  simp only [e_1492_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14521 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1492 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14487)) := by
  have c91479 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91480 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91481 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91482 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91483 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91484 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1492_18 := arithEq_of_rows h (row := 1492) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_18
  simp only [← c91479, ← c91480, ← c91481] at e_1492_18
  have e_1492_19 := arithEq_of_rows h (row := 1492) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1492_19
  simp only [← c91482, ← c91483, k2, ← c91484] at e_1492_19
  have hr := e_1492_19
  simp only [e_1492_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14522 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1493 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14488)) := by
  have c91485 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91486 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91487 := (publicBatchWrapper86_copies2858 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91488 := (publicBatchWrapper86_copies2859 a h).1
  have c91489 := (publicBatchWrapper86_copies2859 a h).2.1
  have c91490 := (publicBatchWrapper86_copies2859 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1493_0 := arithEq_of_rows h (row := 1493) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_0
  simp only [← c91485, ← c91486, ← c91487] at e_1493_0
  have e_1493_1 := arithEq_of_rows h (row := 1493) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_1
  simp only [← c91488, ← c91489, k2, ← c91490] at e_1493_1
  have hr := e_1493_1
  simp only [e_1493_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14523 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1493 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14489)) := by
  have c91491 := (publicBatchWrapper86_copies2859 a h).2.2.2.1
  have c91492 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.1
  have c91493 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.1
  have c91494 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.1
  have c91495 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.1
  have c91496 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1493_2 := arithEq_of_rows h (row := 1493) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_2
  simp only [← c91491, ← c91492, ← c91493] at e_1493_2
  have e_1493_3 := arithEq_of_rows h (row := 1493) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_3
  simp only [← c91494, ← c91495, k2, ← c91496] at e_1493_3
  have hr := e_1493_3
  simp only [e_1493_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14524 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1493 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14490)) := by
  have c91497 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.1
  have c91498 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91499 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91500 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91501 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91502 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1493_4 := arithEq_of_rows h (row := 1493) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_4
  simp only [← c91497, ← c91498, ← c91499] at e_1493_4
  have e_1493_5 := arithEq_of_rows h (row := 1493) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_5
  simp only [← c91500, ← c91501, k2, ← c91502] at e_1493_5
  have hr := e_1493_5
  simp only [e_1493_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14525 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1493 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14491)) := by
  have c91503 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91504 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91505 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91506 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91507 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91508 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1493_6 := arithEq_of_rows h (row := 1493) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_6
  simp only [← c91503, ← c91504, ← c91505] at e_1493_6
  have e_1493_7 := arithEq_of_rows h (row := 1493) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_7
  simp only [← c91506, ← c91507, k2, ← c91508] at e_1493_7
  have hr := e_1493_7
  simp only [e_1493_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14526 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1493 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14492)) := by
  have c91509 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91510 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91511 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91512 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91513 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91514 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1493_8 := arithEq_of_rows h (row := 1493) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_8
  simp only [← c91509, ← c91510, ← c91511] at e_1493_8
  have e_1493_9 := arithEq_of_rows h (row := 1493) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_9
  simp only [← c91512, ← c91513, k2, ← c91514] at e_1493_9
  have hr := e_1493_9
  simp only [e_1493_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14527 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1493 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14493)) := by
  have c91515 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91516 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91517 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91518 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91519 := (publicBatchWrapper86_copies2859 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91520 := (publicBatchWrapper86_copies2860 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1493_10 := arithEq_of_rows h (row := 1493) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_10
  simp only [← c91515, ← c91516, ← c91517] at e_1493_10
  have e_1493_11 := arithEq_of_rows h (row := 1493) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_11
  simp only [← c91518, ← c91519, k2, ← c91520] at e_1493_11
  have hr := e_1493_11
  simp only [e_1493_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14528 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1493 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14494)) := by
  have c91521 := (publicBatchWrapper86_copies2860 a h).2.1
  have c91522 := (publicBatchWrapper86_copies2860 a h).2.2.1
  have c91523 := (publicBatchWrapper86_copies2860 a h).2.2.2.1
  have c91524 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.1
  have c91525 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.1
  have c91526 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1493_12 := arithEq_of_rows h (row := 1493) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_12
  simp only [← c91521, ← c91522, ← c91523] at e_1493_12
  have e_1493_13 := arithEq_of_rows h (row := 1493) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_13
  simp only [← c91524, ← c91525, k2, ← c91526] at e_1493_13
  have hr := e_1493_13
  simp only [e_1493_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14529 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1493 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14495)) := by
  have c91527 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.1
  have c91528 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.1
  have c91529 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.1
  have c91530 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91531 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91532 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1493_14 := arithEq_of_rows h (row := 1493) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_14
  simp only [← c91527, ← c91528, ← c91529] at e_1493_14
  have e_1493_15 := arithEq_of_rows h (row := 1493) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_15
  simp only [← c91530, ← c91531, k2, ← c91532] at e_1493_15
  have hr := e_1493_15
  simp only [e_1493_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14530 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1493 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14496)) := by
  have c91533 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91534 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91535 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91536 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91537 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91538 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1493_16 := arithEq_of_rows h (row := 1493) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_16
  simp only [← c91533, ← c91534, ← c91535] at e_1493_16
  have e_1493_17 := arithEq_of_rows h (row := 1493) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_17
  simp only [← c91536, ← c91537, k2, ← c91538] at e_1493_17
  have hr := e_1493_17
  simp only [e_1493_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14531 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1493 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14497)) := by
  have c91539 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91540 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91541 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91542 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91543 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91544 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1493_18 := arithEq_of_rows h (row := 1493) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_18
  simp only [← c91539, ← c91540, ← c91541] at e_1493_18
  have e_1493_19 := arithEq_of_rows h (row := 1493) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1493_19
  simp only [← c91542, ← c91543, k2, ← c91544] at e_1493_19
  have hr := e_1493_19
  simp only [e_1493_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14532 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1494 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14498)) := by
  have c91545 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91546 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91547 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91548 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91549 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91550 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1494_0 := arithEq_of_rows h (row := 1494) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_0
  simp only [← c91545, ← c91546, ← c91547] at e_1494_0
  have e_1494_1 := arithEq_of_rows h (row := 1494) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_1
  simp only [← c91548, ← c91549, k2, ← c91550] at e_1494_1
  have hr := e_1494_1
  simp only [e_1494_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14533 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1494 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14499)) := by
  have c91551 := (publicBatchWrapper86_copies2860 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91552 := (publicBatchWrapper86_copies2861 a h).1
  have c91553 := (publicBatchWrapper86_copies2861 a h).2.1
  have c91554 := (publicBatchWrapper86_copies2861 a h).2.2.1
  have c91555 := (publicBatchWrapper86_copies2861 a h).2.2.2.1
  have c91556 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1494_2 := arithEq_of_rows h (row := 1494) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_2
  simp only [← c91551, ← c91552, ← c91553] at e_1494_2
  have e_1494_3 := arithEq_of_rows h (row := 1494) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_3
  simp only [← c91554, ← c91555, k2, ← c91556] at e_1494_3
  have hr := e_1494_3
  simp only [e_1494_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14534 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1494 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14500)) := by
  have c91557 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.1
  have c91558 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.1
  have c91559 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.1
  have c91560 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.1
  have c91561 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.1
  have c91562 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1494_4 := arithEq_of_rows h (row := 1494) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_4
  simp only [← c91557, ← c91558, ← c91559] at e_1494_4
  have e_1494_5 := arithEq_of_rows h (row := 1494) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_5
  simp only [← c91560, ← c91561, k2, ← c91562] at e_1494_5
  have hr := e_1494_5
  simp only [e_1494_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14535 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1494 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14501)) := by
  have c91563 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91564 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91565 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91566 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91567 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91568 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1494_6 := arithEq_of_rows h (row := 1494) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_6
  simp only [← c91563, ← c91564, ← c91565] at e_1494_6
  have e_1494_7 := arithEq_of_rows h (row := 1494) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_7
  simp only [← c91566, ← c91567, k2, ← c91568] at e_1494_7
  have hr := e_1494_7
  simp only [e_1494_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14536 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1494 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14502)) := by
  have c91569 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91570 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91571 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91572 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91573 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91574 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1494_8 := arithEq_of_rows h (row := 1494) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_8
  simp only [← c91569, ← c91570, ← c91571] at e_1494_8
  have e_1494_9 := arithEq_of_rows h (row := 1494) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_9
  simp only [← c91572, ← c91573, k2, ← c91574] at e_1494_9
  have hr := e_1494_9
  simp only [e_1494_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14537 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1494 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14503)) := by
  have c91575 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91576 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91577 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91578 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91579 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91580 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1494_10 := arithEq_of_rows h (row := 1494) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_10
  simp only [← c91575, ← c91576, ← c91577] at e_1494_10
  have e_1494_11 := arithEq_of_rows h (row := 1494) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_11
  simp only [← c91578, ← c91579, k2, ← c91580] at e_1494_11
  have hr := e_1494_11
  simp only [e_1494_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14538 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1494 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14504)) := by
  have c91581 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91582 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91583 := (publicBatchWrapper86_copies2861 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91584 := (publicBatchWrapper86_copies2862 a h).1
  have c91585 := (publicBatchWrapper86_copies2862 a h).2.1
  have c91586 := (publicBatchWrapper86_copies2862 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1494_12 := arithEq_of_rows h (row := 1494) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_12
  simp only [← c91581, ← c91582, ← c91583] at e_1494_12
  have e_1494_13 := arithEq_of_rows h (row := 1494) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_13
  simp only [← c91584, ← c91585, k2, ← c91586] at e_1494_13
  have hr := e_1494_13
  simp only [e_1494_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14539 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1494 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14505)) := by
  have c91587 := (publicBatchWrapper86_copies2862 a h).2.2.2.1
  have c91588 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.1
  have c91589 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.1
  have c91590 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.1
  have c91591 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.1
  have c91592 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1494_14 := arithEq_of_rows h (row := 1494) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_14
  simp only [← c91587, ← c91588, ← c91589] at e_1494_14
  have e_1494_15 := arithEq_of_rows h (row := 1494) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_15
  simp only [← c91590, ← c91591, k2, ← c91592] at e_1494_15
  have hr := e_1494_15
  simp only [e_1494_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14540 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1494 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14506)) := by
  have c91593 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.1
  have c91594 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91595 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91596 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91597 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91598 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1494_16 := arithEq_of_rows h (row := 1494) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_16
  simp only [← c91593, ← c91594, ← c91595] at e_1494_16
  have e_1494_17 := arithEq_of_rows h (row := 1494) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_17
  simp only [← c91596, ← c91597, k2, ← c91598] at e_1494_17
  have hr := e_1494_17
  simp only [e_1494_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14541 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1494 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14507)) := by
  have c91599 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91600 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91601 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91602 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91603 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91604 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1494_18 := arithEq_of_rows h (row := 1494) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_18
  simp only [← c91599, ← c91600, ← c91601] at e_1494_18
  have e_1494_19 := arithEq_of_rows h (row := 1494) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1494_19
  simp only [← c91602, ← c91603, k2, ← c91604] at e_1494_19
  have hr := e_1494_19
  simp only [e_1494_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14542 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1495 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14508)) := by
  have c91605 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91606 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91607 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91608 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91609 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91610 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1495_0 := arithEq_of_rows h (row := 1495) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_0
  simp only [← c91605, ← c91606, ← c91607] at e_1495_0
  have e_1495_1 := arithEq_of_rows h (row := 1495) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_1
  simp only [← c91608, ← c91609, k2, ← c91610] at e_1495_1
  have hr := e_1495_1
  simp only [e_1495_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14543 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1495 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14509)) := by
  have c91611 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91612 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91613 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91614 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91615 := (publicBatchWrapper86_copies2862 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91616 := (publicBatchWrapper86_copies2863 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1495_2 := arithEq_of_rows h (row := 1495) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_2
  simp only [← c91611, ← c91612, ← c91613] at e_1495_2
  have e_1495_3 := arithEq_of_rows h (row := 1495) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_3
  simp only [← c91614, ← c91615, k2, ← c91616] at e_1495_3
  have hr := e_1495_3
  simp only [e_1495_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14544 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1495 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14510)) := by
  have c91617 := (publicBatchWrapper86_copies2863 a h).2.1
  have c91618 := (publicBatchWrapper86_copies2863 a h).2.2.1
  have c91619 := (publicBatchWrapper86_copies2863 a h).2.2.2.1
  have c91620 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.1
  have c91621 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.1
  have c91622 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1495_4 := arithEq_of_rows h (row := 1495) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_4
  simp only [← c91617, ← c91618, ← c91619] at e_1495_4
  have e_1495_5 := arithEq_of_rows h (row := 1495) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_5
  simp only [← c91620, ← c91621, k2, ← c91622] at e_1495_5
  have hr := e_1495_5
  simp only [e_1495_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14545 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1495 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14511)) := by
  have c91623 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.1
  have c91624 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.1
  have c91625 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.1
  have c91626 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91627 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91628 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1495_6 := arithEq_of_rows h (row := 1495) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_6
  simp only [← c91623, ← c91624, ← c91625] at e_1495_6
  have e_1495_7 := arithEq_of_rows h (row := 1495) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_7
  simp only [← c91626, ← c91627, k2, ← c91628] at e_1495_7
  have hr := e_1495_7
  simp only [e_1495_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14546 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1495 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14512)) := by
  have c91629 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91630 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91631 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91632 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91633 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91634 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1495_8 := arithEq_of_rows h (row := 1495) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_8
  simp only [← c91629, ← c91630, ← c91631] at e_1495_8
  have e_1495_9 := arithEq_of_rows h (row := 1495) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_9
  simp only [← c91632, ← c91633, k2, ← c91634] at e_1495_9
  have hr := e_1495_9
  simp only [e_1495_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14547 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1495 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14513)) := by
  have c91635 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91636 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91637 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91638 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91639 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91640 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1495_10 := arithEq_of_rows h (row := 1495) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_10
  simp only [← c91635, ← c91636, ← c91637] at e_1495_10
  have e_1495_11 := arithEq_of_rows h (row := 1495) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_11
  simp only [← c91638, ← c91639, k2, ← c91640] at e_1495_11
  have hr := e_1495_11
  simp only [e_1495_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14548 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1495 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14514)) := by
  have c91641 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91642 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91643 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91644 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91645 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91646 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1495_12 := arithEq_of_rows h (row := 1495) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_12
  simp only [← c91641, ← c91642, ← c91643] at e_1495_12
  have e_1495_13 := arithEq_of_rows h (row := 1495) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_13
  simp only [← c91644, ← c91645, k2, ← c91646] at e_1495_13
  have hr := e_1495_13
  simp only [e_1495_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14549 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1495 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14515)) := by
  have c91647 := (publicBatchWrapper86_copies2863 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91648 := (publicBatchWrapper86_copies2864 a h).1
  have c91649 := (publicBatchWrapper86_copies2864 a h).2.1
  have c91650 := (publicBatchWrapper86_copies2864 a h).2.2.1
  have c91651 := (publicBatchWrapper86_copies2864 a h).2.2.2.1
  have c91652 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1495_14 := arithEq_of_rows h (row := 1495) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_14
  simp only [← c91647, ← c91648, ← c91649] at e_1495_14
  have e_1495_15 := arithEq_of_rows h (row := 1495) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_15
  simp only [← c91650, ← c91651, k2, ← c91652] at e_1495_15
  have hr := e_1495_15
  simp only [e_1495_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14550 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1495 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14516)) := by
  have c91653 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.1
  have c91654 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.1
  have c91655 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.1
  have c91656 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.1
  have c91657 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.1
  have c91658 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1495_16 := arithEq_of_rows h (row := 1495) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_16
  simp only [← c91653, ← c91654, ← c91655] at e_1495_16
  have e_1495_17 := arithEq_of_rows h (row := 1495) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_17
  simp only [← c91656, ← c91657, k2, ← c91658] at e_1495_17
  have hr := e_1495_17
  simp only [e_1495_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14551 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1495 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14517)) := by
  have c91659 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91660 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91661 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91662 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91663 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91664 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1495_18 := arithEq_of_rows h (row := 1495) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_18
  simp only [← c91659, ← c91660, ← c91661] at e_1495_18
  have e_1495_19 := arithEq_of_rows h (row := 1495) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1495_19
  simp only [← c91662, ← c91663, k2, ← c91664] at e_1495_19
  have hr := e_1495_19
  simp only [e_1495_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14552 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1496 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14518)) := by
  have c91665 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91666 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91667 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91668 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91669 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91670 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1496_0 := arithEq_of_rows h (row := 1496) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_0
  simp only [← c91665, ← c91666, ← c91667] at e_1496_0
  have e_1496_1 := arithEq_of_rows h (row := 1496) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_1
  simp only [← c91668, ← c91669, k2, ← c91670] at e_1496_1
  have hr := e_1496_1
  simp only [e_1496_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14553 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1496 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14519)) := by
  have c91671 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91672 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91673 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91674 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91675 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91676 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1496_2 := arithEq_of_rows h (row := 1496) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_2
  simp only [← c91671, ← c91672, ← c91673] at e_1496_2
  have e_1496_3 := arithEq_of_rows h (row := 1496) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_3
  simp only [← c91674, ← c91675, k2, ← c91676] at e_1496_3
  have hr := e_1496_3
  simp only [e_1496_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14554 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1496 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14520)) := by
  have c91677 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91678 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91679 := (publicBatchWrapper86_copies2864 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91680 := (publicBatchWrapper86_copies2865 a h).1
  have c91681 := (publicBatchWrapper86_copies2865 a h).2.1
  have c91682 := (publicBatchWrapper86_copies2865 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1496_4 := arithEq_of_rows h (row := 1496) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_4
  simp only [← c91677, ← c91678, ← c91679] at e_1496_4
  have e_1496_5 := arithEq_of_rows h (row := 1496) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_5
  simp only [← c91680, ← c91681, k2, ← c91682] at e_1496_5
  have hr := e_1496_5
  simp only [e_1496_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14555 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1496 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14521)) := by
  have c91683 := (publicBatchWrapper86_copies2865 a h).2.2.2.1
  have c91684 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.1
  have c91685 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.1
  have c91686 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.1
  have c91687 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.1
  have c91688 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1496_6 := arithEq_of_rows h (row := 1496) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_6
  simp only [← c91683, ← c91684, ← c91685] at e_1496_6
  have e_1496_7 := arithEq_of_rows h (row := 1496) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_7
  simp only [← c91686, ← c91687, k2, ← c91688] at e_1496_7
  have hr := e_1496_7
  simp only [e_1496_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14556 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1496 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14522)) := by
  have c91689 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.1
  have c91690 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91691 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91692 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91693 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91694 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1496_8 := arithEq_of_rows h (row := 1496) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_8
  simp only [← c91689, ← c91690, ← c91691] at e_1496_8
  have e_1496_9 := arithEq_of_rows h (row := 1496) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_9
  simp only [← c91692, ← c91693, k2, ← c91694] at e_1496_9
  have hr := e_1496_9
  simp only [e_1496_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14557 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1496 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14523)) := by
  have c91695 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91696 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91697 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91698 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91699 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91700 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1496_10 := arithEq_of_rows h (row := 1496) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_10
  simp only [← c91695, ← c91696, ← c91697] at e_1496_10
  have e_1496_11 := arithEq_of_rows h (row := 1496) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_11
  simp only [← c91698, ← c91699, k2, ← c91700] at e_1496_11
  have hr := e_1496_11
  simp only [e_1496_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14558 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1496 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14524)) := by
  have c91701 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91702 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91703 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91704 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91705 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91706 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1496_12 := arithEq_of_rows h (row := 1496) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_12
  simp only [← c91701, ← c91702, ← c91703] at e_1496_12
  have e_1496_13 := arithEq_of_rows h (row := 1496) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_13
  simp only [← c91704, ← c91705, k2, ← c91706] at e_1496_13
  have hr := e_1496_13
  simp only [e_1496_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14559 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1496 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14525)) := by
  have c91707 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91708 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91709 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91710 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91711 := (publicBatchWrapper86_copies2865 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91712 := (publicBatchWrapper86_copies2866 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1496_14 := arithEq_of_rows h (row := 1496) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_14
  simp only [← c91707, ← c91708, ← c91709] at e_1496_14
  have e_1496_15 := arithEq_of_rows h (row := 1496) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_15
  simp only [← c91710, ← c91711, k2, ← c91712] at e_1496_15
  have hr := e_1496_15
  simp only [e_1496_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14560 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1496 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14526)) := by
  have c91713 := (publicBatchWrapper86_copies2866 a h).2.1
  have c91714 := (publicBatchWrapper86_copies2866 a h).2.2.1
  have c91715 := (publicBatchWrapper86_copies2866 a h).2.2.2.1
  have c91716 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.1
  have c91717 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.1
  have c91718 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1496_16 := arithEq_of_rows h (row := 1496) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_16
  simp only [← c91713, ← c91714, ← c91715] at e_1496_16
  have e_1496_17 := arithEq_of_rows h (row := 1496) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_17
  simp only [← c91716, ← c91717, k2, ← c91718] at e_1496_17
  have hr := e_1496_17
  simp only [e_1496_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14561 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1496 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14527)) := by
  have c91719 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.1
  have c91720 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.1
  have c91721 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.1
  have c91722 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91723 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91724 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1496_18 := arithEq_of_rows h (row := 1496) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_18
  simp only [← c91719, ← c91720, ← c91721] at e_1496_18
  have e_1496_19 := arithEq_of_rows h (row := 1496) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1496_19
  simp only [← c91722, ← c91723, k2, ← c91724] at e_1496_19
  have hr := e_1496_19
  simp only [e_1496_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14562 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1497 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14528)) := by
  have c91725 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91726 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91727 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91728 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91729 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91730 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1497_0 := arithEq_of_rows h (row := 1497) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_0
  simp only [← c91725, ← c91726, ← c91727] at e_1497_0
  have e_1497_1 := arithEq_of_rows h (row := 1497) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_1
  simp only [← c91728, ← c91729, k2, ← c91730] at e_1497_1
  have hr := e_1497_1
  simp only [e_1497_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14563 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1497 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14529)) := by
  have c91731 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91732 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91733 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91734 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91735 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91736 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1497_2 := arithEq_of_rows h (row := 1497) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_2
  simp only [← c91731, ← c91732, ← c91733] at e_1497_2
  have e_1497_3 := arithEq_of_rows h (row := 1497) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_3
  simp only [← c91734, ← c91735, k2, ← c91736] at e_1497_3
  have hr := e_1497_3
  simp only [e_1497_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14564 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1497 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14530)) := by
  have c91737 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91738 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91739 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91740 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91741 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91742 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1497_4 := arithEq_of_rows h (row := 1497) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_4
  simp only [← c91737, ← c91738, ← c91739] at e_1497_4
  have e_1497_5 := arithEq_of_rows h (row := 1497) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_5
  simp only [← c91740, ← c91741, k2, ← c91742] at e_1497_5
  have hr := e_1497_5
  simp only [e_1497_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14565 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1497 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14531)) := by
  have c91743 := (publicBatchWrapper86_copies2866 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91744 := (publicBatchWrapper86_copies2867 a h).1
  have c91745 := (publicBatchWrapper86_copies2867 a h).2.1
  have c91746 := (publicBatchWrapper86_copies2867 a h).2.2.1
  have c91747 := (publicBatchWrapper86_copies2867 a h).2.2.2.1
  have c91748 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1497_6 := arithEq_of_rows h (row := 1497) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_6
  simp only [← c91743, ← c91744, ← c91745] at e_1497_6
  have e_1497_7 := arithEq_of_rows h (row := 1497) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_7
  simp only [← c91746, ← c91747, k2, ← c91748] at e_1497_7
  have hr := e_1497_7
  simp only [e_1497_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14566 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1497 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14532)) := by
  have c91749 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.1
  have c91750 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.1
  have c91751 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.1
  have c91752 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.1
  have c91753 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.1
  have c91754 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1497_8 := arithEq_of_rows h (row := 1497) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_8
  simp only [← c91749, ← c91750, ← c91751] at e_1497_8
  have e_1497_9 := arithEq_of_rows h (row := 1497) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_9
  simp only [← c91752, ← c91753, k2, ← c91754] at e_1497_9
  have hr := e_1497_9
  simp only [e_1497_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14567 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1497 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14533)) := by
  have c91755 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91756 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91757 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91758 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91759 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91760 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1497_10 := arithEq_of_rows h (row := 1497) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_10
  simp only [← c91755, ← c91756, ← c91757] at e_1497_10
  have e_1497_11 := arithEq_of_rows h (row := 1497) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_11
  simp only [← c91758, ← c91759, k2, ← c91760] at e_1497_11
  have hr := e_1497_11
  simp only [e_1497_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14568 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1497 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14534)) := by
  have c91761 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91762 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91763 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91764 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91765 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91766 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1497_12 := arithEq_of_rows h (row := 1497) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_12
  simp only [← c91761, ← c91762, ← c91763] at e_1497_12
  have e_1497_13 := arithEq_of_rows h (row := 1497) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_13
  simp only [← c91764, ← c91765, k2, ← c91766] at e_1497_13
  have hr := e_1497_13
  simp only [e_1497_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14569 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1497 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14535)) := by
  have c91767 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91768 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91769 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91770 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91771 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91772 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1497_14 := arithEq_of_rows h (row := 1497) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_14
  simp only [← c91767, ← c91768, ← c91769] at e_1497_14
  have e_1497_15 := arithEq_of_rows h (row := 1497) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_15
  simp only [← c91770, ← c91771, k2, ← c91772] at e_1497_15
  have hr := e_1497_15
  simp only [e_1497_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14570 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1497 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14536)) := by
  have c91773 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91774 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91775 := (publicBatchWrapper86_copies2867 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91776 := (publicBatchWrapper86_copies2868 a h).1
  have c91777 := (publicBatchWrapper86_copies2868 a h).2.1
  have c91778 := (publicBatchWrapper86_copies2868 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1497_16 := arithEq_of_rows h (row := 1497) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_16
  simp only [← c91773, ← c91774, ← c91775] at e_1497_16
  have e_1497_17 := arithEq_of_rows h (row := 1497) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_17
  simp only [← c91776, ← c91777, k2, ← c91778] at e_1497_17
  have hr := e_1497_17
  simp only [e_1497_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14571 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1497 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14537)) := by
  have c91779 := (publicBatchWrapper86_copies2868 a h).2.2.2.1
  have c91780 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.1
  have c91781 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.1
  have c91782 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.1
  have c91783 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.1
  have c91784 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1497_18 := arithEq_of_rows h (row := 1497) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_18
  simp only [← c91779, ← c91780, ← c91781] at e_1497_18
  have e_1497_19 := arithEq_of_rows h (row := 1497) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1497_19
  simp only [← c91782, ← c91783, k2, ← c91784] at e_1497_19
  have hr := e_1497_19
  simp only [e_1497_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14572 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1498 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14538)) := by
  have c91785 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.1
  have c91786 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91787 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91788 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91789 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91790 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1498_0 := arithEq_of_rows h (row := 1498) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_0
  simp only [← c91785, ← c91786, ← c91787] at e_1498_0
  have e_1498_1 := arithEq_of_rows h (row := 1498) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_1
  simp only [← c91788, ← c91789, k2, ← c91790] at e_1498_1
  have hr := e_1498_1
  simp only [e_1498_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14573 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1498 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14539)) := by
  have c91791 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91792 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91793 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91794 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91795 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91796 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1498_2 := arithEq_of_rows h (row := 1498) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_2
  simp only [← c91791, ← c91792, ← c91793] at e_1498_2
  have e_1498_3 := arithEq_of_rows h (row := 1498) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_3
  simp only [← c91794, ← c91795, k2, ← c91796] at e_1498_3
  have hr := e_1498_3
  simp only [e_1498_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14574 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1498 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14540)) := by
  have c91797 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91798 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91799 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91800 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91801 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91802 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1498_4 := arithEq_of_rows h (row := 1498) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_4
  simp only [← c91797, ← c91798, ← c91799] at e_1498_4
  have e_1498_5 := arithEq_of_rows h (row := 1498) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_5
  simp only [← c91800, ← c91801, k2, ← c91802] at e_1498_5
  have hr := e_1498_5
  simp only [e_1498_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14575 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1498 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14541)) := by
  have c91803 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91804 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91805 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91806 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91807 := (publicBatchWrapper86_copies2868 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91808 := (publicBatchWrapper86_copies2869 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1498_6 := arithEq_of_rows h (row := 1498) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_6
  simp only [← c91803, ← c91804, ← c91805] at e_1498_6
  have e_1498_7 := arithEq_of_rows h (row := 1498) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_7
  simp only [← c91806, ← c91807, k2, ← c91808] at e_1498_7
  have hr := e_1498_7
  simp only [e_1498_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14576 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1498 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14542)) := by
  have c91809 := (publicBatchWrapper86_copies2869 a h).2.1
  have c91810 := (publicBatchWrapper86_copies2869 a h).2.2.1
  have c91811 := (publicBatchWrapper86_copies2869 a h).2.2.2.1
  have c91812 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.1
  have c91813 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.1
  have c91814 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1498_8 := arithEq_of_rows h (row := 1498) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_8
  simp only [← c91809, ← c91810, ← c91811] at e_1498_8
  have e_1498_9 := arithEq_of_rows h (row := 1498) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_9
  simp only [← c91812, ← c91813, k2, ← c91814] at e_1498_9
  have hr := e_1498_9
  simp only [e_1498_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14577 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1498 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14543)) := by
  have c91815 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.1
  have c91816 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.1
  have c91817 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.1
  have c91818 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91819 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91820 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1498_10 := arithEq_of_rows h (row := 1498) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_10
  simp only [← c91815, ← c91816, ← c91817] at e_1498_10
  have e_1498_11 := arithEq_of_rows h (row := 1498) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_11
  simp only [← c91818, ← c91819, k2, ← c91820] at e_1498_11
  have hr := e_1498_11
  simp only [e_1498_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14578 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1498 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14544)) := by
  have c91821 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91822 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91823 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91824 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91825 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91826 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1498_12 := arithEq_of_rows h (row := 1498) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_12
  simp only [← c91821, ← c91822, ← c91823] at e_1498_12
  have e_1498_13 := arithEq_of_rows h (row := 1498) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_13
  simp only [← c91824, ← c91825, k2, ← c91826] at e_1498_13
  have hr := e_1498_13
  simp only [e_1498_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14579 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1498 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14545)) := by
  have c91827 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91828 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91829 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91830 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91831 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91832 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1498_14 := arithEq_of_rows h (row := 1498) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_14
  simp only [← c91827, ← c91828, ← c91829] at e_1498_14
  have e_1498_15 := arithEq_of_rows h (row := 1498) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_15
  simp only [← c91830, ← c91831, k2, ← c91832] at e_1498_15
  have hr := e_1498_15
  simp only [e_1498_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14580 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1498 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14546)) := by
  have c91833 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91834 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91835 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91836 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91837 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91838 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1498_16 := arithEq_of_rows h (row := 1498) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_16
  simp only [← c91833, ← c91834, ← c91835] at e_1498_16
  have e_1498_17 := arithEq_of_rows h (row := 1498) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_17
  simp only [← c91836, ← c91837, k2, ← c91838] at e_1498_17
  have hr := e_1498_17
  simp only [e_1498_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14581 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1498 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14547)) := by
  have c91839 := (publicBatchWrapper86_copies2869 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91840 := (publicBatchWrapper86_copies2870 a h).1
  have c91841 := (publicBatchWrapper86_copies2870 a h).2.1
  have c91842 := (publicBatchWrapper86_copies2870 a h).2.2.1
  have c91843 := (publicBatchWrapper86_copies2870 a h).2.2.2.1
  have c91844 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1498_18 := arithEq_of_rows h (row := 1498) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_18
  simp only [← c91839, ← c91840, ← c91841] at e_1498_18
  have e_1498_19 := arithEq_of_rows h (row := 1498) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1498_19
  simp only [← c91842, ← c91843, k2, ← c91844] at e_1498_19
  have hr := e_1498_19
  simp only [e_1498_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14582 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1499 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14548)) := by
  have c91845 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.1
  have c91846 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.1
  have c91847 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.1
  have c91848 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.1
  have c91849 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.1
  have c91850 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1499_0 := arithEq_of_rows h (row := 1499) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_0
  simp only [← c91845, ← c91846, ← c91847] at e_1499_0
  have e_1499_1 := arithEq_of_rows h (row := 1499) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_1
  simp only [← c91848, ← c91849, k2, ← c91850] at e_1499_1
  have hr := e_1499_1
  simp only [e_1499_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14583 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1499 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14549)) := by
  have c91851 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91852 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91853 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91854 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91855 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91856 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1499_2 := arithEq_of_rows h (row := 1499) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_2
  simp only [← c91851, ← c91852, ← c91853] at e_1499_2
  have e_1499_3 := arithEq_of_rows h (row := 1499) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_3
  simp only [← c91854, ← c91855, k2, ← c91856] at e_1499_3
  have hr := e_1499_3
  simp only [e_1499_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14584 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1499 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14550)) := by
  have c91857 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91858 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91859 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91860 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91861 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91862 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1499_4 := arithEq_of_rows h (row := 1499) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_4
  simp only [← c91857, ← c91858, ← c91859] at e_1499_4
  have e_1499_5 := arithEq_of_rows h (row := 1499) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_5
  simp only [← c91860, ← c91861, k2, ← c91862] at e_1499_5
  have hr := e_1499_5
  simp only [e_1499_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14585 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1499 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14551)) := by
  have c91863 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91864 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91865 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91866 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91867 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91868 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1499_6 := arithEq_of_rows h (row := 1499) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_6
  simp only [← c91863, ← c91864, ← c91865] at e_1499_6
  have e_1499_7 := arithEq_of_rows h (row := 1499) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_7
  simp only [← c91866, ← c91867, k2, ← c91868] at e_1499_7
  have hr := e_1499_7
  simp only [e_1499_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14586 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1499 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14552)) := by
  have c91869 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91870 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91871 := (publicBatchWrapper86_copies2870 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91872 := (publicBatchWrapper86_copies2871 a h).1
  have c91873 := (publicBatchWrapper86_copies2871 a h).2.1
  have c91874 := (publicBatchWrapper86_copies2871 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1499_8 := arithEq_of_rows h (row := 1499) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_8
  simp only [← c91869, ← c91870, ← c91871] at e_1499_8
  have e_1499_9 := arithEq_of_rows h (row := 1499) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_9
  simp only [← c91872, ← c91873, k2, ← c91874] at e_1499_9
  have hr := e_1499_9
  simp only [e_1499_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14587 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1499 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14553)) := by
  have c91875 := (publicBatchWrapper86_copies2871 a h).2.2.2.1
  have c91876 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.1
  have c91877 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.1
  have c91878 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.1
  have c91879 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.1
  have c91880 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1499_10 := arithEq_of_rows h (row := 1499) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_10
  simp only [← c91875, ← c91876, ← c91877] at e_1499_10
  have e_1499_11 := arithEq_of_rows h (row := 1499) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_11
  simp only [← c91878, ← c91879, k2, ← c91880] at e_1499_11
  have hr := e_1499_11
  simp only [e_1499_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14588 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1499 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14554)) := by
  have c91881 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.1
  have c91882 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91883 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91884 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91885 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91886 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1499_12 := arithEq_of_rows h (row := 1499) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_12
  simp only [← c91881, ← c91882, ← c91883] at e_1499_12
  have e_1499_13 := arithEq_of_rows h (row := 1499) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_13
  simp only [← c91884, ← c91885, k2, ← c91886] at e_1499_13
  have hr := e_1499_13
  simp only [e_1499_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14589 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1499 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14555)) := by
  have c91887 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91888 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91889 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91890 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91891 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91892 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1499_14 := arithEq_of_rows h (row := 1499) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_14
  simp only [← c91887, ← c91888, ← c91889] at e_1499_14
  have e_1499_15 := arithEq_of_rows h (row := 1499) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_15
  simp only [← c91890, ← c91891, k2, ← c91892] at e_1499_15
  have hr := e_1499_15
  simp only [e_1499_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14590 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1499 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14556)) := by
  have c91893 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91894 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91895 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91896 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91897 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91898 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1499_16 := arithEq_of_rows h (row := 1499) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_16
  simp only [← c91893, ← c91894, ← c91895] at e_1499_16
  have e_1499_17 := arithEq_of_rows h (row := 1499) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_17
  simp only [← c91896, ← c91897, k2, ← c91898] at e_1499_17
  have hr := e_1499_17
  simp only [e_1499_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14591 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1499 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14557)) := by
  have c91899 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91900 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91901 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91902 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91903 := (publicBatchWrapper86_copies2871 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91904 := (publicBatchWrapper86_copies2872 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1499_18 := arithEq_of_rows h (row := 1499) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_18
  simp only [← c91899, ← c91900, ← c91901] at e_1499_18
  have e_1499_19 := arithEq_of_rows h (row := 1499) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1499_19
  simp only [← c91902, ← c91903, k2, ← c91904] at e_1499_19
  have hr := e_1499_19
  simp only [e_1499_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14592 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1500 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14558)) := by
  have c91905 := (publicBatchWrapper86_copies2872 a h).2.1
  have c91906 := (publicBatchWrapper86_copies2872 a h).2.2.1
  have c91907 := (publicBatchWrapper86_copies2872 a h).2.2.2.1
  have c91908 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.1
  have c91909 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.1
  have c91910 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1500_0 := arithEq_of_rows h (row := 1500) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_0
  simp only [← c91905, ← c91906, ← c91907] at e_1500_0
  have e_1500_1 := arithEq_of_rows h (row := 1500) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_1
  simp only [← c91908, ← c91909, k2, ← c91910] at e_1500_1
  have hr := e_1500_1
  simp only [e_1500_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14593 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1500 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14559)) := by
  have c91911 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.1
  have c91912 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.1
  have c91913 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.1
  have c91914 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91915 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91916 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1500_2 := arithEq_of_rows h (row := 1500) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_2
  simp only [← c91911, ← c91912, ← c91913] at e_1500_2
  have e_1500_3 := arithEq_of_rows h (row := 1500) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_3
  simp only [← c91914, ← c91915, k2, ← c91916] at e_1500_3
  have hr := e_1500_3
  simp only [e_1500_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14594 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1500 23) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14560)) := by
  have c91917 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91918 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91919 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91920 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91921 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91922 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1500_4 := arithEq_of_rows h (row := 1500) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_4
  simp only [← c91917, ← c91918, ← c91919] at e_1500_4
  have e_1500_5 := arithEq_of_rows h (row := 1500) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_5
  simp only [← c91920, ← c91921, k2, ← c91922] at e_1500_5
  have hr := e_1500_5
  simp only [e_1500_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14595 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1500 31) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14561)) := by
  have c91923 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91924 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91925 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91926 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91927 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91928 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1500_6 := arithEq_of_rows h (row := 1500) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_6
  simp only [← c91923, ← c91924, ← c91925] at e_1500_6
  have e_1500_7 := arithEq_of_rows h (row := 1500) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_7
  simp only [← c91926, ← c91927, k2, ← c91928] at e_1500_7
  have hr := e_1500_7
  simp only [e_1500_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14596 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1500 39) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14562)) := by
  have c91929 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91930 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91931 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91932 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91933 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91934 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1500_8 := arithEq_of_rows h (row := 1500) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_8
  simp only [← c91929, ← c91930, ← c91931] at e_1500_8
  have e_1500_9 := arithEq_of_rows h (row := 1500) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_9
  simp only [← c91932, ← c91933, k2, ← c91934] at e_1500_9
  have hr := e_1500_9
  simp only [e_1500_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14597 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1500 47) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14563)) := by
  have c91935 := (publicBatchWrapper86_copies2872 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91936 := (publicBatchWrapper86_copies2873 a h).1
  have c91937 := (publicBatchWrapper86_copies2873 a h).2.1
  have c91938 := (publicBatchWrapper86_copies2873 a h).2.2.1
  have c91939 := (publicBatchWrapper86_copies2873 a h).2.2.2.1
  have c91940 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1500_10 := arithEq_of_rows h (row := 1500) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_10
  simp only [← c91935, ← c91936, ← c91937] at e_1500_10
  have e_1500_11 := arithEq_of_rows h (row := 1500) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_11
  simp only [← c91938, ← c91939, k2, ← c91940] at e_1500_11
  have hr := e_1500_11
  simp only [e_1500_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14598 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1500 55) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14564)) := by
  have c91941 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.1
  have c91942 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.1
  have c91943 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.1
  have c91944 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.1
  have c91945 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.1
  have c91946 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1500_12 := arithEq_of_rows h (row := 1500) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_12
  simp only [← c91941, ← c91942, ← c91943] at e_1500_12
  have e_1500_13 := arithEq_of_rows h (row := 1500) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_13
  simp only [← c91944, ← c91945, k2, ← c91946] at e_1500_13
  have hr := e_1500_13
  simp only [e_1500_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14599 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1500 63) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14565)) := by
  have c91947 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91948 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91949 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91950 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91951 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91952 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1500_14 := arithEq_of_rows h (row := 1500) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_14
  simp only [← c91947, ← c91948, ← c91949] at e_1500_14
  have e_1500_15 := arithEq_of_rows h (row := 1500) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_15
  simp only [← c91950, ← c91951, k2, ← c91952] at e_1500_15
  have hr := e_1500_15
  simp only [e_1500_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14600 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1500 71) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14566)) := by
  have c91953 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91954 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91955 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91956 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91957 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91958 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1500_16 := arithEq_of_rows h (row := 1500) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_16
  simp only [← c91953, ← c91954, ← c91955] at e_1500_16
  have e_1500_17 := arithEq_of_rows h (row := 1500) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_17
  simp only [← c91956, ← c91957, k2, ← c91958] at e_1500_17
  have hr := e_1500_17
  simp only [e_1500_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14601 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1500 79) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14567)) := by
  have c91959 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91960 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91961 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91962 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91963 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91964 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1500_18 := arithEq_of_rows h (row := 1500) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_18
  simp only [← c91959, ← c91960, ← c91961] at e_1500_18
  have e_1500_19 := arithEq_of_rows h (row := 1500) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1500_19
  simp only [← c91962, ← c91963, k2, ← c91964] at e_1500_19
  have hr := e_1500_19
  simp only [e_1500_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14602 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1501 7) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14568)) := by
  have c91965 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91966 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91967 := (publicBatchWrapper86_copies2873 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c91968 := (publicBatchWrapper86_copies2874 a h).1
  have c91969 := (publicBatchWrapper86_copies2874 a h).2.1
  have c91970 := (publicBatchWrapper86_copies2874 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1501_0 := arithEq_of_rows h (row := 1501) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_0
  simp only [← c91965, ← c91966, ← c91967] at e_1501_0
  have e_1501_1 := arithEq_of_rows h (row := 1501) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_1
  simp only [← c91968, ← c91969, k2, ← c91970] at e_1501_1
  have hr := e_1501_1
  simp only [e_1501_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14603 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1501 15) = bselect (a (.wire 70 75)) (a (.virt 200)) (a (.virt 14569)) := by
  have c91971 := (publicBatchWrapper86_copies2874 a h).2.2.2.1
  have c91972 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.1
  have c91973 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.1
  have c91974 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.1
  have c91975 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.1
  have c91976 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1501_2 := arithEq_of_rows h (row := 1501) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_2
  simp only [← c91971, ← c91972, ← c91973] at e_1501_2
  have e_1501_3 := arithEq_of_rows h (row := 1501) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_3
  simp only [← c91974, ← c91975, k2, ← c91976] at e_1501_3
  have hr := e_1501_3
  simp only [e_1501_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14604 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1501 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14610)) := by
  have c91977 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.1
  have c91978 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.1
  have c91979 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c91980 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91981 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91982 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1501_4 := arithEq_of_rows h (row := 1501) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_4
  simp only [← c91977, ← c91978, ← c91979] at e_1501_4
  have e_1501_5 := arithEq_of_rows h (row := 1501) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_5
  simp only [← c91980, ← c91981, k2, ← c91982] at e_1501_5
  have hr := e_1501_5
  simp only [e_1501_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14605 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1501 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14611)) := by
  have c91983 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91984 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91985 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91986 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91987 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91988 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1501_6 := arithEq_of_rows h (row := 1501) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_6
  simp only [← c91983, ← c91984, ← c91985] at e_1501_6
  have e_1501_7 := arithEq_of_rows h (row := 1501) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_7
  simp only [← c91986, ← c91987, k2, ← c91988] at e_1501_7
  have hr := e_1501_7
  simp only [e_1501_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14606 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1501 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14612)) := by
  have c91989 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91990 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91991 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91992 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91993 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91994 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1501_8 := arithEq_of_rows h (row := 1501) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_8
  simp only [← c91989, ← c91990, ← c91991] at e_1501_8
  have e_1501_9 := arithEq_of_rows h (row := 1501) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_9
  simp only [← c91992, ← c91993, k2, ← c91994] at e_1501_9
  have hr := e_1501_9
  simp only [e_1501_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14607 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1501 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14613)) := by
  have c91995 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91996 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91997 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91998 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c91999 := (publicBatchWrapper86_copies2874 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92000 := (publicBatchWrapper86_copies2875 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1501_10 := arithEq_of_rows h (row := 1501) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_10
  simp only [← c91995, ← c91996, ← c91997] at e_1501_10
  have e_1501_11 := arithEq_of_rows h (row := 1501) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_11
  simp only [← c91998, ← c91999, k2, ← c92000] at e_1501_11
  have hr := e_1501_11
  simp only [e_1501_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14608 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1501 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14614)) := by
  have c92001 := (publicBatchWrapper86_copies2875 a h).2.1
  have c92002 := (publicBatchWrapper86_copies2875 a h).2.2.1
  have c92003 := (publicBatchWrapper86_copies2875 a h).2.2.2.1
  have c92004 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.1
  have c92005 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.1
  have c92006 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1501_12 := arithEq_of_rows h (row := 1501) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_12
  simp only [← c92001, ← c92002, ← c92003] at e_1501_12
  have e_1501_13 := arithEq_of_rows h (row := 1501) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_13
  simp only [← c92004, ← c92005, k2, ← c92006] at e_1501_13
  have hr := e_1501_13
  simp only [e_1501_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14609 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1501 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14615)) := by
  have c92007 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.1
  have c92008 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.1
  have c92009 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.1
  have c92010 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92011 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92012 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1501_14 := arithEq_of_rows h (row := 1501) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_14
  simp only [← c92007, ← c92008, ← c92009] at e_1501_14
  have e_1501_15 := arithEq_of_rows h (row := 1501) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_15
  simp only [← c92010, ← c92011, k2, ← c92012] at e_1501_15
  have hr := e_1501_15
  simp only [e_1501_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14610 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1501 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14616)) := by
  have c92013 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92014 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92015 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92016 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92017 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92018 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1501_16 := arithEq_of_rows h (row := 1501) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_16
  simp only [← c92013, ← c92014, ← c92015] at e_1501_16
  have e_1501_17 := arithEq_of_rows h (row := 1501) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_17
  simp only [← c92016, ← c92017, k2, ← c92018] at e_1501_17
  have hr := e_1501_17
  simp only [e_1501_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14611 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1501 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14617)) := by
  have c92019 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92020 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92021 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92022 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92023 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92024 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1501_18 := arithEq_of_rows h (row := 1501) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_18
  simp only [← c92019, ← c92020, ← c92021] at e_1501_18
  have e_1501_19 := arithEq_of_rows h (row := 1501) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1501_19
  simp only [← c92022, ← c92023, k2, ← c92024] at e_1501_19
  have hr := e_1501_19
  simp only [e_1501_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14612 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1502 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14618)) := by
  have c92025 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92026 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92027 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92028 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92029 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92030 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1502_0 := arithEq_of_rows h (row := 1502) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_0
  simp only [← c92025, ← c92026, ← c92027] at e_1502_0
  have e_1502_1 := arithEq_of_rows h (row := 1502) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_1
  simp only [← c92028, ← c92029, k2, ← c92030] at e_1502_1
  have hr := e_1502_1
  simp only [e_1502_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14613 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1502 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14619)) := by
  have c92031 := (publicBatchWrapper86_copies2875 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92032 := (publicBatchWrapper86_copies2876 a h).1
  have c92033 := (publicBatchWrapper86_copies2876 a h).2.1
  have c92034 := (publicBatchWrapper86_copies2876 a h).2.2.1
  have c92035 := (publicBatchWrapper86_copies2876 a h).2.2.2.1
  have c92036 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1502_2 := arithEq_of_rows h (row := 1502) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_2
  simp only [← c92031, ← c92032, ← c92033] at e_1502_2
  have e_1502_3 := arithEq_of_rows h (row := 1502) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_3
  simp only [← c92034, ← c92035, k2, ← c92036] at e_1502_3
  have hr := e_1502_3
  simp only [e_1502_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14614 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1502 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14620)) := by
  have c92037 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.1
  have c92038 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.1
  have c92039 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.1
  have c92040 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.1
  have c92041 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.1
  have c92042 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1502_4 := arithEq_of_rows h (row := 1502) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_4
  simp only [← c92037, ← c92038, ← c92039] at e_1502_4
  have e_1502_5 := arithEq_of_rows h (row := 1502) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_5
  simp only [← c92040, ← c92041, k2, ← c92042] at e_1502_5
  have hr := e_1502_5
  simp only [e_1502_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14615 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1502 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14621)) := by
  have c92043 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92044 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92045 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92046 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92047 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92048 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1502_6 := arithEq_of_rows h (row := 1502) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_6
  simp only [← c92043, ← c92044, ← c92045] at e_1502_6
  have e_1502_7 := arithEq_of_rows h (row := 1502) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_7
  simp only [← c92046, ← c92047, k2, ← c92048] at e_1502_7
  have hr := e_1502_7
  simp only [e_1502_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14616 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1502 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14622)) := by
  have c92049 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92050 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92051 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92052 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92053 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92054 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1502_8 := arithEq_of_rows h (row := 1502) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_8
  simp only [← c92049, ← c92050, ← c92051] at e_1502_8
  have e_1502_9 := arithEq_of_rows h (row := 1502) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_9
  simp only [← c92052, ← c92053, k2, ← c92054] at e_1502_9
  have hr := e_1502_9
  simp only [e_1502_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14617 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1502 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14623)) := by
  have c92055 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92056 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92057 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92058 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92059 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92060 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1502_10 := arithEq_of_rows h (row := 1502) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_10
  simp only [← c92055, ← c92056, ← c92057] at e_1502_10
  have e_1502_11 := arithEq_of_rows h (row := 1502) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_11
  simp only [← c92058, ← c92059, k2, ← c92060] at e_1502_11
  have hr := e_1502_11
  simp only [e_1502_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14618 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1502 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14624)) := by
  have c92061 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92062 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92063 := (publicBatchWrapper86_copies2876 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92064 := (publicBatchWrapper86_copies2877 a h).1
  have c92065 := (publicBatchWrapper86_copies2877 a h).2.1
  have c92066 := (publicBatchWrapper86_copies2877 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1502_12 := arithEq_of_rows h (row := 1502) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_12
  simp only [← c92061, ← c92062, ← c92063] at e_1502_12
  have e_1502_13 := arithEq_of_rows h (row := 1502) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_13
  simp only [← c92064, ← c92065, k2, ← c92066] at e_1502_13
  have hr := e_1502_13
  simp only [e_1502_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14619 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1502 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14625)) := by
  have c92067 := (publicBatchWrapper86_copies2877 a h).2.2.2.1
  have c92068 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.1
  have c92069 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.1
  have c92070 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.1
  have c92071 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.1
  have c92072 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1502_14 := arithEq_of_rows h (row := 1502) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_14
  simp only [← c92067, ← c92068, ← c92069] at e_1502_14
  have e_1502_15 := arithEq_of_rows h (row := 1502) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_15
  simp only [← c92070, ← c92071, k2, ← c92072] at e_1502_15
  have hr := e_1502_15
  simp only [e_1502_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14620 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1502 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14626)) := by
  have c92073 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.1
  have c92074 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92075 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92076 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92077 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92078 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1502_16 := arithEq_of_rows h (row := 1502) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_16
  simp only [← c92073, ← c92074, ← c92075] at e_1502_16
  have e_1502_17 := arithEq_of_rows h (row := 1502) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_17
  simp only [← c92076, ← c92077, k2, ← c92078] at e_1502_17
  have hr := e_1502_17
  simp only [e_1502_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14621 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1502 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14627)) := by
  have c92079 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92080 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92081 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92082 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92083 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92084 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1502_18 := arithEq_of_rows h (row := 1502) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_18
  simp only [← c92079, ← c92080, ← c92081] at e_1502_18
  have e_1502_19 := arithEq_of_rows h (row := 1502) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1502_19
  simp only [← c92082, ← c92083, k2, ← c92084] at e_1502_19
  have hr := e_1502_19
  simp only [e_1502_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14622 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1503 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14628)) := by
  have c92085 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92086 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92087 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92088 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92089 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92090 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1503_0 := arithEq_of_rows h (row := 1503) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_0
  simp only [← c92085, ← c92086, ← c92087] at e_1503_0
  have e_1503_1 := arithEq_of_rows h (row := 1503) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_1
  simp only [← c92088, ← c92089, k2, ← c92090] at e_1503_1
  have hr := e_1503_1
  simp only [e_1503_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14623 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1503 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14629)) := by
  have c92091 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92092 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92093 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92094 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92095 := (publicBatchWrapper86_copies2877 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92096 := (publicBatchWrapper86_copies2878 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1503_2 := arithEq_of_rows h (row := 1503) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_2
  simp only [← c92091, ← c92092, ← c92093] at e_1503_2
  have e_1503_3 := arithEq_of_rows h (row := 1503) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_3
  simp only [← c92094, ← c92095, k2, ← c92096] at e_1503_3
  have hr := e_1503_3
  simp only [e_1503_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14624 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1503 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14630)) := by
  have c92097 := (publicBatchWrapper86_copies2878 a h).2.1
  have c92098 := (publicBatchWrapper86_copies2878 a h).2.2.1
  have c92099 := (publicBatchWrapper86_copies2878 a h).2.2.2.1
  have c92100 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.1
  have c92101 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.1
  have c92102 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1503_4 := arithEq_of_rows h (row := 1503) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_4
  simp only [← c92097, ← c92098, ← c92099] at e_1503_4
  have e_1503_5 := arithEq_of_rows h (row := 1503) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_5
  simp only [← c92100, ← c92101, k2, ← c92102] at e_1503_5
  have hr := e_1503_5
  simp only [e_1503_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14625 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1503 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14631)) := by
  have c92103 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.1
  have c92104 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.1
  have c92105 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.1
  have c92106 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92107 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92108 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1503_6 := arithEq_of_rows h (row := 1503) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_6
  simp only [← c92103, ← c92104, ← c92105] at e_1503_6
  have e_1503_7 := arithEq_of_rows h (row := 1503) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_7
  simp only [← c92106, ← c92107, k2, ← c92108] at e_1503_7
  have hr := e_1503_7
  simp only [e_1503_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14626 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1503 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14632)) := by
  have c92109 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92110 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92111 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92112 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92113 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92114 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1503_8 := arithEq_of_rows h (row := 1503) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_8
  simp only [← c92109, ← c92110, ← c92111] at e_1503_8
  have e_1503_9 := arithEq_of_rows h (row := 1503) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_9
  simp only [← c92112, ← c92113, k2, ← c92114] at e_1503_9
  have hr := e_1503_9
  simp only [e_1503_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14627 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1503 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14633)) := by
  have c92115 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92116 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92117 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92118 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92119 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92120 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1503_10 := arithEq_of_rows h (row := 1503) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_10
  simp only [← c92115, ← c92116, ← c92117] at e_1503_10
  have e_1503_11 := arithEq_of_rows h (row := 1503) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_11
  simp only [← c92118, ← c92119, k2, ← c92120] at e_1503_11
  have hr := e_1503_11
  simp only [e_1503_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14628 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1503 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14634)) := by
  have c92121 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92122 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92123 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92124 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92125 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92126 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1503_12 := arithEq_of_rows h (row := 1503) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_12
  simp only [← c92121, ← c92122, ← c92123] at e_1503_12
  have e_1503_13 := arithEq_of_rows h (row := 1503) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_13
  simp only [← c92124, ← c92125, k2, ← c92126] at e_1503_13
  have hr := e_1503_13
  simp only [e_1503_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14629 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1503 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14635)) := by
  have c92127 := (publicBatchWrapper86_copies2878 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92128 := (publicBatchWrapper86_copies2879 a h).1
  have c92129 := (publicBatchWrapper86_copies2879 a h).2.1
  have c92130 := (publicBatchWrapper86_copies2879 a h).2.2.1
  have c92131 := (publicBatchWrapper86_copies2879 a h).2.2.2.1
  have c92132 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1503_14 := arithEq_of_rows h (row := 1503) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_14
  simp only [← c92127, ← c92128, ← c92129] at e_1503_14
  have e_1503_15 := arithEq_of_rows h (row := 1503) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_15
  simp only [← c92130, ← c92131, k2, ← c92132] at e_1503_15
  have hr := e_1503_15
  simp only [e_1503_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14630 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1503 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14636)) := by
  have c92133 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.1
  have c92134 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.1
  have c92135 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.1
  have c92136 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.1
  have c92137 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.1
  have c92138 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1503_16 := arithEq_of_rows h (row := 1503) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_16
  simp only [← c92133, ← c92134, ← c92135] at e_1503_16
  have e_1503_17 := arithEq_of_rows h (row := 1503) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_17
  simp only [← c92136, ← c92137, k2, ← c92138] at e_1503_17
  have hr := e_1503_17
  simp only [e_1503_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14631 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1503 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14637)) := by
  have c92139 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92140 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92141 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92142 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92143 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92144 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1503_18 := arithEq_of_rows h (row := 1503) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_18
  simp only [← c92139, ← c92140, ← c92141] at e_1503_18
  have e_1503_19 := arithEq_of_rows h (row := 1503) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1503_19
  simp only [← c92142, ← c92143, k2, ← c92144] at e_1503_19
  have hr := e_1503_19
  simp only [e_1503_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14632 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1504 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14638)) := by
  have c92145 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92146 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92147 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92148 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92149 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92150 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1504_0 := arithEq_of_rows h (row := 1504) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_0
  simp only [← c92145, ← c92146, ← c92147] at e_1504_0
  have e_1504_1 := arithEq_of_rows h (row := 1504) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_1
  simp only [← c92148, ← c92149, k2, ← c92150] at e_1504_1
  have hr := e_1504_1
  simp only [e_1504_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14633 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1504 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14639)) := by
  have c92151 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92152 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92153 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92154 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92155 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92156 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1504_2 := arithEq_of_rows h (row := 1504) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_2
  simp only [← c92151, ← c92152, ← c92153] at e_1504_2
  have e_1504_3 := arithEq_of_rows h (row := 1504) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_3
  simp only [← c92154, ← c92155, k2, ← c92156] at e_1504_3
  have hr := e_1504_3
  simp only [e_1504_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14634 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1504 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14640)) := by
  have c92157 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92158 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92159 := (publicBatchWrapper86_copies2879 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92160 := (publicBatchWrapper86_copies2880 a h).1
  have c92161 := (publicBatchWrapper86_copies2880 a h).2.1
  have c92162 := (publicBatchWrapper86_copies2880 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1504_4 := arithEq_of_rows h (row := 1504) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_4
  simp only [← c92157, ← c92158, ← c92159] at e_1504_4
  have e_1504_5 := arithEq_of_rows h (row := 1504) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_5
  simp only [← c92160, ← c92161, k2, ← c92162] at e_1504_5
  have hr := e_1504_5
  simp only [e_1504_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14635 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1504 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14641)) := by
  have c92163 := (publicBatchWrapper86_copies2880 a h).2.2.2.1
  have c92164 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.1
  have c92165 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.1
  have c92166 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.1
  have c92167 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.1
  have c92168 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1504_6 := arithEq_of_rows h (row := 1504) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_6
  simp only [← c92163, ← c92164, ← c92165] at e_1504_6
  have e_1504_7 := arithEq_of_rows h (row := 1504) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_7
  simp only [← c92166, ← c92167, k2, ← c92168] at e_1504_7
  have hr := e_1504_7
  simp only [e_1504_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14636 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1504 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14642)) := by
  have c92169 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.1
  have c92170 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92171 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92172 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92173 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92174 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1504_8 := arithEq_of_rows h (row := 1504) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_8
  simp only [← c92169, ← c92170, ← c92171] at e_1504_8
  have e_1504_9 := arithEq_of_rows h (row := 1504) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_9
  simp only [← c92172, ← c92173, k2, ← c92174] at e_1504_9
  have hr := e_1504_9
  simp only [e_1504_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14637 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1504 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14643)) := by
  have c92175 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92176 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92177 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92178 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92179 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92180 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1504_10 := arithEq_of_rows h (row := 1504) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_10
  simp only [← c92175, ← c92176, ← c92177] at e_1504_10
  have e_1504_11 := arithEq_of_rows h (row := 1504) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_11
  simp only [← c92178, ← c92179, k2, ← c92180] at e_1504_11
  have hr := e_1504_11
  simp only [e_1504_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14638 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1504 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14644)) := by
  have c92181 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92182 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92183 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92184 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92185 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92186 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1504_12 := arithEq_of_rows h (row := 1504) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_12
  simp only [← c92181, ← c92182, ← c92183] at e_1504_12
  have e_1504_13 := arithEq_of_rows h (row := 1504) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_13
  simp only [← c92184, ← c92185, k2, ← c92186] at e_1504_13
  have hr := e_1504_13
  simp only [e_1504_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14639 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1504 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14645)) := by
  have c92187 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92188 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92189 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92190 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92191 := (publicBatchWrapper86_copies2880 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92192 := (publicBatchWrapper86_copies2881 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1504_14 := arithEq_of_rows h (row := 1504) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_14
  simp only [← c92187, ← c92188, ← c92189] at e_1504_14
  have e_1504_15 := arithEq_of_rows h (row := 1504) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_15
  simp only [← c92190, ← c92191, k2, ← c92192] at e_1504_15
  have hr := e_1504_15
  simp only [e_1504_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14640 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1504 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14646)) := by
  have c92193 := (publicBatchWrapper86_copies2881 a h).2.1
  have c92194 := (publicBatchWrapper86_copies2881 a h).2.2.1
  have c92195 := (publicBatchWrapper86_copies2881 a h).2.2.2.1
  have c92196 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.1
  have c92197 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.1
  have c92198 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1504_16 := arithEq_of_rows h (row := 1504) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_16
  simp only [← c92193, ← c92194, ← c92195] at e_1504_16
  have e_1504_17 := arithEq_of_rows h (row := 1504) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_17
  simp only [← c92196, ← c92197, k2, ← c92198] at e_1504_17
  have hr := e_1504_17
  simp only [e_1504_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14641 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1504 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14647)) := by
  have c92199 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.1
  have c92200 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.1
  have c92201 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.1
  have c92202 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92203 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92204 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1504_18 := arithEq_of_rows h (row := 1504) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_18
  simp only [← c92199, ← c92200, ← c92201] at e_1504_18
  have e_1504_19 := arithEq_of_rows h (row := 1504) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1504_19
  simp only [← c92202, ← c92203, k2, ← c92204] at e_1504_19
  have hr := e_1504_19
  simp only [e_1504_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14642 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1505 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14648)) := by
  have c92205 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92206 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92207 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92208 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92209 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92210 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1505_0 := arithEq_of_rows h (row := 1505) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_0
  simp only [← c92205, ← c92206, ← c92207] at e_1505_0
  have e_1505_1 := arithEq_of_rows h (row := 1505) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_1
  simp only [← c92208, ← c92209, k2, ← c92210] at e_1505_1
  have hr := e_1505_1
  simp only [e_1505_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14643 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1505 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14649)) := by
  have c92211 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92212 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92213 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92214 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92215 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92216 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1505_2 := arithEq_of_rows h (row := 1505) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_2
  simp only [← c92211, ← c92212, ← c92213] at e_1505_2
  have e_1505_3 := arithEq_of_rows h (row := 1505) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_3
  simp only [← c92214, ← c92215, k2, ← c92216] at e_1505_3
  have hr := e_1505_3
  simp only [e_1505_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14644 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1505 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14650)) := by
  have c92217 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92218 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92219 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92220 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92221 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92222 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1505_4 := arithEq_of_rows h (row := 1505) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_4
  simp only [← c92217, ← c92218, ← c92219] at e_1505_4
  have e_1505_5 := arithEq_of_rows h (row := 1505) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_5
  simp only [← c92220, ← c92221, k2, ← c92222] at e_1505_5
  have hr := e_1505_5
  simp only [e_1505_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14645 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1505 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14651)) := by
  have c92223 := (publicBatchWrapper86_copies2881 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92224 := (publicBatchWrapper86_copies2882 a h).1
  have c92225 := (publicBatchWrapper86_copies2882 a h).2.1
  have c92226 := (publicBatchWrapper86_copies2882 a h).2.2.1
  have c92227 := (publicBatchWrapper86_copies2882 a h).2.2.2.1
  have c92228 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1505_6 := arithEq_of_rows h (row := 1505) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_6
  simp only [← c92223, ← c92224, ← c92225] at e_1505_6
  have e_1505_7 := arithEq_of_rows h (row := 1505) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_7
  simp only [← c92226, ← c92227, k2, ← c92228] at e_1505_7
  have hr := e_1505_7
  simp only [e_1505_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14646 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1505 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14652)) := by
  have c92229 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.1
  have c92230 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.1
  have c92231 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.1
  have c92232 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.1
  have c92233 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.1
  have c92234 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1505_8 := arithEq_of_rows h (row := 1505) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_8
  simp only [← c92229, ← c92230, ← c92231] at e_1505_8
  have e_1505_9 := arithEq_of_rows h (row := 1505) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_9
  simp only [← c92232, ← c92233, k2, ← c92234] at e_1505_9
  have hr := e_1505_9
  simp only [e_1505_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14647 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1505 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14653)) := by
  have c92235 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92236 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92237 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92238 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92239 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92240 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1505_10 := arithEq_of_rows h (row := 1505) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_10
  simp only [← c92235, ← c92236, ← c92237] at e_1505_10
  have e_1505_11 := arithEq_of_rows h (row := 1505) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_11
  simp only [← c92238, ← c92239, k2, ← c92240] at e_1505_11
  have hr := e_1505_11
  simp only [e_1505_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14648 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1505 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14654)) := by
  have c92241 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92242 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92243 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92244 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92245 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92246 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1505_12 := arithEq_of_rows h (row := 1505) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_12
  simp only [← c92241, ← c92242, ← c92243] at e_1505_12
  have e_1505_13 := arithEq_of_rows h (row := 1505) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_13
  simp only [← c92244, ← c92245, k2, ← c92246] at e_1505_13
  have hr := e_1505_13
  simp only [e_1505_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14649 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1505 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14655)) := by
  have c92247 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92248 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92249 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92250 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92251 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92252 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1505_14 := arithEq_of_rows h (row := 1505) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_14
  simp only [← c92247, ← c92248, ← c92249] at e_1505_14
  have e_1505_15 := arithEq_of_rows h (row := 1505) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_15
  simp only [← c92250, ← c92251, k2, ← c92252] at e_1505_15
  have hr := e_1505_15
  simp only [e_1505_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14650 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1505 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14656)) := by
  have c92253 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92254 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92255 := (publicBatchWrapper86_copies2882 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92256 := (publicBatchWrapper86_copies2883 a h).1
  have c92257 := (publicBatchWrapper86_copies2883 a h).2.1
  have c92258 := (publicBatchWrapper86_copies2883 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1505_16 := arithEq_of_rows h (row := 1505) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_16
  simp only [← c92253, ← c92254, ← c92255] at e_1505_16
  have e_1505_17 := arithEq_of_rows h (row := 1505) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_17
  simp only [← c92256, ← c92257, k2, ← c92258] at e_1505_17
  have hr := e_1505_17
  simp only [e_1505_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14651 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1505 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14657)) := by
  have c92259 := (publicBatchWrapper86_copies2883 a h).2.2.2.1
  have c92260 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.1
  have c92261 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.1
  have c92262 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.1
  have c92263 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.1
  have c92264 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1505_18 := arithEq_of_rows h (row := 1505) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_18
  simp only [← c92259, ← c92260, ← c92261] at e_1505_18
  have e_1505_19 := arithEq_of_rows h (row := 1505) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1505_19
  simp only [← c92262, ← c92263, k2, ← c92264] at e_1505_19
  have hr := e_1505_19
  simp only [e_1505_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14652 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1506 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14658)) := by
  have c92265 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.1
  have c92266 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92267 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92268 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92269 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92270 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1506_0 := arithEq_of_rows h (row := 1506) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_0
  simp only [← c92265, ← c92266, ← c92267] at e_1506_0
  have e_1506_1 := arithEq_of_rows h (row := 1506) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_1
  simp only [← c92268, ← c92269, k2, ← c92270] at e_1506_1
  have hr := e_1506_1
  simp only [e_1506_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14653 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1506 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14659)) := by
  have c92271 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92272 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92273 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92274 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92275 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92276 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1506_2 := arithEq_of_rows h (row := 1506) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_2
  simp only [← c92271, ← c92272, ← c92273] at e_1506_2
  have e_1506_3 := arithEq_of_rows h (row := 1506) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_3
  simp only [← c92274, ← c92275, k2, ← c92276] at e_1506_3
  have hr := e_1506_3
  simp only [e_1506_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14654 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1506 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14660)) := by
  have c92277 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92278 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92279 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92280 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92281 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92282 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1506_4 := arithEq_of_rows h (row := 1506) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_4
  simp only [← c92277, ← c92278, ← c92279] at e_1506_4
  have e_1506_5 := arithEq_of_rows h (row := 1506) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_5
  simp only [← c92280, ← c92281, k2, ← c92282] at e_1506_5
  have hr := e_1506_5
  simp only [e_1506_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14655 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1506 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14661)) := by
  have c92283 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92284 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92285 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92286 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92287 := (publicBatchWrapper86_copies2883 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92288 := (publicBatchWrapper86_copies2884 a h).1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1506_6 := arithEq_of_rows h (row := 1506) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_6
  simp only [← c92283, ← c92284, ← c92285] at e_1506_6
  have e_1506_7 := arithEq_of_rows h (row := 1506) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_7
  simp only [← c92286, ← c92287, k2, ← c92288] at e_1506_7
  have hr := e_1506_7
  simp only [e_1506_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14656 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1506 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14662)) := by
  have c92289 := (publicBatchWrapper86_copies2884 a h).2.1
  have c92290 := (publicBatchWrapper86_copies2884 a h).2.2.1
  have c92291 := (publicBatchWrapper86_copies2884 a h).2.2.2.1
  have c92292 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.1
  have c92293 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.1
  have c92294 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1506_8 := arithEq_of_rows h (row := 1506) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_8
  simp only [← c92289, ← c92290, ← c92291] at e_1506_8
  have e_1506_9 := arithEq_of_rows h (row := 1506) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_9
  simp only [← c92292, ← c92293, k2, ← c92294] at e_1506_9
  have hr := e_1506_9
  simp only [e_1506_8] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14657 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1506 47) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14663)) := by
  have c92295 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.1
  have c92296 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.1
  have c92297 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.1
  have c92298 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.1
  have c92299 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92300 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1506_10 := arithEq_of_rows h (row := 1506) (i := 10) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_10
  simp only [← c92295, ← c92296, ← c92297] at e_1506_10
  have e_1506_11 := arithEq_of_rows h (row := 1506) (i := 11) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_11
  simp only [← c92298, ← c92299, k2, ← c92300] at e_1506_11
  have hr := e_1506_11
  simp only [e_1506_10] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14658 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1506 55) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14664)) := by
  have c92301 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92302 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92303 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92304 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92305 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92306 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1506_12 := arithEq_of_rows h (row := 1506) (i := 12) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_12
  simp only [← c92301, ← c92302, ← c92303] at e_1506_12
  have e_1506_13 := arithEq_of_rows h (row := 1506) (i := 13) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_13
  simp only [← c92304, ← c92305, k2, ← c92306] at e_1506_13
  have hr := e_1506_13
  simp only [e_1506_12] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14659 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1506 63) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14665)) := by
  have c92307 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92308 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92309 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92310 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92311 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92312 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1506_14 := arithEq_of_rows h (row := 1506) (i := 14) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_14
  simp only [← c92307, ← c92308, ← c92309] at e_1506_14
  have e_1506_15 := arithEq_of_rows h (row := 1506) (i := 15) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_15
  simp only [← c92310, ← c92311, k2, ← c92312] at e_1506_15
  have hr := e_1506_15
  simp only [e_1506_14] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14660 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1506 71) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14666)) := by
  have c92313 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92314 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92315 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92316 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92317 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92318 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1506_16 := arithEq_of_rows h (row := 1506) (i := 16) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_16
  simp only [← c92313, ← c92314, ← c92315] at e_1506_16
  have e_1506_17 := arithEq_of_rows h (row := 1506) (i := 17) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_17
  simp only [← c92316, ← c92317, k2, ← c92318] at e_1506_17
  have hr := e_1506_17
  simp only [e_1506_16] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14661 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1506 79) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14667)) := by
  have c92319 := (publicBatchWrapper86_copies2884 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92320 := (publicBatchWrapper86_copies2885 a h).1
  have c92321 := (publicBatchWrapper86_copies2885 a h).2.1
  have c92322 := (publicBatchWrapper86_copies2885 a h).2.2.1
  have c92323 := (publicBatchWrapper86_copies2885 a h).2.2.2.1
  have c92324 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1506_18 := arithEq_of_rows h (row := 1506) (i := 18) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_18
  simp only [← c92319, ← c92320, ← c92321] at e_1506_18
  have e_1506_19 := arithEq_of_rows h (row := 1506) (i := 19) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1506_19
  simp only [← c92322, ← c92323, k2, ← c92324] at e_1506_19
  have hr := e_1506_19
  simp only [e_1506_18] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14662 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1507 7) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14668)) := by
  have c92325 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.1
  have c92326 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.1
  have c92327 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.1
  have c92328 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.1
  have c92329 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.1
  have c92330 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1507_0 := arithEq_of_rows h (row := 1507) (i := 0) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_0
  simp only [← c92325, ← c92326, ← c92327] at e_1507_0
  have e_1507_1 := arithEq_of_rows h (row := 1507) (i := 1) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_1
  simp only [← c92328, ← c92329, k2, ← c92330] at e_1507_1
  have hr := e_1507_1
  simp only [e_1507_0] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14663 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1507 15) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14669)) := by
  have c92331 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.1
  have c92332 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92333 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92334 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92335 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92336 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1507_2 := arithEq_of_rows h (row := 1507) (i := 2) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_2
  simp only [← c92331, ← c92332, ← c92333] at e_1507_2
  have e_1507_3 := arithEq_of_rows h (row := 1507) (i := 3) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_3
  simp only [← c92334, ← c92335, k2, ← c92336] at e_1507_3
  have hr := e_1507_3
  simp only [e_1507_2] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14664 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1507 23) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14670)) := by
  have c92337 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92338 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92339 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92340 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92341 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92342 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1507_4 := arithEq_of_rows h (row := 1507) (i := 4) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_4
  simp only [← c92337, ← c92338, ← c92339] at e_1507_4
  have e_1507_5 := arithEq_of_rows h (row := 1507) (i := 5) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_5
  simp only [← c92340, ← c92341, k2, ← c92342] at e_1507_5
  have hr := e_1507_5
  simp only [e_1507_4] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14665 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1507 31) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14671)) := by
  have c92343 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92344 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92345 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92346 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92347 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92348 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1507_6 := arithEq_of_rows h (row := 1507) (i := 6) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_6
  simp only [← c92343, ← c92344, ← c92345] at e_1507_6
  have e_1507_7 := arithEq_of_rows h (row := 1507) (i := 7) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_7
  simp only [← c92346, ← c92347, k2, ← c92348] at e_1507_7
  have hr := e_1507_7
  simp only [e_1507_6] at hr
  simp only [bselect, k2]
  linear_combination hr

theorem publicBatchWrapper86_f14666 (a : Assignment p) (h : Satisfies (publicBatchWrapper86 p) a) :
    a (.wire 1507 39) = bselect (a (.wire 72 39)) (a (.virt 200)) (a (.virt 14672)) := by
  have c92349 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92350 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have c92351 := (publicBatchWrapper86_copies2885 a h).2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  have c92352 := (publicBatchWrapper86_copies2886 a h).1
  have c92353 := (publicBatchWrapper86_copies2886 a h).2.1
  have c92354 := (publicBatchWrapper86_copies2886 a h).2.2.1
  obtain ⟨k0, k1, k2, k3⟩ := publicBatchWrapper86_consts a h
  have e_1507_8 := arithEq_of_rows h (row := 1507) (i := 8) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_8
  simp only [← c92349, ← c92350, ← c92351] at e_1507_8
  have e_1507_9 := arithEq_of_rows h (row := 1507) (i := 9) rfl (by decide)
  simp only [Nat.reduceMul, Nat.reduceAdd] at e_1507_9
  simp only [← c92352, ← c92353, k2, ← c92354] at e_1507_9
  have hr := e_1507_9
  simp only [e_1507_8] at hr
  simp only [bselect, k2]
  linear_combination hr

end Plonky2Spec.Generated
